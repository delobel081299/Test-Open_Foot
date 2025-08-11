import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import math

from backend.utils.logger import setup_logger
from .advanced_detector import Detection, FootballROI, FootballClass

logger = setup_logger(__name__)

class PlayerRole(Enum):
    GOALKEEPER = "goalkeeper"
    DEFENDER = "defender"
    MIDFIELDER = "midfielder"
    FORWARD = "forward"
    REFEREE = "referee"
    COACH = "coach"

@dataclass
class FootballFieldGeometry:
    """Football field geometry for spatial analysis"""
    field_corners: List[Tuple[int, int]]
    goal_posts: List[Tuple[int, int, int, int]]  # x1, y1, x2, y2
    penalty_boxes: List[Tuple[int, int, int, int]]
    center_circle: Tuple[int, int, int]  # x, y, radius
    touchlines: List[Tuple[int, int, int, int]]
    goal_lines: List[Tuple[int, int, int, int]]

@dataclass
class EnhancedDetection(Detection):
    """Enhanced detection with football-specific attributes"""
    player_role: Optional[PlayerRole] = None
    team_id: Optional[int] = None
    jersey_number: Optional[int] = None
    field_position: Optional[Tuple[float, float]] = None  # Normalized field coordinates
    movement_vector: Optional[Tuple[float, float]] = None
    occlusion_level: float = 0.0
    spatial_context: Optional[str] = None

class FootballPostProcessor:
    """Advanced post-processing for football-specific detection refinement"""
    
    def __init__(self, field_geometry: Optional[FootballFieldGeometry] = None):
        self.field_geometry = field_geometry
        self.roi_processor = ROIProcessor()
        self.occlusion_handler = OcclusionHandler()
        self.spatial_analyzer = SpatialAnalyzer(field_geometry)
        self.team_classifier = TeamClassifier()
        
        # Football-specific thresholds
        self.min_player_area = 800
        self.max_player_area = 15000
        self.min_ball_area = 15
        self.max_ball_area = 500
        self.player_aspect_ratio_range = (1.5, 4.0)
        self.ball_aspect_ratio_range = (0.7, 1.4)
        
        logger.info("Football post-processor initialized")
    
    def process_detections(self, 
                          detections: List[Detection], 
                          frame: np.ndarray,
                          frame_idx: int = 0) -> List[EnhancedDetection]:
        """Main post-processing pipeline for football detections"""
        
        enhanced_detections = []
        
        # Convert to enhanced detections
        for det in detections:
            enhanced_det = EnhancedDetection(
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                track_id=det.track_id,
                attention_score=det.attention_score,
                temporal_consistency=det.temporal_consistency
            )
            enhanced_detections.append(enhanced_det)
        
        # Apply football-specific filtering
        enhanced_detections = self._filter_by_size_constraints(enhanced_detections)
        enhanced_detections = self._filter_by_aspect_ratio(enhanced_detections)
        
        # ROI-based filtering
        if self.field_geometry:
            enhanced_detections = self.roi_processor.filter_by_field_roi(
                enhanced_detections, frame, self.field_geometry
            )
        
        # Handle occlusions
        enhanced_detections = self.occlusion_handler.resolve_occlusions(
            enhanced_detections, frame
        )
        
        # Spatial analysis
        enhanced_detections = self.spatial_analyzer.analyze_spatial_context(
            enhanced_detections, frame
        )
        
        # Team classification for players
        enhanced_detections = self.team_classifier.classify_teams(
            enhanced_detections, frame
        )
        
        # Player role classification
        enhanced_detections = self._classify_player_roles(enhanced_detections)
        
        # Validate final detections
        enhanced_detections = self._validate_football_logic(enhanced_detections)
        
        logger.debug(f"Post-processed {len(enhanced_detections)} detections")
        return enhanced_detections
    
    def _filter_by_size_constraints(self, detections: List[EnhancedDetection]) -> List[EnhancedDetection]:
        """Filter detections based on realistic size constraints"""
        filtered = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            area = (x2 - x1) * (y2 - y1)
            
            if det.class_name == "player":
                if self.min_player_area <= area <= self.max_player_area:
                    filtered.append(det)
                else:
                    logger.debug(f"Player filtered by size: area={area}")
            
            elif det.class_name == "ball":
                if self.min_ball_area <= area <= self.max_ball_area:
                    filtered.append(det)
                else:
                    logger.debug(f"Ball filtered by size: area={area}")
            
            else:
                filtered.append(det)  # Keep other classes as-is
        
        return filtered
    
    def _filter_by_aspect_ratio(self, detections: List[EnhancedDetection]) -> List[EnhancedDetection]:
        """Filter detections based on aspect ratio constraints"""
        filtered = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            width = x2 - x1
            height = y2 - y1
            
            if height == 0:
                continue
            
            aspect_ratio = height / width
            
            if det.class_name == "player":
                if self.player_aspect_ratio_range[0] <= aspect_ratio <= self.player_aspect_ratio_range[1]:
                    filtered.append(det)
                else:
                    logger.debug(f"Player filtered by aspect ratio: {aspect_ratio:.2f}")
            
            elif det.class_name == "ball":
                ball_aspect = max(width, height) / min(width, height)
                if self.ball_aspect_ratio_range[0] <= ball_aspect <= self.ball_aspect_ratio_range[1]:
                    filtered.append(det)
                else:
                    logger.debug(f"Ball filtered by aspect ratio: {ball_aspect:.2f}")
            
            else:
                filtered.append(det)
        
        return filtered
    
    def _classify_player_roles(self, detections: List[EnhancedDetection]) -> List[EnhancedDetection]:
        """Classify player roles based on position and context"""
        
        players = [d for d in detections if d.class_name == "player"]
        
        if not players or not self.field_geometry:
            return detections
        
        for player in players:
            # Analyze position relative to goal areas
            role = self._determine_player_role(player)
            player.player_role = role
        
        return detections
    
    def _determine_player_role(self, player: EnhancedDetection) -> PlayerRole:
        """Determine player role based on field position"""
        
        if not player.field_position:
            return PlayerRole.MIDFIELDER  # Default
        
        x, y = player.field_position
        
        # Simple role classification based on field position
        if x < 0.2 or x > 0.8:  # Near goals
            if y < 0.3 or y > 0.7:  # In penalty area
                return PlayerRole.GOALKEEPER
            else:
                return PlayerRole.DEFENDER
        elif x < 0.4 or x > 0.6:  # Defensive/attacking thirds
            return PlayerRole.DEFENDER if x < 0.5 else PlayerRole.FORWARD
        else:  # Middle third
            return PlayerRole.MIDFIELDER
    
    def _validate_football_logic(self, detections: List[EnhancedDetection]) -> List[EnhancedDetection]:
        """Apply football-specific logic validation"""
        
        # Count players per team
        team_counts = {}
        goalkeepers_per_team = {}
        
        for det in detections:
            if det.class_name == "player" and det.team_id is not None:
                team_counts[det.team_id] = team_counts.get(det.team_id, 0) + 1
                
                if det.player_role == PlayerRole.GOALKEEPER:
                    goalkeepers_per_team[det.team_id] = goalkeepers_per_team.get(det.team_id, 0) + 1
        
        # Validate team sizes (should not exceed 11 players each)
        validated = []
        for det in detections:
            if det.class_name == "player":
                if det.team_id and team_counts.get(det.team_id, 0) > 11:
                    # Too many players for this team, reduce confidence
                    det.confidence *= 0.8
                
                # Only one goalkeeper per team
                if (det.player_role == PlayerRole.GOALKEEPER and 
                    det.team_id and goalkeepers_per_team.get(det.team_id, 0) > 1):
                    det.confidence *= 0.5
            
            validated.append(det)
        
        # Only one ball should be detected
        balls = [d for d in validated if d.class_name == "ball"]
        if len(balls) > 1:
            # Keep only the highest confidence ball
            balls.sort(key=lambda x: x.confidence, reverse=True)
            validated = [d for d in validated if d.class_name != "ball"] + [balls[0]]
        
        return validated

class ROIProcessor:
    """Process detections within Region of Interest (football field)"""
    
    def filter_by_field_roi(self, 
                           detections: List[EnhancedDetection], 
                           frame: np.ndarray,
                           field_geometry: FootballFieldGeometry) -> List[EnhancedDetection]:
        """Filter detections to only those within the football field"""
        
        field_mask = self._create_field_mask(frame.shape[:2], field_geometry)
        filtered = []
        
        for det in detections:
            if self._is_detection_in_roi(det, field_mask):
                # Calculate normalized field position
                det.field_position = self._calculate_field_position(det, field_geometry)
                filtered.append(det)
        
        return filtered
    
    def _create_field_mask(self, 
                          frame_shape: Tuple[int, int], 
                          field_geometry: FootballFieldGeometry) -> np.ndarray:
        """Create binary mask for football field area"""
        
        mask = np.zeros(frame_shape, dtype=np.uint8)
        
        if field_geometry.field_corners:
            # Create polygon mask from field corners
            field_polygon = np.array(field_geometry.field_corners, dtype=np.int32)
            cv2.fillPoly(mask, [field_polygon], 255)
        
        return mask
    
    def _is_detection_in_roi(self, detection: EnhancedDetection, field_mask: np.ndarray) -> bool:
        """Check if detection center is within field ROI"""
        
        x1, y1, x2, y2 = detection.bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        if (0 <= center_y < field_mask.shape[0] and 
            0 <= center_x < field_mask.shape[1]):
            return field_mask[center_y, center_x] > 0
        
        return False
    
    def _calculate_field_position(self, 
                                 detection: EnhancedDetection, 
                                 field_geometry: FootballFieldGeometry) -> Tuple[float, float]:
        """Calculate normalized position on football field (0-1)"""
        
        x1, y1, x2, y2 = detection.bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Simple normalization (would need proper homography in real implementation)
        if field_geometry.field_corners:
            corners = field_geometry.field_corners
            min_x = min(c[0] for c in corners)
            max_x = max(c[0] for c in corners)
            min_y = min(c[1] for c in corners)
            max_y = max(c[1] for c in corners)
            
            norm_x = (center_x - min_x) / (max_x - min_x) if max_x > min_x else 0.5
            norm_y = (center_y - min_y) / (max_y - min_y) if max_y > min_y else 0.5
            
            return (max(0, min(1, norm_x)), max(0, min(1, norm_y)))
        
        return (0.5, 0.5)

class OcclusionHandler:
    """Handle overlapping detections and occlusions"""
    
    def resolve_occlusions(self, 
                          detections: List[EnhancedDetection], 
                          frame: np.ndarray) -> List[EnhancedDetection]:
        """Resolve occlusions using advanced algorithms"""
        
        # Group overlapping detections
        overlap_groups = self._find_overlap_groups(detections)
        
        resolved = []
        for group in overlap_groups:
            if len(group) == 1:
                resolved.extend(group)
            else:
                resolved_group = self._resolve_group_occlusion(group, frame)
                resolved.extend(resolved_group)
        
        return resolved
    
    def _find_overlap_groups(self, detections: List[EnhancedDetection]) -> List[List[EnhancedDetection]]:
        """Find groups of overlapping detections"""
        
        groups = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                iou = self._calculate_iou(det1.bbox, det2.bbox)
                if iou > 0.3:  # Overlapping threshold
                    group.append(det2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _resolve_group_occlusion(self, 
                                group: List[EnhancedDetection], 
                                frame: np.ndarray) -> List[EnhancedDetection]:
        """Resolve occlusion within a group of overlapping detections"""
        
        if len(group) <= 1:
            return group
        
        # Sort by confidence and attention score
        group.sort(key=lambda d: (d.confidence, d.attention_score or 0), reverse=True)
        
        resolved = []
        
        # Keep best detection
        best_det = group[0]
        resolved.append(best_det)
        
        # Analyze others for partial occlusion
        for det in group[1:]:
            occlusion_level = self._calculate_occlusion_level(det, best_det)
            det.occlusion_level = occlusion_level
            
            # Keep if different class or low occlusion
            if (det.class_name != best_det.class_name or 
                occlusion_level < 0.7):
                resolved.append(det)
        
        return resolved
    
    def _calculate_iou(self, box1: Tuple[float, float, float, float], 
                      box2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union"""
        
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_occlusion_level(self, 
                                  det1: EnhancedDetection, 
                                  det2: EnhancedDetection) -> float:
        """Calculate how much det1 is occluded by det2"""
        
        x1_1, y1_1, x2_1, y2_1 = det1.bbox
        x1_2, y1_2, x2_2, y2_2 = det2.bbox
        
        # Intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        return intersection / area1 if area1 > 0 else 0.0

class SpatialAnalyzer:
    """Analyze spatial relationships between detections"""
    
    def __init__(self, field_geometry: Optional[FootballFieldGeometry] = None):
        self.field_geometry = field_geometry
    
    def analyze_spatial_context(self, 
                               detections: List[EnhancedDetection], 
                               frame: np.ndarray) -> List[EnhancedDetection]:
        """Analyze spatial context for each detection"""
        
        for det in detections:
            det.spatial_context = self._determine_spatial_context(det, detections)
        
        return detections
    
    def _determine_spatial_context(self, 
                                  detection: EnhancedDetection, 
                                  all_detections: List[EnhancedDetection]) -> str:
        """Determine spatial context (e.g., 'near_goal', 'center_field')"""
        
        if not detection.field_position:
            return "unknown"
        
        x, y = detection.field_position
        
        # Define spatial contexts
        if x < 0.2 or x > 0.8:
            if y < 0.3 or y > 0.7:
                return "penalty_area"
            else:
                return "near_goal"
        elif 0.4 <= x <= 0.6 and 0.4 <= y <= 0.6:
            return "center_field"
        elif x < 0.4:
            return "defensive_third"
        elif x > 0.6:
            return "attacking_third"
        else:
            return "midfield"

class TeamClassifier:
    """Classify players into teams based on jersey colors"""
    
    def __init__(self):
        self.team_colors = {}
        self.color_tolerance = 30
    
    def classify_teams(self, 
                      detections: List[EnhancedDetection], 
                      frame: np.ndarray) -> List[EnhancedDetection]:
        """Classify players into teams based on jersey colors"""
        
        players = [d for d in detections if d.class_name == "player"]
        
        if len(players) < 2:
            return detections
        
        # Extract jersey colors
        jersey_colors = []
        for player in players:
            color = self._extract_jersey_color(player, frame)
            if color is not None:
                jersey_colors.append(color)
            else:
                jersey_colors.append([0, 0, 0])  # Default black
        
        if not jersey_colors:
            return detections
        
        # Cluster colors into teams (simple K-means with k=3: team1, team2, referee)
        team_assignments = self._cluster_colors(jersey_colors, n_clusters=3)
        
        # Assign team IDs
        for i, player in enumerate(players):
            if i < len(team_assignments):
                player.team_id = team_assignments[i]
        
        return detections
    
    def _extract_jersey_color(self, 
                             player: EnhancedDetection, 
                             frame: np.ndarray) -> Optional[List[int]]:
        """Extract dominant jersey color from player bounding box"""
        
        x1, y1, x2, y2 = map(int, player.bbox)
        
        # Ensure bounds are within frame
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract upper torso region (jersey area)
        torso_y1 = y1 + int((y2 - y1) * 0.2)  # Skip head
        torso_y2 = y1 + int((y2 - y1) * 0.7)  # Upper torso
        
        torso_region = frame[torso_y1:torso_y2, x1:x2]
        
        if torso_region.size == 0:
            return None
        
        # Calculate dominant color (simple mean)
        mean_color = np.mean(torso_region.reshape(-1, 3), axis=0)
        return mean_color.astype(int).tolist()
    
    def _cluster_colors(self, colors: List[List[int]], n_clusters: int = 3) -> List[int]:
        """Simple color clustering for team assignment"""
        
        if not colors:
            return []
        
        # Convert to numpy array
        colors_array = np.array(colors)
        
        # Simple K-means clustering (basic implementation)
        from sklearn.cluster import KMeans
        
        try:
            kmeans = KMeans(n_clusters=min(n_clusters, len(colors)), random_state=42)
            team_assignments = kmeans.fit_predict(colors_array)
            return team_assignments.tolist()
        except:
            # Fallback: assign based on color similarity
            return [0] * len(colors)