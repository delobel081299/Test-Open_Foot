import time
import torch
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import psutil
import GPUtil

from backend.utils.logger import setup_logger
from .advanced_detector import AdvancedDetector, Detection
from .football_postprocessor import EnhancedDetection

logger = setup_logger(__name__)

@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics for football detection"""
    
    # Performance Metrics
    avg_fps: float
    max_fps: float
    min_fps: float
    std_fps: float
    
    # Timing Metrics (milliseconds)
    avg_inference_time: float
    avg_postprocess_time: float
    avg_total_time: float
    
    # Accuracy Metrics
    map_50: float
    map_75: float
    map_50_95: float
    
    # Per-class AP
    ap_player: float
    ap_ball: float
    ap_goal: float
    ap_referee: float
    ap_coach: float
    
    # Detection Statistics
    total_detections: int
    avg_detections_per_frame: float
    detection_consistency: float
    
    # Resource Usage
    avg_gpu_memory_mb: float
    max_gpu_memory_mb: float
    avg_cpu_usage: float
    avg_gpu_utilization: float
    
    # Quality Metrics
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    
    # Target Achievement
    target_fps_achieved: bool
    target_accuracy_achieved: bool
    memory_efficient: bool

@dataclass
class GroundTruthAnnotation:
    """Ground truth annotation for evaluation"""
    frame_id: int
    bbox: Tuple[float, float, float, float]
    class_id: int
    class_name: str
    difficult: bool = False
    crowd: bool = False

class BenchmarkEvaluator:
    """Comprehensive benchmark and evaluation system for football detection"""
    
    def __init__(self, 
                 detector: AdvancedDetector,
                 config: Dict[str, Any]):
        
        self.detector = detector
        self.config = config
        self.benchmark_results = {}
        self.evaluation_history = []
        
        # Performance tracking
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        
        logger.info("Benchmark evaluator initialized")
    
    def run_comprehensive_benchmark(self, 
                                  test_videos: List[str],
                                  ground_truth_path: Optional[str] = None,
                                  output_dir: str = "benchmark_results") -> BenchmarkMetrics:
        """Run comprehensive benchmark evaluation"""
        
        logger.info(f"Starting comprehensive benchmark with {len(test_videos)} videos")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        all_metrics = []
        
        for video_path in test_videos:
            logger.info(f"Benchmarking video: {video_path}")
            
            # Load ground truth if available
            gt_annotations = None
            if ground_truth_path:
                gt_annotations = self._load_ground_truth(ground_truth_path, video_path)
            
            # Run benchmark on single video
            video_metrics = self._benchmark_single_video(
                video_path, gt_annotations, output_path
            )
            all_metrics.append(video_metrics)
        
        # Aggregate metrics
        final_metrics = self._aggregate_metrics(all_metrics)
        
        # Generate comprehensive report
        self._generate_benchmark_report(final_metrics, output_path)
        
        logger.info(f"Benchmark complete. Results saved to {output_path}")
        return final_metrics
    
    def _benchmark_single_video(self, 
                               video_path: str,
                               ground_truth: Optional[List[GroundTruthAnnotation]] = None,
                               output_dir: Path = None) -> Dict[str, Any]:
        """Benchmark detection on a single video"""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps_measurements = []
        inference_times = []
        postprocess_times = []
        memory_usage = []
        cpu_usage = []
        gpu_utilization = []
        
        all_predictions = []
        frame_count = 0
        
        logger.info(f"Processing video: {video_path}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Monitor system resources
                cpu_percent = psutil.cpu_percent()
                cpu_usage.append(cpu_percent)
                
                if self.gpu_available and self.gpu:
                    gpu_util = self.gpu.load * 100
                    gpu_mem = self.gpu.memoryUsed
                    gpu_utilization.append(gpu_util)
                    memory_usage.append(gpu_mem)
                
                # Benchmark detection
                start_time = time.time()
                
                # Detection
                detection_start = time.time()
                detections = self.detector.detect_frame(frame)
                detection_end = time.time()
                
                # Post-processing time is included in detect_frame
                total_time = time.time() - start_time
                
                # Record metrics
                fps = 1.0 / total_time if total_time > 0 else 0
                fps_measurements.append(fps)
                inference_times.append((detection_end - detection_start) * 1000)  # ms
                
                # Store predictions for accuracy evaluation
                frame_predictions = {
                    'frame_id': frame_count,
                    'detections': detections,
                    'timestamp': time.time()
                }
                all_predictions.append(frame_predictions)
                
                frame_count += 1
                
                # Log progress
                if frame_count % 100 == 0:
                    avg_fps = np.mean(fps_measurements[-100:])
                    logger.info(f"Processed {frame_count} frames, avg FPS: {avg_fps:.1f}")
        
        finally:
            cap.release()
        
        # Calculate performance metrics
        performance_metrics = {
            'fps_stats': {
                'avg': np.mean(fps_measurements),
                'max': np.max(fps_measurements),
                'min': np.min(fps_measurements),
                'std': np.std(fps_measurements)
            },
            'timing_stats': {
                'avg_inference_ms': np.mean(inference_times),
                'avg_total_ms': np.mean([1000/fps for fps in fps_measurements if fps > 0])
            },
            'resource_stats': {
                'avg_cpu_usage': np.mean(cpu_usage) if cpu_usage else 0,
                'avg_gpu_utilization': np.mean(gpu_utilization) if gpu_utilization else 0,
                'avg_gpu_memory': np.mean(memory_usage) if memory_usage else 0,
                'max_gpu_memory': np.max(memory_usage) if memory_usage else 0
            },
            'detection_stats': {
                'total_frames': frame_count,
                'total_detections': sum(len(p['detections']) for p in all_predictions),
                'avg_detections_per_frame': sum(len(p['detections']) for p in all_predictions) / frame_count if frame_count > 0 else 0
            }
        }
        
        # Calculate accuracy metrics if ground truth available
        accuracy_metrics = {}
        if ground_truth:
            accuracy_metrics = self._calculate_accuracy_metrics(all_predictions, ground_truth)
        
        return {
            'video_path': video_path,
            'performance': performance_metrics,
            'accuracy': accuracy_metrics,
            'predictions': all_predictions
        }
    
    def _calculate_accuracy_metrics(self, 
                                   predictions: List[Dict],
                                   ground_truth: List[GroundTruthAnnotation]) -> Dict[str, float]:
        """Calculate comprehensive accuracy metrics"""
        
        logger.info("Calculating accuracy metrics...")
        
        # Group ground truth by frame
        gt_by_frame = {}
        for gt in ground_truth:
            if gt.frame_id not in gt_by_frame:
                gt_by_frame[gt.frame_id] = []
            gt_by_frame[gt.frame_id].append(gt)
        
        # Calculate mAP at different IoU thresholds
        map_50 = self._calculate_map(predictions, gt_by_frame, iou_threshold=0.5)
        map_75 = self._calculate_map(predictions, gt_by_frame, iou_threshold=0.75)
        map_50_95 = self._calculate_map_range(predictions, gt_by_frame, iou_range=(0.5, 0.95, 0.05))
        
        # Calculate per-class AP
        class_aps = self._calculate_class_wise_ap(predictions, gt_by_frame)
        
        # Calculate precision, recall, F1
        precision, recall, f1 = self._calculate_precision_recall_f1(predictions, gt_by_frame)
        
        return {
            'mAP@0.5': map_50,
            'mAP@0.75': map_75,
            'mAP@[0.5:0.95]': map_50_95,
            'AP_player': class_aps.get('player', 0.0),
            'AP_ball': class_aps.get('ball', 0.0),
            'AP_goal': class_aps.get('goal', 0.0),
            'AP_referee': class_aps.get('referee', 0.0),
            'AP_coach': class_aps.get('coach', 0.0),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _calculate_map(self, 
                      predictions: List[Dict],
                      ground_truth: Dict[int, List[GroundTruthAnnotation]],
                      iou_threshold: float = 0.5) -> float:
        """Calculate mean Average Precision at specific IoU threshold"""
        
        class_names = ['player', 'ball', 'goal', 'referee', 'coach']
        class_aps = []
        
        for class_name in class_names:
            ap = self._calculate_single_class_ap(
                predictions, ground_truth, class_name, iou_threshold
            )
            class_aps.append(ap)
        
        return np.mean(class_aps) if class_aps else 0.0
    
    def _calculate_single_class_ap(self, 
                                  predictions: List[Dict],
                                  ground_truth: Dict[int, List[GroundTruthAnnotation]],
                                  class_name: str,
                                  iou_threshold: float) -> float:
        """Calculate Average Precision for a single class"""
        
        # Collect all predictions and ground truth for this class
        all_predictions = []
        all_ground_truth = []
        
        for pred_frame in predictions:
            frame_id = pred_frame['frame_id']
            
            # Get predictions for this class
            class_predictions = [
                det for det in pred_frame['detections']
                if det.class_name == class_name
            ]
            
            for det in class_predictions:
                all_predictions.append({
                    'frame_id': frame_id,
                    'bbox': det.bbox,
                    'confidence': det.confidence
                })
            
            # Get ground truth for this class and frame
            if frame_id in ground_truth:
                class_gt = [
                    gt for gt in ground_truth[frame_id]
                    if gt.class_name == class_name and not gt.difficult
                ]
                
                for gt in class_gt:
                    all_ground_truth.append({
                        'frame_id': frame_id,
                        'bbox': gt.bbox,
                        'matched': False
                    })
        
        if not all_predictions or not all_ground_truth:
            return 0.0
        
        # Sort predictions by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate precision and recall at each prediction
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        
        for pred in all_predictions:
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(all_ground_truth):
                if (gt['frame_id'] == pred['frame_id'] and not gt['matched']):
                    iou = self._calculate_bbox_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
            
            # Check if prediction is correct
            if best_iou >= iou_threshold:
                tp += 1
                all_ground_truth[best_gt_idx]['matched'] = True
            else:
                fp += 1
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / len(all_ground_truth) if all_ground_truth else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate Average Precision using interpolation
        ap = self._interpolate_precision_recall(precisions, recalls)
        return ap
    
    def _calculate_bbox_iou(self, 
                           bbox1: Tuple[float, float, float, float],
                           bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two bounding boxes"""
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _interpolate_precision_recall(self, 
                                     precisions: List[float], 
                                     recalls: List[float]) -> float:
        """Calculate AP using 11-point interpolation"""
        
        if not precisions or not recalls:
            return 0.0
        
        # 11-point interpolation
        recall_thresholds = np.linspace(0, 1, 11)
        interpolated_precisions = []
        
        for recall_threshold in recall_thresholds:
            # Find maximum precision for recall >= threshold
            max_precision = 0
            for i, recall in enumerate(recalls):
                if recall >= recall_threshold:
                    max_precision = max(max_precision, precisions[i])
            interpolated_precisions.append(max_precision)
        
        return np.mean(interpolated_precisions)
    
    def _calculate_map_range(self, 
                            predictions: List[Dict],
                            ground_truth: Dict[int, List[GroundTruthAnnotation]],
                            iou_range: Tuple[float, float, float]) -> float:
        """Calculate mAP over a range of IoU thresholds"""
        
        start_iou, end_iou, step = iou_range
        iou_thresholds = np.arange(start_iou, end_iou + step, step)
        
        maps = []
        for iou_thresh in iou_thresholds:
            map_score = self._calculate_map(predictions, ground_truth, iou_thresh)
            maps.append(map_score)
        
        return np.mean(maps) if maps else 0.0
    
    def _calculate_class_wise_ap(self, 
                                predictions: List[Dict],
                                ground_truth: Dict[int, List[GroundTruthAnnotation]]) -> Dict[str, float]:
        """Calculate AP for each class separately"""
        
        class_names = ['player', 'ball', 'goal', 'referee', 'coach']
        class_aps = {}
        
        for class_name in class_names:
            ap = self._calculate_single_class_ap(
                predictions, ground_truth, class_name, iou_threshold=0.5
            )
            class_aps[class_name] = ap
        
        return class_aps
    
    def _calculate_precision_recall_f1(self, 
                                      predictions: List[Dict],
                                      ground_truth: Dict[int, List[GroundTruthAnnotation]]) -> Tuple[float, float, float]:
        """Calculate overall precision, recall, and F1 score"""
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for pred_frame in predictions:
            frame_id = pred_frame['frame_id']
            frame_predictions = pred_frame['detections']
            frame_ground_truth = ground_truth.get(frame_id, [])
            
            # Match predictions to ground truth
            matched_gt = set()
            
            for pred in frame_predictions:
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt in enumerate(frame_ground_truth):
                    if gt.class_name == pred.class_name and i not in matched_gt:
                        iou = self._calculate_bbox_iou(pred.bbox, gt.bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                
                if best_iou >= 0.5:  # IoU threshold
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1
            
            # Count false negatives
            total_fn += len(frame_ground_truth) - len(matched_gt)
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def _load_ground_truth(self, 
                          gt_path: str, 
                          video_path: str) -> List[GroundTruthAnnotation]:
        """Load ground truth annotations"""
        
        # This is a placeholder - actual implementation would depend on annotation format
        # Supports COCO, YOLO, or custom formats
        
        gt_file = Path(gt_path) / f"{Path(video_path).stem}.json"
        
        if not gt_file.exists():
            logger.warning(f"Ground truth file not found: {gt_file}")
            return []
        
        try:
            with open(gt_file, 'r') as f:
                gt_data = json.load(f)
            
            annotations = []
            for ann in gt_data.get('annotations', []):
                annotation = GroundTruthAnnotation(
                    frame_id=ann['frame_id'],
                    bbox=tuple(ann['bbox']),
                    class_id=ann['class_id'],
                    class_name=ann['class_name'],
                    difficult=ann.get('difficult', False),
                    crowd=ann.get('crowd', False)
                )
                annotations.append(annotation)
            
            return annotations
            
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            return []
    
    def _aggregate_metrics(self, video_metrics: List[Dict[str, Any]]) -> BenchmarkMetrics:
        """Aggregate metrics from multiple videos"""
        
        # Aggregate performance metrics
        all_fps = []
        all_inference_times = []
        all_cpu_usage = []
        all_gpu_util = []
        all_gpu_memory = []
        
        # Aggregate accuracy metrics
        all_map_50 = []
        all_map_75 = []
        all_map_50_95 = []
        all_ap_player = []
        all_ap_ball = []
        all_precision = []
        all_recall = []
        all_f1 = []
        
        total_detections = 0
        total_frames = 0
        
        for video_result in video_metrics:
            perf = video_result['performance']
            acc = video_result.get('accuracy', {})
            
            # Performance metrics
            fps_stats = perf['fps_stats']
            all_fps.extend([fps_stats['avg']])  # One avg per video
            all_inference_times.append(perf['timing_stats']['avg_inference_ms'])
            
            if 'resource_stats' in perf:
                all_cpu_usage.append(perf['resource_stats']['avg_cpu_usage'])
                all_gpu_util.append(perf['resource_stats']['avg_gpu_utilization'])
                all_gpu_memory.append(perf['resource_stats']['avg_gpu_memory'])
            
            # Detection stats
            total_detections += perf['detection_stats']['total_detections']
            total_frames += perf['detection_stats']['total_frames']
            
            # Accuracy metrics
            if acc:
                all_map_50.append(acc.get('mAP@0.5', 0))
                all_map_75.append(acc.get('mAP@0.75', 0))
                all_map_50_95.append(acc.get('mAP@[0.5:0.95]', 0))
                all_ap_player.append(acc.get('AP_player', 0))
                all_ap_ball.append(acc.get('AP_ball', 0))
                all_precision.append(acc.get('precision', 0))
                all_recall.append(acc.get('recall', 0))
                all_f1.append(acc.get('f1_score', 0))
        
        # Calculate aggregated metrics
        avg_fps = np.mean(all_fps) if all_fps else 0
        target_fps = self.config.get('performance', {}).get('target_fps', 60)
        target_map = self.config.get('benchmarking', {}).get('performance_targets', {}).get('min_map_50', 0.8)
        max_memory = self.config.get('benchmarking', {}).get('performance_targets', {}).get('max_memory_mb', 4000)
        
        return BenchmarkMetrics(
            # Performance
            avg_fps=avg_fps,
            max_fps=np.max(all_fps) if all_fps else 0,
            min_fps=np.min(all_fps) if all_fps else 0,
            std_fps=np.std(all_fps) if all_fps else 0,
            
            # Timing
            avg_inference_time=np.mean(all_inference_times) if all_inference_times else 0,
            avg_postprocess_time=0,  # Would need to track separately
            avg_total_time=1000/avg_fps if avg_fps > 0 else 0,
            
            # Accuracy
            map_50=np.mean(all_map_50) if all_map_50 else 0,
            map_75=np.mean(all_map_75) if all_map_75 else 0,
            map_50_95=np.mean(all_map_50_95) if all_map_50_95 else 0,
            
            # Per-class AP
            ap_player=np.mean(all_ap_player) if all_ap_player else 0,
            ap_ball=np.mean(all_ap_ball) if all_ap_ball else 0,
            ap_goal=0,  # Would need to calculate
            ap_referee=0,
            ap_coach=0,
            
            # Detection stats
            total_detections=total_detections,
            avg_detections_per_frame=total_detections/total_frames if total_frames > 0 else 0,
            detection_consistency=0,  # Would need temporal analysis
            
            # Resources
            avg_gpu_memory_mb=np.mean(all_gpu_memory) if all_gpu_memory else 0,
            max_gpu_memory_mb=np.max(all_gpu_memory) if all_gpu_memory else 0,
            avg_cpu_usage=np.mean(all_cpu_usage) if all_cpu_usage else 0,
            avg_gpu_utilization=np.mean(all_gpu_util) if all_gpu_util else 0,
            
            # Quality
            precision=np.mean(all_precision) if all_precision else 0,
            recall=np.mean(all_recall) if all_recall else 0,
            f1_score=np.mean(all_f1) if all_f1 else 0,
            false_positive_rate=0,  # Would need to calculate
            
            # Targets
            target_fps_achieved=avg_fps >= target_fps,
            target_accuracy_achieved=np.mean(all_map_50) >= target_map if all_map_50 else False,
            memory_efficient=np.mean(all_gpu_memory) <= max_memory if all_gpu_memory else True
        )
    
    def _generate_benchmark_report(self, metrics: BenchmarkMetrics, output_dir: Path):
        """Generate comprehensive benchmark report"""
        
        # Save metrics as JSON
        metrics_dict = asdict(metrics)
        with open(output_dir / "benchmark_metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Generate visualizations
        self._create_performance_plots(metrics, output_dir)
        
        # Generate markdown report
        self._create_markdown_report(metrics, output_dir)
        
        logger.info(f"Benchmark report generated in {output_dir}")
    
    def _create_performance_plots(self, metrics: BenchmarkMetrics, output_dir: Path):
        """Create performance visualization plots"""
        
        plt.style.use('seaborn-v0_8')
        
        # Performance overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # FPS Performance
        ax1.bar(['Avg FPS', 'Target FPS'], 
                [metrics.avg_fps, self.config.get('performance', {}).get('target_fps', 60)])
        ax1.set_title('FPS Performance')
        ax1.set_ylabel('Frames per Second')
        
        # Accuracy Metrics
        accuracy_metrics = ['mAP@0.5', 'mAP@0.75', 'Precision', 'Recall']
        accuracy_values = [metrics.map_50, metrics.map_75, metrics.precision, metrics.recall]
        ax2.bar(accuracy_metrics, accuracy_values)
        ax2.set_title('Accuracy Metrics')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        
        # Per-class AP
        classes = ['Player', 'Ball', 'Goal', 'Referee', 'Coach']
        class_aps = [metrics.ap_player, metrics.ap_ball, metrics.ap_goal, 
                     metrics.ap_referee, metrics.ap_coach]
        ax3.bar(classes, class_aps)
        ax3.set_title('Per-Class Average Precision')
        ax3.set_ylabel('AP Score')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        # Resource Usage
        resource_metrics = ['GPU Memory (MB)', 'CPU Usage (%)', 'GPU Utilization (%)']
        resource_values = [metrics.avg_gpu_memory_mb, metrics.avg_cpu_usage, 
                          metrics.avg_gpu_utilization]
        ax4.bar(resource_metrics, resource_values)
        ax4.set_title('Resource Usage')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Performance plots created")
    
    def _create_markdown_report(self, metrics: BenchmarkMetrics, output_dir: Path):
        """Create markdown benchmark report"""
        
        report = f"""# Football Detection Benchmark Report

## Executive Summary

This report presents the performance evaluation results for the advanced football detection system.

### Key Metrics
- **Average FPS**: {metrics.avg_fps:.1f}
- **Target FPS Achieved**: {'✅' if metrics.target_fps_achieved else '❌'}
- **Mean Average Precision (mAP@0.5)**: {metrics.map_50:.3f}
- **Target Accuracy Achieved**: {'✅' if metrics.target_accuracy_achieved else '❌'}
- **Memory Efficient**: {'✅' if metrics.memory_efficient else '❌'}

## Performance Metrics

### Speed Performance
- **Average FPS**: {metrics.avg_fps:.2f}
- **Maximum FPS**: {metrics.max_fps:.2f}
- **Minimum FPS**: {metrics.min_fps:.2f}
- **FPS Standard Deviation**: {metrics.std_fps:.2f}

### Timing Breakdown
- **Average Inference Time**: {metrics.avg_inference_time:.1f} ms
- **Average Post-processing Time**: {metrics.avg_postprocess_time:.1f} ms
- **Average Total Time**: {metrics.avg_total_time:.1f} ms

## Accuracy Metrics

### Overall Accuracy
- **mAP@0.5**: {metrics.map_50:.3f}
- **mAP@0.75**: {metrics.map_75:.3f}
- **mAP@[0.5:0.95]**: {metrics.map_50_95:.3f}

### Per-Class Performance
- **Player AP**: {metrics.ap_player:.3f}
- **Ball AP**: {metrics.ap_ball:.3f}
- **Goal AP**: {metrics.ap_goal:.3f}
- **Referee AP**: {metrics.ap_referee:.3f}
- **Coach AP**: {metrics.ap_coach:.3f}

### Classification Metrics
- **Precision**: {metrics.precision:.3f}
- **Recall**: {metrics.recall:.3f}
- **F1 Score**: {metrics.f1_score:.3f}

## Resource Usage

### GPU Performance
- **Average GPU Memory**: {metrics.avg_gpu_memory_mb:.1f} MB
- **Maximum GPU Memory**: {metrics.max_gpu_memory_mb:.1f} MB
- **Average GPU Utilization**: {metrics.avg_gpu_utilization:.1f}%

### CPU Performance
- **Average CPU Usage**: {metrics.avg_cpu_usage:.1f}%

## Detection Statistics

- **Total Detections**: {metrics.total_detections:,}
- **Average Detections per Frame**: {metrics.avg_detections_per_frame:.1f}
- **Detection Consistency**: {metrics.detection_consistency:.3f}

## Target Achievement

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| FPS | {self.config.get('performance', {}).get('target_fps', 60)} | {metrics.avg_fps:.1f} | {'✅' if metrics.target_fps_achieved else '❌'} |
| Accuracy (mAP@0.5) | {self.config.get('benchmarking', {}).get('performance_targets', {}).get('min_map_50', 0.8)} | {metrics.map_50:.3f} | {'✅' if metrics.target_accuracy_achieved else '❌'} |
| Memory Usage | < {self.config.get('benchmarking', {}).get('performance_targets', {}).get('max_memory_mb', 4000)} MB | {metrics.avg_gpu_memory_mb:.1f} MB | {'✅' if metrics.memory_efficient else '❌'} |

## Recommendations

"""
        
        # Add recommendations based on results
        if not metrics.target_fps_achieved:
            report += "- **Performance**: Consider optimizing model architecture or reducing input resolution to achieve target FPS.\n"
        
        if not metrics.target_accuracy_achieved:
            report += "- **Accuracy**: Consider fine-tuning model on football-specific dataset or adjusting confidence thresholds.\n"
        
        if not metrics.memory_efficient:
            report += "- **Memory**: Consider using model quantization or reducing batch size to optimize memory usage.\n"
        
        if metrics.target_fps_achieved and metrics.target_accuracy_achieved and metrics.memory_efficient:
            report += "- **Excellent Performance**: All targets achieved! System is ready for production deployment.\n"
        
        report += f"""
## Configuration Used

```yaml
Primary Model: {self.config.get('primary_model', {}).get('type', 'Unknown')}
Confidence Threshold: {self.config.get('primary_model', {}).get('confidence_threshold', 'Unknown')}
Target FPS: {self.config.get('performance', {}).get('target_fps', 'Unknown')}
Batch Size: {self.config.get('performance', {}).get('batch_size', 'Unknown')}
TensorRT Enabled: {self.config.get('performance', {}).get('gpu_optimization', {}).get('tensorrt_enabled', 'Unknown')}
```

---
*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        with open(output_dir / "benchmark_report.md", 'w') as f:
            f.write(report)
        
        logger.info("Markdown report created")