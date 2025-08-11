// API Response Types
export interface ApiResponse<T> {
  data?: T
  message?: string
  error?: string
  status: 'success' | 'error' | 'loading'
}

// Upload Types
export interface UploadProgress {
  job_id: string
  status: 'uploading' | 'processing' | 'completed' | 'failed' | 'duplicate'
  progress: number
  filename: string
  started_at: number
  elapsed_time?: number
  eta_seconds?: number
  error?: string
}

export interface VideoInfo {
  id: number
  filename: string
  upload_date: string
  file_size: number
  duration: number
  fps: number
  resolution: string
  status: string
}

// Analysis Types
export interface AnalysisJob {
  job_id: string
  video_id: number
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled'
  progress: number
  current_stage: string
  message: string
  started_at: number
  elapsed_seconds?: number
  eta_seconds?: number
  config: AnalysisConfig
}

export interface AnalysisConfig {
  detection_confidence: number
  tracking_max_age: number
  fps: number
  analyze_poses: boolean
  analyze_actions: boolean
  analyze_tactics: boolean
  generate_heatmaps: boolean
  detailed_scoring: boolean
}

// Results Types
export interface AnalysisResults {
  job_id: string
  video_id: number
  analysis_id: number
  status: string
  generated_at: string
  pagination: PaginationInfo
  summary: AnalysisSummary
  overall_scores: OverallScores
  players: PlayerResult[]
  team_statistics?: TeamStatistics
  key_moments?: KeyMoment[]
}

export interface PaginationInfo {
  current_page: number
  total_pages: number
  total_items: number
  items_per_page: number
  has_next: boolean
  has_prev: boolean
}

export interface AnalysisSummary {
  total_players_analyzed: number
  video_duration: number
  frames_processed: number
  tracks_generated: number
  actions_classified: number
  analysis_config: AnalysisConfig
}

export interface OverallScores {
  overall_score: number
  team_cohesion: number
  tactical_efficiency: number
  individual_performance: number
}

// Player Types
export interface PlayerResult {
  player_id: number
  jersey_number: number
  team: 'home' | 'away'
  position: string
  scores: PlayerScores
  metrics?: PlayerMetrics
  performance?: PlayerPerformance
  feedback?: string
}

export interface PlayerScores {
  overall: number
  biomechanics: number
  technical: number
  tactical: number
}

export interface PlayerMetrics {
  distance_covered: number
  top_speed: number
  average_speed: number
  passes_completed: number
  pass_accuracy: number
  shots: number
  tackles: number
  interceptions: number
}

export interface PlayerPerformance {
  strengths: string[]
  weaknesses: string[]
  recommendations: string[]
}

// Team Types
export interface TeamStatistics {
  home_team: TeamStats
  away_team: TeamStats
  comparison: TeamComparison
}

export interface TeamStats {
  name: string
  formation: string
  possession: number
  passes: number
  pass_accuracy: number
  shots: number
  shots_on_target: number
  distance_covered: number
  average_speed: number
}

export interface TeamComparison {
  possession_difference: number
  pass_accuracy_difference: number
  territorial_advantage: number
  pressure_index: number
}

// Event Types
export interface KeyMoment {
  timestamp: number
  type: string
  description: string
  players_involved: number[]
  score_impact: number
  frame_number: number
}

// Video Player Types
export interface VideoAnnotation {
  timestamp: number
  type: 'goal' | 'foul' | 'offside' | 'substitution' | 'card' | 'other'
  title: string
  description: string
  players?: number[]
  position?: { x: number; y: number }
}

export interface VideoPlayerState {
  isPlaying: boolean
  currentTime: number
  duration: number
  playbackRate: number
  volume: number
  isFullscreen: boolean
}

// Chart Types
export interface ChartDataPoint {
  label: string
  value: number
  color?: string
}

export interface RadarChartData {
  labels: string[]
  datasets: {
    label: string
    data: number[]
    backgroundColor?: string
    borderColor?: string
    pointBackgroundColor?: string
  }[]
}

// Report Types
export interface ReportConfig {
  template: 'standard' | 'player_focus' | 'tactical' | 'coach'
  language: 'fr' | 'en'
  include_charts: boolean
  include_heatmaps: boolean
}

export interface ReportJob {
  job_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  stage: string
  started_at: number
  report_id?: number
  file_size?: number
  error?: string
}

// UI Types
export interface Tab {
  id: string
  label: string
  icon?: string
  component?: any
}

export interface FilterOption {
  label: string
  value: string | number
  selected: boolean
}