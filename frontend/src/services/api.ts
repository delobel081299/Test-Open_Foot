import axios from 'axios';

// API base URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    
    // Handle common errors
    if (error.response?.status === 401) {
      // Unauthorized - redirect to login
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    
    return Promise.reject(error);
  }
);

// Types
export interface Video {
  id: number;
  filename: string;
  status: string;
  upload_date: string;
  file_size: number;
  duration: number;
  fps: number;
  resolution: string;
}

export interface UploadResponse {
  id: number;
  filename: string;
  status: string;
  message: string;
}

export interface AnalysisStatus {
  video_id: number;
  status: string;
  progress: number;
  analysis_id?: number;
  completed_at?: string;
}

export interface PlayerScore {
  player_id: number;
  jersey_number?: number;
  team: string;
  position?: string;
  scores: {
    biomechanics: number;
    technical: number;
    tactical: number;
    overall: number;
  };
  metrics: {
    distance_covered: number;
    top_speed: number;
    passes_completed: number;
    pass_accuracy: number;
    shots: number;
    tackles: number;
  };
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
}

export interface AnalysisResults {
  video_id: number;
  analysis_id: number;
  overall_scores: any;
  player_scores: PlayerScore[];
  team_statistics: any;
  key_moments: any[];
  summary: {
    total_players_analyzed: number;
    average_score: number;
    duration_analyzed: number;
    frames_processed: number;
  };
}

export interface TimelineEvent {
  timestamp: number;
  frame_number: number;
  event_type: string;
  description: string;
  players_involved: number[];
  position: [number, number];
  confidence: number;
}

export interface HeatmapData {
  video_id: number;
  player_id: number;
  heatmap: number[][];
  field_dimensions: {
    width: number;
    height: number;
  };
  statistics: {
    average_position: [number, number];
    area_covered: number;
    time_in_thirds: any;
  };
}

export interface Report {
  report_id: number;
  type: string;
  status: string;
  download_url: string;
}

// API Functions

// Upload video
export const uploadVideo = async (
  file: File,
  onProgress?: (progress: number) => void
): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        onProgress(progress);
      }
    },
  });

  return response.data;
};

// Get upload status
export const getUploadStatus = async (videoId: number): Promise<Video> => {
  const response = await api.get(`/upload/status/${videoId}`);
  return response.data;
};

// Start analysis
export const startAnalysis = async (
  videoId: number,
  params?: any
): Promise<{ status: string; message: string }> => {
  const response = await api.post(`/analysis/start/${videoId}`, params);
  return response.data;
};

// Get analysis status
export const getAnalysisStatus = async (videoId: number): Promise<AnalysisStatus> => {
  const response = await api.get(`/analysis/status/${videoId}`);
  return response.data;
};

// Cancel analysis
export const cancelAnalysis = async (videoId: number): Promise<{ message: string }> => {
  const response = await api.post(`/analysis/cancel/${videoId}`);
  return response.data;
};

// Get analysis results
export const getAnalysisResults = async (videoId: number): Promise<AnalysisResults> => {
  const response = await api.get(`/results/${videoId}`);
  return response.data;
};

// Get player results
export const getPlayerResults = async (
  videoId: number,
  team?: string,
  minScore?: number
): Promise<{ video_id: number; players: PlayerScore[] }> => {
  const params = new URLSearchParams();
  if (team) params.append('team', team);
  if (minScore !== undefined) params.append('min_score', minScore.toString());

  const response = await api.get(`/results/${videoId}/players?${params}`);
  return response.data;
};

// Get timeline
export const getTimeline = async (
  videoId: number,
  eventType?: string
): Promise<{ video_id: number; timeline: TimelineEvent[] }> => {
  const params = eventType ? `?event_type=${eventType}` : '';
  const response = await api.get(`/results/${videoId}/timeline${params}`);
  return response.data;
};

// Get player heatmap
export const getPlayerHeatmap = async (
  videoId: number,
  playerId: number
): Promise<HeatmapData> => {
  const response = await api.get(`/results/${videoId}/heatmap/${playerId}`);
  return response.data;
};

// Get comparisons
export const getComparisons = async (videoId: number): Promise<any> => {
  const response = await api.get(`/results/${videoId}/comparisons`);
  return response.data;
};

// Generate report
export const generateReport = async (
  videoId: number,
  reportType: string = 'pdf',
  language: string = 'en'
): Promise<Report> => {
  const response = await api.post(`/reports/generate/${videoId}`, {
    report_type: reportType,
    language,
  });
  return response.data;
};

// Download report
export const downloadReport = async (reportId: number): Promise<Blob> => {
  const response = await api.get(`/reports/download/${reportId}`, {
    responseType: 'blob',
  });
  return response.data;
};

// Generate annotated video
export const generateAnnotatedVideo = async (
  videoId: number,
  options: {
    include_scores?: boolean;
    include_tracking?: boolean;
    include_events?: boolean;
  } = {}
): Promise<Report> => {
  const response = await api.post(`/reports/video/${videoId}/annotate`, options);
  return response.data;
};

// Get report templates
export const getReportTemplates = async (): Promise<any> => {
  const response = await api.get('/reports/templates');
  return response.data;
};

// Share report
export const shareReport = async (
  reportId: number,
  shareOptions: any
): Promise<any> => {
  const response = await api.post(`/reports/share/${reportId}`, shareOptions);
  return response.data;
};

// Health check
export const healthCheck = async (): Promise<any> => {
  const response = await api.get('/health');
  return response.data;
};

export default api;