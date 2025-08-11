import axios from 'axios'
import type { AxiosInstance, AxiosResponse } from 'axios'
import type {
  ApiResponse,
  UploadProgress,
  VideoInfo,
  AnalysisJob,
  AnalysisConfig,
  AnalysisResults,
  ReportJob,
  ReportConfig
} from '@/types'

class ApiService {
  private api: AxiosInstance

  constructor() {
    this.api = axios.create({
      baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
      timeout: 30000,
    })

    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        console.log(`🔄 API Request: ${config.method?.toUpperCase()} ${config.url}`)
        return config
      },
      (error) => {
        console.error('❌ API Request Error:', error)
        return Promise.reject(error)
      }
    )

    // Response interceptor
    this.api.interceptors.response.use(
      (response) => {
        console.log(`✅ API Response: ${response.status} ${response.config.url}`)
        return response
      },
      (error) => {
        console.error('❌ API Response Error:', error.response?.data || error.message)
        return Promise.reject(error)
      }
    )
  }

  // Upload Methods
  async uploadVideo(
    file: File,
    metadata?: Record<string, any>,
    onProgress?: (progress: number) => void
  ): Promise<ApiResponse<{ job_id: string; video_id: number }>> {
    const formData = new FormData()
    formData.append('file', file)
    
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata))
    }

    try {
      const response = await this.api.post('/api/upload', formData, {
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total && onProgress) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
            onProgress(progress)
          }
        },
      })

      return {
        data: response.data,
        status: 'success'
      }
    } catch (error: any) {
      return {
        error: error.response?.data?.detail || error.message,
        status: 'error'
      }
    }
  }

  async getUploadProgress(jobId: string): Promise<ApiResponse<UploadProgress>> {
    try {
      const response = await this.api.get(`/api/progress/${jobId}`)
      return {
        data: response.data,
        status: 'success'
      }
    } catch (error: any) {
      return {
        error: error.response?.data?.detail || error.message,
        status: 'error'
      }
    }
  }

  // Analysis Methods
  async startAnalysis(
    jobId: string,
    config?: Partial<AnalysisConfig>
  ): Promise<ApiResponse<AnalysisJob>> {
    try {
      const response = await this.api.post(`/api/analyze/${jobId}`, {
        config
      })
      return {
        data: response.data,
        status: 'success'
      }
    } catch (error: any) {
      return {
        error: error.response?.data?.detail || error.message,
        status: 'error'
      }
    }
  }

  async getAnalysisStatus(jobId: string): Promise<ApiResponse<AnalysisJob>> {
    try {
      const response = await this.api.get(`/api/status/${jobId}`)
      return {
        data: response.data,
        status: 'success'
      }
    } catch (error: any) {
      return {
        error: error.response?.data?.detail || error.message,
        status: 'error'
      }
    }
  }

  async cancelAnalysis(jobId: string): Promise<ApiResponse<{ message: string }>> {
    try {
      const response = await this.api.post(`/api/cancel/${jobId}`)
      return {
        data: response.data,
        status: 'success'
      }
    } catch (error: any) {
      return {
        error: error.response?.data?.detail || error.message,
        status: 'error'
      }
    }
  }

  // Results Methods
  async getResults(
    jobId: string,
    options?: {
      page?: number
      limit?: number
      player_filter?: string
      team_filter?: string
      min_score?: number
      include_details?: boolean
    }
  ): Promise<ApiResponse<AnalysisResults>> {
    try {
      const params = new URLSearchParams()
      
      if (options?.page) params.append('page', options.page.toString())
      if (options?.limit) params.append('limit', options.limit.toString())
      if (options?.player_filter) params.append('player_filter', options.player_filter)
      if (options?.team_filter) params.append('team_filter', options.team_filter)
      if (options?.min_score) params.append('min_score', options.min_score.toString())
      if (options?.include_details !== undefined) params.append('include_details', options.include_details.toString())

      const queryString = params.toString()
      const url = `/api/results/${jobId}${queryString ? `?${queryString}` : ''}`
      
      const response = await this.api.get(url)
      return {
        data: response.data,
        status: 'success'
      }
    } catch (error: any) {
      return {
        error: error.response?.data?.detail || error.message,
        status: 'error'
      }
    }
  }

  // Report Methods
  async generatePdfReport(
    jobId: string,
    config?: Partial<ReportConfig>
  ): Promise<ApiResponse<ReportJob>> {
    try {
      const params = new URLSearchParams()
      
      if (config?.template) params.append('template', config.template)
      if (config?.language) params.append('language', config.language)
      if (config?.include_charts !== undefined) params.append('include_charts', config.include_charts.toString())
      if (config?.include_heatmaps !== undefined) params.append('include_heatmaps', config.include_heatmaps.toString())

      const queryString = params.toString()
      const url = `/api/report/${jobId}/pdf${queryString ? `?${queryString}` : ''}`
      
      const response = await this.api.get(url)
      return {
        data: response.data,
        status: 'success'
      }
    } catch (error: any) {
      return {
        error: error.response?.data?.detail || error.message,
        status: 'error'
      }
    }
  }

  async getPdfProgress(pdfJobId: string): Promise<ApiResponse<ReportJob>> {
    try {
      const response = await this.api.get(`/api/report/progress/${pdfJobId}`)
      return {
        data: response.data,
        status: 'success'
      }
    } catch (error: any) {
      return {
        error: error.response?.data?.detail || error.message,
        status: 'error'
      }
    }
  }

  async downloadPdfReport(jobId: string, filename?: string): Promise<void> {
    try {
      const response = await this.api.get(`/api/report/${jobId}/pdf`, {
        responseType: 'blob'
      })
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', filename || `football_analysis_${jobId}.pdf`)
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Error downloading PDF:', error)
      throw error
    }
  }

  // SSE (Server-Sent Events) for real-time updates
  createEventSource(jobId: string): EventSource {
    const url = `${this.api.defaults.baseURL}/api/status/${jobId}?sse=true`
    return new EventSource(url)
  }

  // Utility Methods
  async healthCheck(): Promise<ApiResponse<any>> {
    try {
      const response = await this.api.get('/health')
      return {
        data: response.data,
        status: 'success'
      }
    } catch (error: any) {
      return {
        error: error.response?.data?.detail || error.message,
        status: 'error'
      }
    }
  }

  async getApiInfo(): Promise<ApiResponse<any>> {
    try {
      const response = await this.api.get('/api/info')
      return {
        data: response.data,
        status: 'success'
      }
    } catch (error: any) {
      return {
        error: error.response?.data?.detail || error.message,
        status: 'error'
      }
    }
  }

  // Test connection method
  async testConnection(): Promise<ApiResponse<any>> {
    try {
      const response = await this.api.get('/health')
      return {
        data: response.data,
        status: 'success'
      }
    } catch (error: any) {
      return {
        error: error.response?.data?.detail || error.message,
        status: 'error'
      }
    }
  }
}

// Create singleton instance
export const apiService = new ApiService()
export default apiService