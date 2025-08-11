import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  UploadProgress,
  AnalysisJob,
  AnalysisResults,
  ReportJob,
  AnalysisConfig
} from '@/types'
import apiService from '@/services/api'

export const useAppStore = defineStore('app', () => {
  // State
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  const currentUpload = ref<UploadProgress | null>(null)
  const currentAnalysis = ref<AnalysisJob | null>(null)
  const analysisResults = ref<AnalysisResults | null>(null)
  const currentReport = ref<ReportJob | null>(null)

  // UI State
  const sidebarOpen = ref(true)
  const darkMode = ref(false)
  const selectedTab = ref('overview')

  // Getters
  const isUploading = computed(() => 
    currentUpload.value?.status === 'uploading'
  )
  
  const isAnalyzing = computed(() => 
    currentAnalysis.value?.status === 'processing'
  )
  
  const hasResults = computed(() => 
    analysisResults.value?.status === 'completed'
  )

  const uploadProgress = computed(() => 
    currentUpload.value?.progress || 0
  )

  const analysisProgress = computed(() => 
    currentAnalysis.value?.progress || 0
  )

  // Actions
  function setError(message: string) {
    error.value = message
    setTimeout(() => {
      error.value = null
    }, 5000)
  }

  function clearError() {
    error.value = null
  }

  async function uploadVideo(file: File, metadata?: Record<string, any>) {
    isLoading.value = true
    error.value = null

    try {
      const response = await apiService.uploadVideo(
        file,
        metadata,
        (progress) => {
          if (currentUpload.value) {
            currentUpload.value.progress = progress
          }
        }
      )

      if (response.status === 'success' && response.data) {
        // Start polling upload progress
        const jobId = response.data.job_id
        pollUploadProgress(jobId)
        return response.data
      } else {
        throw new Error(response.error || 'Upload failed')
      }
    } catch (err: any) {
      setError(err.message)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function pollUploadProgress(jobId: string) {
    const poll = async () => {
      try {
        const response = await apiService.getUploadProgress(jobId)
        if (response.status === 'success' && response.data) {
          currentUpload.value = response.data
          
          if (['completed', 'failed', 'duplicate'].includes(response.data.status)) {
            return // Stop polling
          }
        }
      } catch (err) {
        console.error('Error polling upload progress:', err)
      }

      // Continue polling every 1 second
      setTimeout(poll, 1000)
    }

    poll()
  }

  async function startAnalysis(jobId: string, config?: Partial<AnalysisConfig>) {
    isLoading.value = true
    error.value = null

    try {
      const response = await apiService.startAnalysis(jobId, config)
      
      if (response.status === 'success' && response.data) {
        currentAnalysis.value = response.data
        
        // Start polling analysis progress
        pollAnalysisProgress(response.data.job_id)
        return response.data
      } else {
        throw new Error(response.error || 'Analysis start failed')
      }
    } catch (err: any) {
      setError(err.message)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function pollAnalysisProgress(jobId: string) {
    const poll = async () => {
      try {
        const response = await apiService.getAnalysisStatus(jobId)
        if (response.status === 'success' && response.data) {
          currentAnalysis.value = response.data
          
          if (['completed', 'failed', 'cancelled'].includes(response.data.status)) {
            if (response.data.status === 'completed') {
              // Load results automatically
              await loadResults(jobId)
            }
            return // Stop polling
          }
        }
      } catch (err) {
        console.error('Error polling analysis progress:', err)
      }

      // Continue polling every 2 seconds
      setTimeout(poll, 2000)
    }

    poll()
  }

  async function cancelAnalysis(jobId: string) {
    try {
      const response = await apiService.cancelAnalysis(jobId)
      if (response.status === 'success') {
        if (currentAnalysis.value) {
          currentAnalysis.value.status = 'cancelled'
        }
      } else {
        throw new Error(response.error || 'Cancellation failed')
      }
    } catch (err: any) {
      setError(err.message)
      throw err
    }
  }

  async function loadResults(
    jobId: string,
    options?: {
      page?: number
      limit?: number
      player_filter?: string
      team_filter?: string
      min_score?: number
      include_details?: boolean
    }
  ) {
    isLoading.value = true
    error.value = null

    try {
      const response = await apiService.getResults(jobId, options)
      
      if (response.status === 'success' && response.data) {
        analysisResults.value = response.data
        return response.data
      } else {
        throw new Error(response.error || 'Failed to load results')
      }
    } catch (err: any) {
      setError(err.message)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function generateReport(jobId: string, config?: any) {
    isLoading.value = true
    error.value = null

    try {
      const response = await apiService.generatePdfReport(jobId, config)
      
      if (response.status === 'success' && response.data) {
        currentReport.value = response.data
        
        if (response.data.status === 'started') {
          // Start polling report progress
          pollReportProgress(response.data.job_id)
        }
        
        return response.data
      } else {
        throw new Error(response.error || 'Report generation failed')
      }
    } catch (err: any) {
      setError(err.message)
      throw err
    } finally {
      isLoading.value = false
    }
  }

  async function pollReportProgress(pdfJobId: string) {
    const poll = async () => {
      try {
        const response = await apiService.getPdfProgress(pdfJobId)
        if (response.status === 'success' && response.data) {
          currentReport.value = response.data
          
          if (['completed', 'failed'].includes(response.data.status)) {
            return // Stop polling
          }
        }
      } catch (err) {
        console.error('Error polling report progress:', err)
      }

      // Continue polling every 2 seconds
      setTimeout(poll, 2000)
    }

    poll()
  }

  async function downloadReport(jobId: string, filename?: string) {
    try {
      await apiService.downloadPdfReport(jobId, filename)
    } catch (err: any) {
      setError(err.message)
      throw err
    }
  }

  // UI Actions
  function toggleSidebar() {
    sidebarOpen.value = !sidebarOpen.value
  }

  function toggleDarkMode() {
    darkMode.value = !darkMode.value
    // Apply to HTML element
    if (darkMode.value) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }

  function setSelectedTab(tab: string) {
    selectedTab.value = tab
  }

  // Reset functions
  function resetUpload() {
    currentUpload.value = null
  }

  function resetAnalysis() {
    currentAnalysis.value = null
  }

  function resetResults() {
    analysisResults.value = null
  }

  function resetAll() {
    resetUpload()
    resetAnalysis()
    resetResults()
    currentReport.value = null
    error.value = null
    isLoading.value = false
  }

  return {
    // State
    isLoading,
    error,
    currentUpload,
    currentAnalysis,
    analysisResults,
    currentReport,
    sidebarOpen,
    darkMode,
    selectedTab,
    
    // Getters
    isUploading,
    isAnalyzing,
    hasResults,
    uploadProgress,
    analysisProgress,
    
    // Actions
    setError,
    clearError,
    uploadVideo,
    startAnalysis,
    cancelAnalysis,
    loadResults,
    generateReport,
    downloadReport,
    toggleSidebar,
    toggleDarkMode,
    setSelectedTab,
    resetUpload,
    resetAnalysis,
    resetResults,
    resetAll
  }
})

export default useAppStore