<template>
  <div class="upload-zone-container">
    <!-- Upload Zone -->
    <div
      ref="uploadZone"
      class="upload-zone"
      :class="{
        'upload-zone--dragging': isDragging,
        'upload-zone--uploading': isUploading,
        'upload-zone--completed': uploadCompleted,
        'upload-zone--error': hasError
      }"
      @drop.prevent="handleDrop"
      @dragover.prevent="handleDragOver"
      @dragenter.prevent="handleDragEnter"
      @dragleave.prevent="handleDragLeave"
      @click="triggerFileInput"
    >
      <!-- File Input (Hidden) -->
      <input
        ref="fileInput"
        type="file"
        accept="video/*,.mp4,.avi,.mov,.mkv,.wmv,.flv"
        class="hidden"
        @change="handleFileSelect"
      />

      <!-- Upload Content -->
      <div v-if="!selectedFile" class="upload-content">
        <div class="upload-icon">
          <CloudUpload :size="64" class="text-blue-500" />
        </div>
        <h3 class="upload-title">
          Glissez votre vidéo ici ou cliquez pour parcourir
        </h3>
        <p class="upload-subtitle">
          Formats supportés: MP4, AVI, MOV, MKV, WMV, FLV
        </p>
        <p class="upload-size-limit">
          Taille maximale: 500 MB
        </p>
        
        <div class="upload-features">
          <div class="feature-item">
            <Eye :size="20" />
            <span>Preview instantané</span>
          </div>
          <div class="feature-item">
            <Zap :size="20" />
            <span>Upload chunked</span>
          </div>
          <div class="feature-item">
            <Shield :size="20" />
            <span>Validation sécurisée</span>
          </div>
        </div>
      </div>

      <!-- File Preview -->
      <div v-else class="file-preview">
        <div class="preview-header">
          <div class="file-info">
            <Film :size="32" class="text-blue-500" />
            <div>
              <h4 class="file-name">{{ selectedFile.name }}</h4>
              <p class="file-details">
                {{ formatFileSize(selectedFile.size) }} • {{ fileType }}
              </p>
            </div>
          </div>
          <button
            v-if="!isUploading && !uploadCompleted"
            @click.stop="removeFile"
            class="remove-button"
            title="Supprimer le fichier"
          >
            <X :size="20" />
          </button>
        </div>

        <!-- Video Preview -->
        <div v-if="videoUrl" class="video-preview">
          <video
            ref="videoPreview"
            :src="videoUrl"
            class="preview-video"
            controls
            preload="metadata"
            @loadedmetadata="handleVideoLoaded"
          />
          
          <div class="video-info">
            <div class="info-item">
              <Clock :size="16" />
              <span>{{ formatDuration(videoDuration) }}</span>
            </div>
            <div class="info-item">
              <Monitor :size="16" />
              <span>{{ videoResolution }}</span>
            </div>
          </div>
        </div>

        <!-- Upload Progress -->
        <div v-if="isUploading" class="upload-progress">
          <div class="progress-info">
            <div class="progress-text">
              <span class="progress-status">{{ uploadStatus }}</span>
              <span class="progress-percentage">{{ uploadProgress }}%</span>
            </div>
            <div class="progress-details">
              <span v-if="uploadETA" class="eta">
                Temps restant: {{ formatTime(uploadETA) }}
              </span>
              <span class="speed">{{ uploadSpeed }}</span>
            </div>
          </div>
          
          <div class="progress-bar">
            <div 
              class="progress-fill"
              :style="{ width: `${uploadProgress}%` }"
            ></div>
          </div>

          <!-- Cancel Button -->
          <button
            @click.stop="cancelUpload"
            class="cancel-button"
            title="Annuler l'upload"
          >
            <Square :size="16" />
            Annuler
          </button>
        </div>

        <!-- Upload Success -->
        <div v-if="uploadCompleted" class="upload-success">
          <div class="success-icon">
            <CheckCircle :size="32" class="text-green-500" />
          </div>
          <p class="success-message">Vidéo uploadée avec succès !</p>
          <div class="success-actions">
            <button @click="startAnalysis" class="btn-primary">
              <Play :size="16" />
              Démarrer l'analyse
            </button>
            <button @click="uploadAnother" class="btn-secondary">
              <Upload :size="16" />
              Uploader une autre vidéo
            </button>
          </div>
        </div>

        <!-- Upload Error -->
        <div v-if="hasError" class="upload-error">
          <div class="error-icon">
            <AlertCircle :size="32" class="text-red-500" />
          </div>
          <p class="error-message">{{ errorMessage }}</p>
          <button @click="retryUpload" class="btn-retry">
            <RotateCcw :size="16" />
            Réessayer
          </button>
        </div>
      </div>

      <!-- Drag Overlay -->
      <div v-if="isDragging" class="drag-overlay">
        <div class="drag-content">
          <Download :size="64" class="text-blue-500" />
          <h3>Relâchez pour uploader</h3>
        </div>
      </div>
    </div>

    <!-- Upload Configuration -->
    <div v-if="selectedFile && !isUploading && !uploadCompleted" class="upload-config">
      <h4 class="config-title">Configuration d'analyse</h4>
      
      <div class="config-options">
        <div class="option-group">
          <label class="option-label">
            <input
              v-model="analysisConfig.analyze_poses"
              type="checkbox"
              class="option-checkbox"
            />
            <span>Analyse biomécanique</span>
          </label>
          <p class="option-description">
            Analyse des mouvements et postures des joueurs
          </p>
        </div>

        <div class="option-group">
          <label class="option-label">
            <input
              v-model="analysisConfig.analyze_actions"
              type="checkbox"
              class="option-checkbox"
            />
            <span>Classification des actions</span>
          </label>
          <p class="option-description">
            Reconnaissance automatique des actions football
          </p>
        </div>

        <div class="option-group">
          <label class="option-label">
            <input
              v-model="analysisConfig.analyze_tactics"
              type="checkbox"
              class="option-checkbox"
            />
            <span>Analyse tactique</span>
          </label>
          <p class="option-description">
            Formation, positionnement et stratégies d'équipe
          </p>
        </div>

        <div class="option-group">
          <label class="option-label">
            <input
              v-model="analysisConfig.generate_heatmaps"
              type="checkbox"
              class="option-checkbox"
            />
            <span>Cartes de chaleur</span>
          </label>
          <p class="option-description">
            Visualisation des zones d'activité des joueurs
          </p>
        </div>
      </div>

      <div class="config-advanced">
        <button
          @click="showAdvanced = !showAdvanced"
          class="advanced-toggle"
        >
          <Settings :size="16" />
          Paramètres avancés
          <ChevronDown 
            :size="16" 
            :class="{ 'rotate-180': showAdvanced }"
            class="transition-transform"
          />
        </button>

        <div v-if="showAdvanced" class="advanced-options">
          <div class="option-row">
            <label class="option-label-inline">
              Confiance détection:
              <input
                v-model.number="analysisConfig.detection_confidence"
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                class="range-input"
              />
              <span class="range-value">{{ analysisConfig.detection_confidence }}</span>
            </label>
          </div>

          <div class="option-row">
            <label class="option-label-inline">
              FPS d'analyse:
              <select
                v-model.number="analysisConfig.fps"
                class="select-input"
              >
                <option :value="15">15 FPS (Rapide)</option>
                <option :value="25">25 FPS (Standard)</option>
                <option :value="30">30 FPS (Précis)</option>
              </select>
            </label>
          </div>
        </div>
      </div>

      <div class="config-actions">
        <button 
          @click="startUpload"
          class="btn-upload"
          :disabled="!isValidFile"
        >
          <Upload :size="16" />
          Commencer l'upload
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import { useAppStore } from '@/stores'
import type { AnalysisConfig } from '@/types'
import {
  CloudUpload,
  Eye,
  Zap,
  Shield,
  Film,
  X,
  Clock,
  Monitor,
  CheckCircle,
  Play,
  Upload,
  AlertCircle,
  RotateCcw,
  Download,
  Square,
  Settings,
  ChevronDown
} from 'lucide-vue-next'

// Props & Emits
interface Props {
  maxFileSize?: number
  acceptedFormats?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  maxFileSize: 500 * 1024 * 1024, // 500MB
  acceptedFormats: () => ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
})

const emit = defineEmits<{
  uploadStarted: [jobId: string]
  uploadCompleted: [data: { job_id: string; video_id: number }]
  uploadError: [error: string]
  analysisStarted: [jobId: string]
}>()

// Store
const store = useAppStore()

// Refs
const uploadZone = ref<HTMLElement>()
const fileInput = ref<HTMLInputElement>()
const videoPreview = ref<HTMLVideoElement>()

// State
const isDragging = ref(false)
const selectedFile = ref<File | null>(null)
const videoUrl = ref<string | null>(null)
const videoDuration = ref(0)
const videoResolution = ref('')
const showAdvanced = ref(false)
const dragCounter = ref(0)

// Upload state
const uploadStartTime = ref(0)
const uploadedBytes = ref(0)

// Analysis configuration
const analysisConfig = reactive<Partial<AnalysisConfig>>({
  analyze_poses: true,
  analyze_actions: true,
  analyze_tactics: true,
  generate_heatmaps: true,
  detection_confidence: 0.5,
  fps: 25,
  detailed_scoring: true
})

// Computed
const isUploading = computed(() => store.isUploading)
const uploadProgress = computed(() => store.uploadProgress)
const uploadCompleted = computed(() => 
  store.currentUpload?.status === 'completed'
)
const hasError = computed(() => 
  store.currentUpload?.status === 'failed' || !!store.error
)
const errorMessage = computed(() => 
  store.currentUpload?.error || store.error || 'Erreur inconnue'
)

const uploadStatus = computed(() => {
  if (!store.currentUpload) return 'Initialisation...'
  
  switch (store.currentUpload.status) {
    case 'uploading':
      return 'Upload en cours...'
    case 'processing':
      return 'Traitement du fichier...'
    case 'validating':
      return 'Validation...'
    default:
      return 'Upload...'
  }
})

const uploadETA = computed(() => 
  store.currentUpload?.eta_seconds
)

const uploadSpeed = computed(() => {
  if (!store.currentUpload || !uploadStartTime.value) return ''
  
  const elapsed = (Date.now() - uploadStartTime.value) / 1000
  if (elapsed > 0) {
    const speed = uploadedBytes.value / elapsed
    return formatSpeed(speed)
  }
  return ''
})

const fileType = computed(() => {
  if (!selectedFile.value) return ''
  return selectedFile.value.name.split('.').pop()?.toUpperCase() || ''
})

const isValidFile = computed(() => {
  if (!selectedFile.value) return false
  
  // Check file size
  if (selectedFile.value.size > props.maxFileSize) return false
  
  // Check file format
  const extension = '.' + selectedFile.value.name.split('.').pop()?.toLowerCase()
  return props.acceptedFormats.includes(extension)
})

// Methods
function triggerFileInput() {
  if (!isUploading.value && fileInput.value) {
    fileInput.value.click()
  }
}

function handleFileSelect(event: Event) {
  const target = event.target as HTMLInputElement
  if (target.files && target.files.length > 0) {
    handleFile(target.files[0])
  }
}

function handleDragOver(event: DragEvent) {
  event.preventDefault()
}

function handleDragEnter(event: DragEvent) {
  event.preventDefault()
  dragCounter.value++
  isDragging.value = true
}

function handleDragLeave(event: DragEvent) {
  event.preventDefault()
  dragCounter.value--
  if (dragCounter.value <= 0) {
    isDragging.value = false
    dragCounter.value = 0
  }
}

function handleDrop(event: DragEvent) {
  event.preventDefault()
  isDragging.value = false
  dragCounter.value = 0
  
  if (event.dataTransfer && event.dataTransfer.files.length > 0) {
    handleFile(event.dataTransfer.files[0])
  }
}

function handleFile(file: File) {
  // Validate file
  if (file.size > props.maxFileSize) {
    store.setError(`Le fichier est trop volumineux. Taille maximale: ${formatFileSize(props.maxFileSize)}`)
    return
  }
  
  const extension = '.' + file.name.split('.').pop()?.toLowerCase()
  if (!props.acceptedFormats.includes(extension)) {
    store.setError(`Format non supporté. Formats acceptés: ${props.acceptedFormats.join(', ')}`)
    return
  }
  
  selectedFile.value = file
  createVideoPreview(file)
  store.clearError()
}

function createVideoPreview(file: File) {
  if (videoUrl.value) {
    URL.revokeObjectURL(videoUrl.value)
  }
  videoUrl.value = URL.createObjectURL(file)
}

function handleVideoLoaded() {
  if (videoPreview.value) {
    videoDuration.value = videoPreview.value.duration
    videoResolution.value = `${videoPreview.value.videoWidth}x${videoPreview.value.videoHeight}`
  }
}

function removeFile() {
  selectedFile.value = null
  if (videoUrl.value) {
    URL.revokeObjectURL(videoUrl.value)
    videoUrl.value = null
  }
  videoDuration.value = 0
  videoResolution.value = ''
  store.resetUpload()
}

async function startUpload() {
  if (!selectedFile.value || !isValidFile.value) return
  
  try {
    uploadStartTime.value = Date.now()
    uploadedBytes.value = 0
    
    const result = await store.uploadVideo(selectedFile.value, {
      analysis_config: analysisConfig
    })
    
    emit('uploadStarted', result.job_id)
    emit('uploadCompleted', result)
  } catch (error: any) {
    emit('uploadError', error.message)
  }
}

async function startAnalysis() {
  if (!store.currentUpload?.video_id) return
  
  try {
    const result = await store.startAnalysis(
      store.currentUpload.job_id || '', 
      analysisConfig
    )
    emit('analysisStarted', result.job_id)
  } catch (error: any) {
    store.setError(error.message)
  }
}

function uploadAnother() {
  removeFile()
  store.resetAll()
}

function retryUpload() {
  store.clearError()
  if (selectedFile.value) {
    startUpload()
  }
}

function cancelUpload() {
  // TODO: Implement upload cancellation
  store.resetUpload()
}

// Utility functions
function formatFileSize(bytes: number): string {
  const sizes = ['B', 'KB', 'MB', 'GB']
  if (bytes === 0) return '0 B'
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${Math.round(bytes / Math.pow(1024, i) * 100) / 100} ${sizes[i]}`
}

function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  
  if (h > 0) {
    return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }
  return `${m}:${s.toString().padStart(2, '0')}`
}

function formatTime(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${minutes}m ${secs}s`
}

function formatSpeed(bytesPerSecond: number): string {
  const mbps = bytesPerSecond / (1024 * 1024)
  return `${mbps.toFixed(1)} MB/s`
}

// Lifecycle
onMounted(() => {
  // Add global drag handlers to prevent default browser behavior
  document.addEventListener('dragover', preventDefault)
  document.addEventListener('drop', preventDefault)
})

onUnmounted(() => {
  // Clean up
  if (videoUrl.value) {
    URL.revokeObjectURL(videoUrl.value)
  }
  
  document.removeEventListener('dragover', preventDefault)
  document.removeEventListener('drop', preventDefault)
})

function preventDefault(e: Event) {
  e.preventDefault()
}
</script>

<style scoped>
.upload-zone-container {
  @apply w-full max-w-4xl mx-auto space-y-6;
}

.upload-zone {
  @apply relative border-2 border-dashed border-gray-300 rounded-xl p-8 transition-all duration-300 cursor-pointer;
  @apply hover:border-blue-400 hover:bg-blue-50/50 dark:hover:bg-blue-900/20;
  min-height: 300px;
}

.upload-zone--dragging {
  @apply border-blue-500 bg-blue-100 dark:bg-blue-900/30 scale-[1.02];
}

.upload-zone--uploading {
  @apply cursor-not-allowed;
}

.upload-zone--completed {
  @apply border-green-400 bg-green-50 dark:bg-green-900/20;
}

.upload-zone--error {
  @apply border-red-400 bg-red-50 dark:bg-red-900/20;
}

.upload-content {
  @apply text-center space-y-4;
}

.upload-icon {
  @apply flex justify-center mb-4;
}

.upload-title {
  @apply text-xl font-semibold text-gray-700 dark:text-gray-200;
}

.upload-subtitle {
  @apply text-gray-500 dark:text-gray-400;
}

.upload-size-limit {
  @apply text-sm text-gray-400 dark:text-gray-500;
}

.upload-features {
  @apply flex justify-center space-x-8 mt-6;
}

.feature-item {
  @apply flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-300;
}

.file-preview {
  @apply space-y-6;
}

.preview-header {
  @apply flex items-center justify-between;
}

.file-info {
  @apply flex items-center space-x-3;
}

.file-name {
  @apply font-medium text-gray-900 dark:text-white truncate max-w-xs;
}

.file-details {
  @apply text-sm text-gray-500 dark:text-gray-400;
}

.remove-button {
  @apply p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors;
}

.video-preview {
  @apply space-y-4;
}

.preview-video {
  @apply w-full max-h-64 rounded-lg bg-black;
}

.video-info {
  @apply flex space-x-6 text-sm text-gray-600 dark:text-gray-300;
}

.info-item {
  @apply flex items-center space-x-2;
}

.upload-progress {
  @apply space-y-4;
}

.progress-info {
  @apply flex justify-between items-start;
}

.progress-text {
  @apply flex items-center space-x-3;
}

.progress-status {
  @apply font-medium text-gray-700 dark:text-gray-200;
}

.progress-percentage {
  @apply text-2xl font-bold text-blue-600 dark:text-blue-400;
}

.progress-details {
  @apply text-sm text-gray-500 dark:text-gray-400 space-x-3;
}

.progress-bar {
  @apply w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden;
}

.progress-fill {
  @apply h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-300 ease-out;
}

.cancel-button {
  @apply flex items-center space-x-2 px-4 py-2 text-sm font-medium text-red-600 hover:text-red-700;
  @apply border border-red-200 hover:border-red-300 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors;
}

.upload-success,
.upload-error {
  @apply text-center space-y-4;
}

.success-icon,
.error-icon {
  @apply flex justify-center;
}

.success-message,
.error-message {
  @apply font-medium text-gray-700 dark:text-gray-200;
}

.success-actions {
  @apply flex justify-center space-x-4;
}

.btn-primary {
  @apply flex items-center space-x-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors;
}

.btn-secondary {
  @apply flex items-center space-x-2 px-6 py-3 border border-gray-300 hover:border-gray-400 text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-colors;
}

.btn-retry {
  @apply flex items-center space-x-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors;
}

.drag-overlay {
  @apply absolute inset-0 bg-blue-100/90 dark:bg-blue-900/90 flex items-center justify-center rounded-xl;
}

.drag-content {
  @apply text-center space-y-4;
}

.drag-content h3 {
  @apply text-xl font-semibold text-blue-700 dark:text-blue-200;
}

.upload-config {
  @apply bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700;
}

.config-title {
  @apply text-lg font-semibold text-gray-900 dark:text-white mb-4;
}

.config-options {
  @apply space-y-4;
}

.option-group {
  @apply space-y-1;
}

.option-label {
  @apply flex items-center space-x-3 font-medium text-gray-700 dark:text-gray-200 cursor-pointer;
}

.option-checkbox {
  @apply w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 focus:ring-2 dark:bg-gray-700 dark:border-gray-600;
}

.option-description {
  @apply text-sm text-gray-500 dark:text-gray-400 ml-7;
}

.config-advanced {
  @apply mt-6 pt-6 border-t border-gray-200 dark:border-gray-700;
}

.advanced-toggle {
  @apply flex items-center space-x-2 text-sm font-medium text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100 transition-colors;
}

.advanced-options {
  @apply mt-4 space-y-4 pl-4 border-l-2 border-gray-200 dark:border-gray-700;
}

.option-row {
  @apply space-y-2;
}

.option-label-inline {
  @apply flex items-center space-x-3 text-sm font-medium text-gray-700 dark:text-gray-200;
}

.range-input {
  @apply flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700;
}

.range-value {
  @apply text-sm font-mono text-gray-600 dark:text-gray-300 min-w-[3rem] text-right;
}

.select-input {
  @apply px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white;
}

.config-actions {
  @apply mt-6 pt-6 border-t border-gray-200 dark:border-gray-700;
}

.btn-upload {
  @apply flex items-center space-x-2 px-8 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium rounded-lg transition-colors;
}

.btn-upload:disabled {
  @apply cursor-not-allowed;
}
</style>