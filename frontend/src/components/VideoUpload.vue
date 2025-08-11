<template>
  <div>
    <!-- File Selected View -->
    <div v-if="selectedFile && !isUploading" class="bg-white rounded-lg shadow-md p-6">
      <div class="flex items-start justify-between mb-4">
        <div class="flex items-center space-x-3">
          <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
            <PlayIcon class="h-6 w-6 text-blue-600" />
          </div>
          <div>
            <h3 class="font-medium text-gray-900">{{ selectedFile.name }}</h3>
            <p class="text-sm text-gray-500">
              {{ formatFileSize(selectedFile.size) }} • {{ selectedFile.type }}
            </p>
          </div>
        </div>
        <button
          @click="handleRemove"
          class="p-1 text-gray-400 hover:text-gray-600 transition-colors"
        >
          <XIcon class="h-5 w-5" />
        </button>
      </div>

      <div v-if="previewUrl" class="mb-4">
        <video
          :src="previewUrl"
          controls
          class="w-full max-h-64 rounded-lg bg-black"
        >
          Your browser does not support the video tag.
        </video>
      </div>

      <div class="flex space-x-3">
        <button
          @click="handleUpload"
          class="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Upload & Analyze
        </button>
        <button
          @click="handleRemove"
          class="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
        >
          Remove
        </button>
      </div>
    </div>

    <!-- Uploading View -->
    <div v-else-if="isUploading" class="bg-white rounded-lg shadow-md p-6">
      <div class="text-center">
        <div class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <CloudArrowUpIcon class="h-8 w-8 text-blue-600 animate-pulse" />
        </div>
        <h3 class="text-lg font-medium text-gray-900 mb-2">
          Uploading Video...
        </h3>
        <p class="text-sm text-gray-500 mb-4">
          {{ selectedFile?.name }}
        </p>
        
        <div class="w-full bg-gray-200 rounded-full h-2 mb-2">
          <div 
            class="bg-blue-600 h-2 rounded-full transition-all duration-300" 
            :style="{ width: `${progress}%` }"
          ></div>
        </div>
        <p class="text-sm text-gray-500">
          {{ Math.round(progress) }}% complete
        </p>
      </div>
    </div>

    <!-- Drop Zone -->
    <div
      v-else
      @drop="onDrop"
      @dragover.prevent
      @dragenter.prevent
      @dragleave="isDragActive = false"
      @dragenter="isDragActive = true"
      @click="openFileDialog"
      :class="[
        'border-2 border-dashed rounded-lg cursor-pointer transition-all',
        isDragActive 
          ? 'border-blue-500 bg-blue-50' 
          : 'border-gray-300 hover:border-gray-400 bg-white'
      ]"
    >
      <input
        ref="fileInput"
        type="file"
        accept="video/*,.mp4,.avi,.mov,.mkv,.webm"
        @change="onFileSelect"
        class="hidden"
      />
      
      <div class="p-12 text-center">
        <CloudArrowUpIcon class="mx-auto h-12 w-12 text-gray-400 mb-4" />
        
        <div v-if="isDragActive">
          <p class="text-lg font-medium text-blue-600 mb-2">
            Drop your video here
          </p>
          <p class="text-sm text-blue-500">
            Release to upload
          </p>
        </div>
        <div v-else>
          <p class="text-lg font-medium text-gray-900 mb-2">
            Upload football video
          </p>
          <p class="text-sm text-gray-500 mb-4">
            Drag and drop your video file here, or click to browse
          </p>
        </div>

        <div class="flex flex-wrap justify-center gap-2 text-xs text-gray-500 mt-4">
          <span class="px-2 py-1 bg-gray-100 rounded">MP4</span>
          <span class="px-2 py-1 bg-gray-100 rounded">AVI</span>
          <span class="px-2 py-1 bg-gray-100 rounded">MOV</span>
          <span class="px-2 py-1 bg-gray-100 rounded">MKV</span>
          <span class="px-2 py-1 bg-gray-100 rounded">WEBM</span>
        </div>
        
        <p class="text-xs text-gray-400 mt-2">
          Maximum file size: 2GB
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { Upload as CloudArrowUpIcon, Play as PlayIcon, X as XIcon } from 'lucide-vue-next'

const props = defineProps({
  isUploading: {
    type: Boolean,
    default: false
  },
  progress: {
    type: Number,
    default: 0
  }
})

const emit = defineEmits(['upload'])

const selectedFile = ref(null)
const previewUrl = ref(null)
const isDragActive = ref(false)
const fileInput = ref(null)

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const validateFile = (file) => {
  // Validate file size (max 2GB)
  const maxSize = 2 * 1024 * 1024 * 1024 // 2GB in bytes
  if (file.size > maxSize) {
    alert('File size must be less than 2GB')
    return false
  }

  // Validate file type
  const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/x-matroska']
  if (!allowedTypes.includes(file.type)) {
    alert('Please select a valid video file (MP4, AVI, MOV, MKV, WEBM)')
    return false
  }

  return true
}

const setFile = (file) => {
  if (!validateFile(file)) return

  selectedFile.value = file
  
  // Create preview URL
  const url = URL.createObjectURL(file)
  previewUrl.value = url
}

const onDrop = (event) => {
  event.preventDefault()
  isDragActive.value = false
  
  const files = event.dataTransfer.files
  if (files.length > 0) {
    setFile(files[0])
  }
}

const onFileSelect = (event) => {
  const files = event.target.files
  if (files.length > 0) {
    setFile(files[0])
  }
}

const openFileDialog = () => {
  fileInput.value?.click()
}

const handleUpload = () => {
  if (selectedFile.value) {
    emit('upload', selectedFile.value)
  }
}

const handleRemove = () => {
  selectedFile.value = null
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = null
  }
  // Reset file input
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}
</script>

<style scoped>
</style>