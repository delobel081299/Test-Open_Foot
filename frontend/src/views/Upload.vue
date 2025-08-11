<template>
  <div class="max-w-4xl mx-auto">
    <div class="text-center mb-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-4">
        Upload Football Video
      </h1>
      <p class="text-lg text-gray-600">
        Upload your football match or training video to start AI-powered analysis
      </p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
      <!-- Upload Section -->
      <div class="lg:col-span-2">
        <VideoUpload
          @upload="handleUpload"
          :is-uploading="isUploading"
          :progress="uploadProgress"
        />
      </div>

      <!-- Info Section -->
      <div class="space-y-6">
        <div class="bg-white rounded-lg shadow-md p-6">
          <h3 class="text-lg font-medium text-gray-900 mb-4">
            Supported Formats
          </h3>
          <ul class="space-y-2 text-sm text-gray-600">
            <li class="flex items-center">
              <span class="w-2 h-2 bg-green-400 rounded-full mr-3"></span>
              MP4 (recommended)
            </li>
            <li class="flex items-center">
              <span class="w-2 h-2 bg-green-400 rounded-full mr-3"></span>
              AVI
            </li>
            <li class="flex items-center">
              <span class="w-2 h-2 bg-green-400 rounded-full mr-3"></span>
              MOV
            </li>
            <li class="flex items-center">
              <span class="w-2 h-2 bg-green-400 rounded-full mr-3"></span>
              MKV
            </li>
            <li class="flex items-center">
              <span class="w-2 h-2 bg-green-400 rounded-full mr-3"></span>
              WEBM
            </li>
          </ul>
        </div>

        <div class="bg-white rounded-lg shadow-md p-6">
          <h3 class="text-lg font-medium text-gray-900 mb-4">
            Requirements
          </h3>
          <ul class="space-y-2 text-sm text-gray-600">
            <li>• Minimum resolution: 640x480</li>
            <li>• Minimum frame rate: 15 FPS</li>
            <li>• Duration: 5 seconds - 1 hour</li>
            <li>• Maximum file size: 2GB</li>
            <li>• Clear view of players</li>
          </ul>
        </div>

        <div class="bg-white rounded-lg shadow-md p-6">
          <h3 class="text-lg font-medium text-gray-900 mb-4">
            Analysis Features
          </h3>
          <ul class="space-y-2 text-sm text-gray-600">
            <li>• Player detection & tracking</li>
            <li>• Biomechanical analysis</li>
            <li>• Technical skill assessment</li>
            <li>• Tactical movement analysis</li>
            <li>• Performance scoring</li>
            <li>• Detailed PDF reports</li>
          </ul>
        </div>

        <div class="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h4 class="font-medium text-blue-900 mb-2">
            💡 Pro Tip
          </h4>
          <p class="text-sm text-blue-700">
            For best results, use videos with a side view of the field 
            and good lighting conditions. Multiple camera angles can 
            provide more comprehensive analysis.
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import VideoUpload from '@/components/VideoUpload.vue'
import apiService from '@/services/api'

const router = useRouter()
const isUploading = ref(false)
const uploadProgress = ref(0)

const handleUpload = async (file) => {
  isUploading.value = true
  uploadProgress.value = 0

  try {
    // Simulate upload progress
    const progressInterval = setInterval(() => {
      uploadProgress.value = Math.min(uploadProgress.value + Math.random() * 10, 90)
      if (uploadProgress.value >= 90) {
        clearInterval(progressInterval)
      }
    }, 500)

    const response = await apiService.uploadVideo(file, {}, (progress) => {
      uploadProgress.value = progress
    })

    clearInterval(progressInterval)
    uploadProgress.value = 100

    // Show success message (you can add a toast notification library here)
    alert('Video uploaded successfully!')
    
    // Navigate to analysis page
    setTimeout(() => {
      router.push(`/analysis/${response.id}`)
    }, 1000)

  } catch (error) {
    console.error('Upload failed:', error)
    alert('Upload failed. Please try again.')
    isUploading.value = false
    uploadProgress.value = 0
  }
}
</script>

<style scoped>
</style>