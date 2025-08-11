<template>
  <div class="max-w-6xl mx-auto">
    <div class="text-center mb-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-4">
        Analysis Reports
      </h1>
      <p class="text-lg text-gray-600">
        Generate and download comprehensive analysis reports
      </p>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <!-- PDF Report -->
      <div class="bg-white rounded-lg shadow-md p-6">
        <div class="flex items-center mb-4">
          <DocumentTextIcon class="h-8 w-8 text-red-600 mr-3" />
          <h3 class="text-lg font-medium text-gray-900">PDF Report</h3>
        </div>
        <p class="text-sm text-gray-600 mb-4">
          Comprehensive analysis report with player statistics, tactical insights, and recommendations.
        </p>
        <button class="w-full px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors">
          Generate PDF Report
        </button>
      </div>

      <!-- Video Report -->
      <div class="bg-white rounded-lg shadow-md p-6">
        <div class="flex items-center mb-4">
          <PlayIcon class="h-8 w-8 text-blue-600 mr-3" />
          <h3 class="text-lg font-medium text-gray-900">Annotated Video</h3>
        </div>
        <p class="text-sm text-gray-600 mb-4">
          Video with AI annotations, player tracking, and performance highlights.
        </p>
        <button class="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
          Generate Video Report
        </button>
      </div>
    </div>

    <div class="mt-8 bg-white rounded-lg shadow-md p-6">
      <h3 class="text-lg font-medium text-gray-900 mb-4">Report History</h3>
      <p class="text-gray-600">
        Video ID: {{ $route.params.videoId || 'N/A' }}
      </p>
      <p class="text-gray-600 mt-2">
        Previous reports will appear here once the backend is connected.
      </p>
    </div>
  </div>
</template>

<script setup>
import { FileText as DocumentTextIcon, Play as PlayIcon } from 'lucide-vue-next'
</script>

<style scoped>
</style>