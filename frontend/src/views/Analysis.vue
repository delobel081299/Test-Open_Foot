<template>
  <div class="max-w-6xl mx-auto">
    <div class="text-center mb-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-4">
        Video Analysis
      </h1>
      <p class="text-lg text-gray-600">
        AI-powered analysis in progress
      </p>
    </div>

    <div class="bg-white rounded-lg shadow-md p-8 text-center">
      <div class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
        <ChartBarIcon class="h-8 w-8 text-blue-600 animate-pulse" />
      </div>
      <h3 class="text-lg font-medium text-gray-900 mb-2">
        Analyzing Video {{ $route.params.videoId }}
      </h3>
      <p class="text-sm text-gray-500 mb-4">
        Please wait while our AI analyzes your football video...
      </p>
      
      <div class="w-full bg-gray-200 rounded-full h-2 mb-4">
        <div class="bg-blue-600 h-2 rounded-full animate-pulse" style="width: 60%"></div>
      </div>
      
      <p class="text-sm text-gray-500">
        This may take a few minutes depending on video length
      </p>
    </div>
  </div>
</template>

<script setup>
import { BarChart3 as ChartBarIcon } from 'lucide-vue-next'
</script>

<style scoped>
</style>