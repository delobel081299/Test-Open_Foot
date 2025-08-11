<template>
  <div class="max-w-6xl mx-auto">
    <div class="text-center mb-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-4">
        Analysis Results
      </h1>
      <p class="text-lg text-gray-600">
        Detailed analysis results for your video
      </p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Overall Score -->
      <div class="bg-white rounded-lg shadow-md p-6 text-center">
        <h3 class="text-lg font-medium text-gray-900 mb-4">Overall Score</h3>
        <div class="text-4xl font-bold text-blue-600 mb-2">8.5</div>
        <div class="text-sm text-gray-600">out of 10</div>
      </div>

      <!-- Technical Score -->
      <div class="bg-white rounded-lg shadow-md p-6 text-center">
        <h3 class="text-lg font-medium text-gray-900 mb-4">Technical</h3>
        <div class="text-4xl font-bold text-green-600 mb-2">9.2</div>
        <div class="text-sm text-gray-600">out of 10</div>
      </div>

      <!-- Tactical Score -->
      <div class="bg-white rounded-lg shadow-md p-6 text-center">
        <h3 class="text-lg font-medium text-gray-900 mb-4">Tactical</h3>
        <div class="text-4xl font-bold text-purple-600 mb-2">7.8</div>
        <div class="text-sm text-gray-600">out of 10</div>
      </div>
    </div>

    <div class="mt-8 bg-white rounded-lg shadow-md p-6">
      <h3 class="text-lg font-medium text-gray-900 mb-4">Player Analysis</h3>
      <p class="text-gray-600">
        Video ID: {{ $route.params.videoId || 'N/A' }}
      </p>
      <p class="text-gray-600 mt-2">
        Detailed player analysis and metrics will appear here once the backend is connected.
      </p>
    </div>
  </div>
</template>

<script setup>
</script>

<style scoped>
</style>