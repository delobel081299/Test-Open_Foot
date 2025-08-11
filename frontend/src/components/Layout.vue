<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Header -->
    <header class="bg-white shadow-sm border-b border-gray-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center h-16">
          <!-- Logo -->
          <div class="flex items-center">
            <router-link to="/" class="flex items-center space-x-3">
              <div class="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <span class="text-white font-bold text-sm">⚽</span>
              </div>
              <div>
                <h1 class="text-xl font-bold text-gray-900">Football AI</h1>
                <p class="text-xs text-gray-500">Analyzer</p>
              </div>
            </router-link>
          </div>

          <!-- Navigation -->
          <nav class="flex space-x-8">
            <router-link
              v-for="item in navigation"
              :key="item.name"
              :to="item.href"
              :class="[
                'flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors',
                isActive(item.href)
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              ]"
            >
              <component :is="item.icon" class="h-5 w-5 mr-2" />
              {{ item.name }}
            </router-link>
          </nav>

          <!-- Settings -->
          <div class="flex items-center space-x-4">
            <button class="p-2 text-gray-400 hover:text-gray-600 transition-colors">
              <Cog6ToothIcon class="h-6 w-6" />
            </button>
          </div>
        </div>
      </div>
    </header>

    <!-- Main content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <slot />
    </main>

    <!-- Footer -->
    <footer class="bg-white border-t border-gray-200 mt-auto">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div class="flex justify-between items-center">
          <div class="text-sm text-gray-500">
            © 2024 Football AI Analyzer. All rights reserved.
          </div>
          <div class="flex space-x-6 text-sm text-gray-500">
            <a href="#" class="hover:text-gray-900 transition-colors">Privacy Policy</a>
            <a href="#" class="hover:text-gray-900 transition-colors">Terms of Service</a>
            <a href="#" class="hover:text-gray-900 transition-colors">Support</a>
          </div>
        </div>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import {
  Home as HomeIcon,
  Upload as CloudArrowUpIcon,
  BarChart3 as ChartBarIcon,
  FileText as DocumentTextIcon,
  Settings as Cog6ToothIcon
} from 'lucide-vue-next'

const route = useRoute()

const navigation = [
  { name: 'Home', href: '/', icon: HomeIcon },
  { name: 'Upload', href: '/upload', icon: CloudArrowUpIcon },
  { name: 'Analysis', href: '/analysis', icon: ChartBarIcon },
  { name: 'Reports', href: '/reports', icon: DocumentTextIcon }
]

const isActive = (path) => {
  if (path === '/') {
    return route.path === '/'
  }
  return route.path.startsWith(path)
}
</script>

<style scoped>
</style>