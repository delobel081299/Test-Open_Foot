<template>
  <div class="report-viewer">
    <!-- Viewer Header -->
    <div class="viewer-header">
      <div class="header-left">
        <h2 class="report-title">
          {{ reportTitle || 'Rapport d\'analyse football' }}
        </h2>
        <div class="report-meta">
          <span v-if="reportDate" class="report-date">
            <Calendar :size="14" />
            {{ formatDate(reportDate) }}
          </span>
          <span v-if="reportSize" class="report-size">
            <File :size="14" />
            {{ formatFileSize(reportSize) }}
          </span>
        </div>
      </div>

      <div class="header-actions">
        <!-- Navigation -->
        <div class="page-navigation">
          <button
            @click="goToPrevPage"
            :disabled="currentPage <= 1"
            class="nav-button"
            title="Page précédente"
          >
            <ChevronLeft :size="16" />
          </button>
          
          <div class="page-info">
            <input
              v-model.number="currentPageInput"
              @blur="goToPage(currentPageInput)"
              @keyup.enter="goToPage(currentPageInput)"
              type="number"
              :min="1"
              :max="totalPages"
              class="page-input"
            />
            <span class="page-separator">/</span>
            <span class="total-pages">{{ totalPages }}</span>
          </div>
          
          <button
            @click="goToNextPage"
            :disabled="currentPage >= totalPages"
            class="nav-button"
            title="Page suivante"
          >
            <ChevronRight :size="16" />
          </button>
        </div>

        <!-- Zoom Controls -->
        <div class="zoom-controls">
          <button
            @click="zoomOut"
            :disabled="zoomLevel <= MIN_ZOOM"
            class="zoom-button"
            title="Zoom arrière"
          >
            <ZoomOut :size="16" />
          </button>
          
          <div class="zoom-display">
            <span class="zoom-percentage">{{ Math.round(zoomLevel * 100) }}%</span>
          </div>
          
          <button
            @click="zoomIn"
            :disabled="zoomLevel >= MAX_ZOOM"
            class="zoom-button"
            title="Zoom avant"
          >
            <ZoomIn :size="16" />
          </button>
          
          <button
            @click="resetZoom"
            class="zoom-button"
            title="Ajuster à la largeur"
          >
            <Maximize2 :size="16" />
          </button>
        </div>

        <!-- View Options -->
        <div class="view-options">
          <button
            @click="toggleSidebar"
            :class="{ 'active': showSidebar }"
            class="view-button"
            title="Afficher/Masquer la navigation"
          >
            <Menu :size="16" />
          </button>
          
          <button
            @click="togglePrintMode"
            :class="{ 'active': isPrintMode }"
            class="view-button"
            title="Mode impression"
          >
            <Printer :size="16" />
          </button>
          
          <button
            @click="toggleFullscreen"
            class="view-button"
            title="Plein écran"
          >
            <Maximize v-if="!isFullscreen" :size="16" />
            <Minimize v-else :size="16" />
          </button>
        </div>

        <!-- Actions -->
        <div class="header-actions-group">
          <button
            @click="downloadReport"
            :disabled="isLoading"
            class="action-button download"
          >
            <Download :size="16" />
            Télécharger
          </button>
          
          <button
            @click="printReport"
            class="action-button print"
          >
            <Printer :size="16" />
            Imprimer
          </button>
          
          <button
            @click="shareReport"
            class="action-button share"
          >
            <Share2 :size="16" />
            Partager
          </button>
        </div>
      </div>
    </div>

    <!-- Main Content -->
    <div class="viewer-content" :class="{ 'with-sidebar': showSidebar, 'print-mode': isPrintMode }">
      <!-- Sidebar Navigation -->
      <div v-if="showSidebar" class="viewer-sidebar">
        <div class="sidebar-header">
          <h3 class="sidebar-title">Sections</h3>
          <button @click="showSidebar = false" class="sidebar-close">
            <X :size="16" />
          </button>
        </div>

        <div class="sidebar-content">
          <div class="outline-tree">
            <div
              v-for="(section, index) in reportOutline"
              :key="section.id"
              class="outline-item"
              :class="{ 'active': currentSection === section.id }"
              @click="goToSection(section)"
            >
              <div class="outline-header">
                <component :is="getSectionIcon(section.type)" :size="14" />
                <span class="outline-title">{{ section.title }}</span>
                <span class="outline-page">p.{{ section.page }}</span>
              </div>
              
              <!-- Sub-sections -->
              <div v-if="section.children?.length" class="outline-children">
                <div
                  v-for="child in section.children"
                  :key="child.id"
                  class="outline-child"
                  @click.stop="goToSection(child)"
                >
                  <span class="child-title">{{ child.title }}</span>
                  <span class="child-page">p.{{ child.page }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- PDF Viewer -->
      <div class="pdf-viewer" ref="pdfContainer">
        <!-- Loading State -->
        <div v-if="isLoading" class="loading-container">
          <div class="loading-spinner">
            <Loader2 :size="48" class="animate-spin text-blue-600" />
          </div>
          <p class="loading-text">{{ loadingText }}</p>
          <div v-if="loadingProgress > 0" class="loading-progress">
            <div class="progress-bar">
              <div
                class="progress-fill"
                :style="{ width: `${loadingProgress}%` }"
              ></div>
            </div>
            <span class="progress-percentage">{{ Math.round(loadingProgress) }}%</span>
          </div>
        </div>

        <!-- PDF Content -->
        <div
          v-else-if="pdfDoc"
          class="pdf-content"
          :style="{ transform: `scale(${zoomLevel})`, transformOrigin: 'top left' }"
        >
          <!-- Single Page Mode -->
          <div v-if="viewMode === 'single'" class="pdf-page-container">
            <canvas
              ref="pdfCanvas"
              class="pdf-page"
              @wheel="handleWheel"
            ></canvas>
          </div>

          <!-- Continuous Mode -->
          <div v-else class="pdf-pages-container">
            <div
              v-for="pageNum in totalPages"
              :key="pageNum"
              class="pdf-page-wrapper"
              :id="`page-${pageNum}`"
            >
              <canvas
                :ref="`page-canvas-${pageNum}`"
                class="pdf-page"
                @click="currentPage = pageNum"
              ></canvas>
            </div>
          </div>
        </div>

        <!-- Error State -->
        <div v-else-if="error" class="error-container">
          <div class="error-icon">
            <AlertTriangle :size="48" class="text-red-500" />
          </div>
          <h3 class="error-title">Erreur de chargement</h3>
          <p class="error-message">{{ error }}</p>
          <button @click="retryLoad" class="retry-button">
            <RotateCcw :size="16" />
            Réessayer
          </button>
        </div>

        <!-- Empty State -->
        <div v-else class="empty-container">
          <div class="empty-icon">
            <FileText :size="48" class="text-gray-400" />
          </div>
          <h3 class="empty-title">Aucun rapport à afficher</h3>
          <p class="empty-message">
            Générez d'abord un rapport pour le visualiser ici.
          </p>
        </div>
      </div>
    </div>

    <!-- Search Dialog -->
    <div v-if="showSearchDialog" class="search-dialog-overlay">
      <div class="search-dialog">
        <div class="search-header">
          <h3>Rechercher dans le rapport</h3>
          <button @click="showSearchDialog = false" class="search-close">
            <X :size="16" />
          </button>
        </div>
        
        <div class="search-content">
          <div class="search-input-group">
            <Search :size="16" class="search-icon" />
            <input
              v-model="searchQuery"
              @keyup.enter="performSearch"
              type="text"
              placeholder="Rechercher du texte..."
              class="search-input"
            />
            <button @click="performSearch" class="search-button">
              Rechercher
            </button>
          </div>
          
          <div v-if="searchResults.length" class="search-results">
            <div class="results-header">
              <span>{{ searchResults.length }} résultat(s) trouvé(s)</span>
            </div>
            <div class="results-list">
              <div
                v-for="(result, index) in searchResults"
                :key="index"
                class="result-item"
                @click="goToSearchResult(result)"
              >
                <div class="result-page">Page {{ result.page }}</div>
                <div class="result-text">{{ result.text }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Keyboard Shortcuts -->
    <div v-if="showShortcuts" class="shortcuts-overlay">
      <div class="shortcuts-panel">
        <div class="shortcuts-header">
          <h3>Raccourcis clavier</h3>
          <button @click="showShortcuts = false" class="shortcuts-close">
            <X :size="16" />
          </button>
        </div>
        <div class="shortcuts-content">
          <div class="shortcut-item">
            <kbd>Ctrl+F</kbd>
            <span>Rechercher</span>
          </div>
          <div class="shortcut-item">
            <kbd>Ctrl+P</kbd>
            <span>Imprimer</span>
          </div>
          <div class="shortcut-item">
            <kbd>Ctrl+D</kbd>
            <span>Télécharger</span>
          </div>
          <div class="shortcut-item">
            <kbd>←/→</kbd>
            <span>Page précédente/suivante</span>
          </div>
          <div class="shortcut-item">
            <kbd>+/-</kbd>
            <span>Zoom avant/arrière</span>
          </div>
          <div class="shortcut-item">
            <kbd>0</kbd>
            <span>Ajuster à la largeur</span>
          </div>
          <div class="shortcut-item">
            <kbd>F11</kbd>
            <span>Plein écran</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import * as pdfjsLib from 'pdfjs-dist'
import type { PDFDocumentProxy, PDFPageProxy } from 'pdfjs-dist'
import {
  Calendar,
  File,
  ChevronLeft,
  ChevronRight,
  ZoomOut,
  ZoomIn,
  Maximize2,
  Menu,
  Printer,
  Maximize,
  Minimize,
  Download,
  Share2,
  X,
  Loader2,
  AlertTriangle,
  RotateCcw,
  FileText,
  Search,
  BarChart3,
  Users,
  Target,
  MapPin,
  Activity
} from 'lucide-vue-next'

// Props
interface Props {
  pdfUrl?: string
  reportTitle?: string
  reportDate?: string
  reportSize?: number
  jobId?: string
}

const props = defineProps<Props>()

// Emits
const emit = defineEmits<{
  download: [jobId: string]
  share: [jobId: string]
  error: [error: string]
}>()

// Refs
const pdfContainer = ref<HTMLElement>()
const pdfCanvas = ref<HTMLCanvasElement>()

// State
const pdfDoc = ref<PDFDocumentProxy | null>(null)
const currentPage = ref(1)
const currentPageInput = ref(1)
const totalPages = ref(0)
const zoomLevel = ref(1)
const isLoading = ref(false)
const loadingProgress = ref(0)
const loadingText = ref('Chargement du rapport...')
const error = ref('')
const isFullscreen = ref(false)
const showSidebar = ref(true)
const isPrintMode = ref(false)
const showSearchDialog = ref(false)
const showShortcuts = ref(false)
const searchQuery = ref('')
const searchResults = ref<any[]>([])
const currentSection = ref('')
const viewMode = ref<'single' | 'continuous'>('single')

// Constants
const MIN_ZOOM = 0.5
const MAX_ZOOM = 3
const ZOOM_STEP = 0.25

// Mock report outline
const reportOutline = ref([
  {
    id: 'summary',
    title: 'Résumé exécutif',
    type: 'summary',
    page: 1,
    children: []
  },
  {
    id: 'players',
    title: 'Analyse des joueurs',
    type: 'players',
    page: 3,
    children: [
      { id: 'individual', title: 'Performances individuelles', page: 3 },
      { id: 'comparisons', title: 'Comparaisons', page: 8 }
    ]
  },
  {
    id: 'team',
    title: 'Analyse d\'équipe',
    type: 'team',
    page: 12,
    children: [
      { id: 'tactics', title: 'Tactiques', page: 12 },
      { id: 'formation', title: 'Formation', page: 15 }
    ]
  },
  {
    id: 'technical',
    title: 'Analyse technique',
    type: 'technical',
    page: 18,
    children: []
  },
  {
    id: 'recommendations',
    title: 'Recommandations',
    type: 'recommendations',
    page: 22,
    children: []
  }
])

// Computed
watch(currentPage, (newPage) => {
  currentPageInput.value = newPage
  if (pdfDoc.value && newPage >= 1 && newPage <= totalPages.value) {
    renderPage(newPage)
  }
})

// Methods
async function loadPDF(url: string) {
  if (!url) return

  isLoading.value = true
  error.value = ''
  loadingProgress.value = 0

  try {
    // Configure PDF.js worker
    pdfjsLib.GlobalWorkerOptions.workerSrc = '/node_modules/pdfjs-dist/build/pdf.worker.js'

    const loadingTask = pdfjsLib.getDocument({
      url,
      withCredentials: false
    })

    loadingTask.onProgress = (progress) => {
      if (progress.total > 0) {
        loadingProgress.value = (progress.loaded / progress.total) * 100
      }
    }

    pdfDoc.value = await loadingTask.promise
    totalPages.value = pdfDoc.value.numPages
    
    loadingText.value = 'Rendu de la première page...'
    await renderPage(1)
    
  } catch (err: any) {
    error.value = `Erreur lors du chargement du PDF: ${err.message}`
    emit('error', error.value)
  } finally {
    isLoading.value = false
    loadingProgress.value = 0
  }
}

async function renderPage(pageNumber: number) {
  if (!pdfDoc.value || !pdfCanvas.value) return

  try {
    const page = await pdfDoc.value.getPage(pageNumber)
    const canvas = pdfCanvas.value
    const context = canvas.getContext('2d')
    
    if (!context) return

    const viewport = page.getViewport({ scale: 1.5 })
    
    canvas.height = viewport.height
    canvas.width = viewport.width

    const renderContext = {
      canvasContext: context,
      viewport: viewport
    }

    await page.render(renderContext).promise
    currentPage.value = pageNumber
    
  } catch (err: any) {
    error.value = `Erreur lors du rendu de la page: ${err.message}`
  }
}

function goToPrevPage() {
  if (currentPage.value > 1) {
    currentPage.value--
  }
}

function goToNextPage() {
  if (currentPage.value < totalPages.value) {
    currentPage.value++
  }
}

function goToPage(pageNumber: number) {
  if (pageNumber >= 1 && pageNumber <= totalPages.value) {
    currentPage.value = pageNumber
  }
}

function zoomIn() {
  if (zoomLevel.value < MAX_ZOOM) {
    zoomLevel.value = Math.min(MAX_ZOOM, zoomLevel.value + ZOOM_STEP)
  }
}

function zoomOut() {
  if (zoomLevel.value > MIN_ZOOM) {
    zoomLevel.value = Math.max(MIN_ZOOM, zoomLevel.value - ZOOM_STEP)
  }
}

function resetZoom() {
  zoomLevel.value = 1
}

function handleWheel(event: WheelEvent) {
  if (event.ctrlKey) {
    event.preventDefault()
    if (event.deltaY < 0) {
      zoomIn()
    } else {
      zoomOut()
    }
  }
}

function toggleSidebar() {
  showSidebar.value = !showSidebar.value
}

function togglePrintMode() {
  isPrintMode.value = !isPrintMode.value
}

async function toggleFullscreen() {
  if (!document.fullscreenElement) {
    await pdfContainer.value?.requestFullscreen()
    isFullscreen.value = true
  } else {
    await document.exitFullscreen()
    isFullscreen.value = false
  }
}

function downloadReport() {
  if (props.jobId) {
    emit('download', props.jobId)
  } else if (props.pdfUrl) {
    // Direct download
    const link = document.createElement('a')
    link.href = props.pdfUrl
    link.download = props.reportTitle || 'rapport_football.pdf'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }
}

function printReport() {
  window.print()
}

function shareReport() {
  if (props.jobId) {
    emit('share', props.jobId)
  }
}

function goToSection(section: any) {
  currentSection.value = section.id
  goToPage(section.page)
}

function getSectionIcon(type: string) {
  const iconMap: Record<string, any> = {
    summary: BarChart3,
    players: Users,
    team: Target,
    technical: Activity,
    recommendations: MapPin
  }
  return iconMap[type] || FileText
}

function performSearch() {
  // Mock search implementation
  searchResults.value = [
    {
      page: 1,
      text: `Résultat trouvé contenant "${searchQuery.value}"...`
    }
  ]
}

function goToSearchResult(result: any) {
  goToPage(result.page)
  showSearchDialog.value = false
}

function retryLoad() {
  if (props.pdfUrl) {
    loadPDF(props.pdfUrl)
  }
}

function handleKeydown(event: KeyboardEvent) {
  if (event.ctrlKey || event.metaKey) {
    switch (event.key) {
      case 'f':
        event.preventDefault()
        showSearchDialog.value = true
        break
      case 'p':
        event.preventDefault()
        printReport()
        break
      case 'd':
        event.preventDefault()
        downloadReport()
        break
    }
  } else {
    switch (event.key) {
      case 'ArrowLeft':
        goToPrevPage()
        break
      case 'ArrowRight':
        goToNextPage()
        break
      case '+':
      case '=':
        zoomIn()
        break
      case '-':
        zoomOut()
        break
      case '0':
        resetZoom()
        break
      case 'F11':
        event.preventDefault()
        toggleFullscreen()
        break
      case '?':
        showShortcuts.value = !showShortcuts.value
        break
    }
  }
}

// Utility Functions
function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleDateString('fr-FR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}

function formatFileSize(bytes: number): string {
  const sizes = ['B', 'KB', 'MB', 'GB']
  if (bytes === 0) return '0 B'
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${Math.round(bytes / Math.pow(1024, i) * 100) / 100} ${sizes[i]}`
}

// Lifecycle
onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
  document.addEventListener('fullscreenchange', () => {
    isFullscreen.value = !!document.fullscreenElement
  })

  if (props.pdfUrl) {
    loadPDF(props.pdfUrl)
  }
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
})

// Watch for prop changes
watch(() => props.pdfUrl, (newUrl) => {
  if (newUrl) {
    loadPDF(newUrl)
  }
})
</script>

<style scoped>
.report-viewer {
  @apply h-full flex flex-col bg-gray-50 dark:bg-gray-900;
}

.viewer-header {
  @apply flex items-center justify-between p-4 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 shadow-sm;
}

.header-left {
  @apply space-y-1;
}

.report-title {
  @apply text-lg font-semibold text-gray-900 dark:text-white;
}

.report-meta {
  @apply flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-300;
}

.report-date,
.report-size {
  @apply flex items-center space-x-1;
}

.header-actions {
  @apply flex items-center space-x-4;
}

.page-navigation {
  @apply flex items-center space-x-2;
}

.nav-button {
  @apply p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 disabled:opacity-50 disabled:cursor-not-allowed;
}

.page-info {
  @apply flex items-center space-x-1 text-sm;
}

.page-input {
  @apply w-12 px-2 py-1 text-center border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-white;
}

.page-separator {
  @apply text-gray-500 dark:text-gray-400;
}

.zoom-controls {
  @apply flex items-center space-x-1;
}

.zoom-button {
  @apply p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed;
}

.zoom-display {
  @apply px-2 text-sm font-mono text-gray-600 dark:text-gray-300;
}

.view-options {
  @apply flex items-center space-x-1;
}

.view-button {
  @apply p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded;
}

.view-button.active {
  @apply text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30;
}

.header-actions-group {
  @apply flex items-center space-x-2;
}

.action-button {
  @apply flex items-center space-x-2 px-3 py-2 text-sm font-medium rounded-lg transition-colors;
}

.action-button.download {
  @apply bg-blue-600 hover:bg-blue-700 text-white disabled:bg-gray-400 disabled:cursor-not-allowed;
}

.action-button.print {
  @apply border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700;
}

.action-button.share {
  @apply border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700;
}

.viewer-content {
  @apply flex-1 flex overflow-hidden;
}

.viewer-content.with-sidebar {
  @apply pr-0;
}

.viewer-content.print-mode {
  @apply bg-white;
}

.viewer-sidebar {
  @apply w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col;
}

.sidebar-header {
  @apply flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700;
}

.sidebar-title {
  @apply font-semibold text-gray-900 dark:text-white;
}

.sidebar-close {
  @apply text-gray-400 hover:text-gray-600 dark:hover:text-gray-200;
}

.sidebar-content {
  @apply flex-1 overflow-y-auto p-4;
}

.outline-tree {
  @apply space-y-2;
}

.outline-item {
  @apply cursor-pointer;
}

.outline-header {
  @apply flex items-center space-x-2 px-2 py-1 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 rounded;
}

.outline-item.active .outline-header {
  @apply bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400;
}

.outline-title {
  @apply flex-1 truncate;
}

.outline-page {
  @apply text-xs text-gray-500 dark:text-gray-400;
}

.outline-children {
  @apply ml-6 mt-1 space-y-1;
}

.outline-child {
  @apply flex items-center justify-between px-2 py-1 text-xs hover:bg-gray-100 dark:hover:bg-gray-700 rounded cursor-pointer;
}

.child-title {
  @apply flex-1 truncate text-gray-700 dark:text-gray-300;
}

.child-page {
  @apply text-gray-500 dark:text-gray-400;
}

.pdf-viewer {
  @apply flex-1 flex flex-col items-center justify-center bg-gray-100 dark:bg-gray-800 overflow-auto;
}

.loading-container,
.error-container,
.empty-container {
  @apply text-center space-y-4;
}

.loading-spinner {
  @apply flex justify-center;
}

.loading-text {
  @apply text-gray-600 dark:text-gray-400;
}

.loading-progress {
  @apply max-w-xs mx-auto space-y-2;
}

.progress-bar {
  @apply w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden;
}

.progress-fill {
  @apply h-full bg-blue-600 transition-all duration-300;
}

.progress-percentage {
  @apply text-sm text-gray-600 dark:text-gray-400;
}

.error-icon,
.empty-icon {
  @apply flex justify-center;
}

.error-title,
.empty-title {
  @apply text-lg font-semibold text-gray-900 dark:text-white;
}

.error-message,
.empty-message {
  @apply text-gray-600 dark:text-gray-400 max-w-md;
}

.retry-button {
  @apply flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors;
}

.pdf-content {
  @apply p-4;
}

.pdf-page-container,
.pdf-pages-container {
  @apply space-y-4;
}

.pdf-page-wrapper {
  @apply bg-white shadow-lg;
}

.pdf-page {
  @apply block border border-gray-300 dark:border-gray-600;
}

.search-dialog-overlay {
  @apply fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50;
}

.search-dialog {
  @apply bg-white dark:bg-gray-800 rounded-xl p-6 max-w-md w-full mx-4;
}

.search-header {
  @apply flex items-center justify-between mb-4;
}

.search-header h3 {
  @apply font-semibold text-gray-900 dark:text-white;
}

.search-close {
  @apply text-gray-400 hover:text-gray-600 dark:hover:text-gray-200;
}

.search-content {
  @apply space-y-4;
}

.search-input-group {
  @apply flex items-center space-x-2;
}

.search-icon {
  @apply text-gray-400;
}

.search-input {
  @apply flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white;
}

.search-button {
  @apply px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors;
}

.search-results {
  @apply space-y-2;
}

.results-header {
  @apply text-sm text-gray-600 dark:text-gray-400;
}

.results-list {
  @apply space-y-1 max-h-40 overflow-y-auto;
}

.result-item {
  @apply p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded cursor-pointer;
}

.result-page {
  @apply text-xs text-blue-600 dark:text-blue-400 font-medium;
}

.result-text {
  @apply text-sm text-gray-700 dark:text-gray-300 truncate;
}

.shortcuts-overlay {
  @apply fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50;
}

.shortcuts-panel {
  @apply bg-white dark:bg-gray-800 rounded-xl p-6 max-w-md w-full mx-4;
}

.shortcuts-header {
  @apply flex items-center justify-between mb-4;
}

.shortcuts-header h3 {
  @apply font-semibold text-gray-900 dark:text-white;
}

.shortcuts-close {
  @apply text-gray-400 hover:text-gray-600 dark:hover:text-gray-200;
}

.shortcuts-content {
  @apply space-y-3;
}

.shortcut-item {
  @apply flex items-center justify-between text-sm;
}

.shortcut-item kbd {
  @apply px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded text-xs font-mono;
}

.shortcut-item span {
  @apply text-gray-600 dark:text-gray-400;
}

/* Print styles */
@media print {
  .viewer-header,
  .viewer-sidebar {
    @apply hidden;
  }
  
  .viewer-content {
    @apply bg-white;
  }
  
  .pdf-viewer {
    @apply bg-white;
  }
}
</style>