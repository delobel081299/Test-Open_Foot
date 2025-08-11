<template>
  <div class="home-page">
    <!-- Hero Section -->
    <section class="hero-section field-background">
      <div class="hero-content">
        <div class="hero-text">
          <h1 class="hero-title animate-fade-in">
            Football AI Analyzer
          </h1>
          <p class="hero-subtitle animate-fade-in delay-200">
            Analysez vos vidéos de football avec l'intelligence artificielle la plus avancée
          </p>
          <div class="hero-features animate-fade-in delay-300">
            <div class="feature-item">
              <Eye :size="20" />
              <span>Détection automatique des joueurs</span>
            </div>
            <div class="feature-item">
              <Activity :size="20" />
              <span>Analyse biomécanique avancée</span>
            </div>
            <div class="feature-item">
              <Target :size="20" />
              <span>Évaluation tactique en temps réel</span>
            </div>
          </div>
        </div>
        
        <div class="hero-action animate-slide-in-up delay-500">
          <button
            @click="scrollToUpload"
            class="hero-cta-button"
          >
            <Upload :size="20" />
            Commencer l'analyse
            <ArrowRight :size="16" />
          </button>
          
          <div class="hero-stats">
            <div class="stat-item">
              <span class="stat-number">10,000+</span>
              <span class="stat-label">Vidéos analysées</span>
            </div>
            <div class="stat-item">
              <span class="stat-number">98%</span>
              <span class="stat-label">Précision</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Upload Section -->
    <section id="upload-section" class="upload-section">
      <div class="container">
        <div class="section-header">
          <h2 class="section-title">Uploadez votre vidéo</h2>
          <p class="section-description">
            Glissez votre fichier vidéo pour commencer l'analyse intelligente
          </p>
        </div>
        
        <UploadZone
          @upload-started="handleUploadStarted"
          @upload-completed="handleUploadCompleted"
          @analysis-started="handleAnalysisStarted"
          class="animate-scale-in"
        />
      </div>
    </section>

    <!-- Analysis Section (shown when analysis is running) -->
    <section
      v-if="currentJobId && (isAnalyzing || hasResults)"
      class="analysis-section"
    >
      <div class="container">
        <div class="section-header">
          <h2 class="section-title">Analyse en cours</h2>
          <p class="section-description">
            Votre vidéo est en cours d'analyse par notre intelligence artificielle
          </p>
        </div>
        
        <AnalysisDashboard
          :job-id="currentJobId"
          @player-selected="handlePlayerSelected"
          @moment-selected="handleMomentSelected"
          class="animate-fade-in"
        />
      </div>
    </section>

    <!-- Video Player Section (shown when video is available) -->
    <section
      v-if="currentVideoSrc && showVideoPlayer"
      class="video-section"
    >
      <div class="container">
        <div class="section-header">
          <h2 class="section-title">Lecteur vidéo</h2>
          <p class="section-description">
            Visualisez votre vidéo avec les annotations d'analyse
          </p>
        </div>
        
        <div class="video-player-wrapper">
          <VideoPlayer
            :video-src="currentVideoSrc"
            :video-title="currentVideoTitle"
            :annotations="videoAnnotations"
            :timeline-events="timelineEvents"
            @annotation-selected="handleAnnotationSelected"
            @time-update="handleVideoTimeUpdate"
            show-stats
            class="animate-slide-in-up"
          />
        </div>
      </div>
    </section>

    <!-- Players Grid Section -->
    <section
      v-if="playersData.length > 0"
      class="players-section"
    >
      <div class="container">
        <div class="section-header">
          <h2 class="section-title">Analyses des joueurs</h2>
          <p class="section-description">
            Performance individuelle et métriques détaillées
          </p>
        </div>
        
        <div class="players-grid">
          <PlayerCard
            v-for="(player, index) in playersData"
            :key="player.player_id"
            :player="player"
            :is-selected="selectedPlayerId === player.player_id"
            :show-evolution="true"
            :evolution-data="getPlayerEvolution(player.player_id)"
            @click="handlePlayerCardClick"
            @view-details="handlePlayerDetails"
            @view-heatmap="handlePlayerHeatmap"
            @compare="handlePlayerCompare"
            :class="`animate-fade-in stagger-${Math.min(index + 1, 5)}`"
          />
        </div>
      </div>
    </section>

    <!-- Report Section (shown when report is available) -->
    <section
      v-if="showReportViewer && currentPdfUrl"
      class="report-section"
    >
      <div class="container">
        <div class="section-header">
          <h2 class="section-title">Rapport d'analyse</h2>
          <p class="section-description">
            Rapport PDF complet avec toutes les analyses et recommandations
          </p>
        </div>
        
        <div class="report-viewer-wrapper">
          <ReportViewer
            :pdf-url="currentPdfUrl"
            :report-title="currentReportTitle"
            :report-date="currentReportDate"
            :job-id="currentJobId"
            @download="handleReportDownload"
            @share="handleReportShare"
            class="animate-slide-in-left"
          />
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="features-section">
      <div class="container">
        <div class="section-header">
          <h2 class="section-title">Fonctionnalités avancées</h2>
          <p class="section-description">
            Découvrez toutes les capacités de notre plateforme d'analyse
          </p>
        </div>
        
        <div class="features-grid">
          <div class="feature-card animate-fade-in stagger-1">
            <div class="feature-icon ai-icon">
              <Cpu :size="32" />
            </div>
            <h3 class="feature-title">IA Avancée</h3>
            <p class="feature-description">
              Algorithmes de deep learning pour une analyse précise des mouvements
            </p>
          </div>
          
          <div class="feature-card animate-fade-in stagger-2">
            <div class="feature-icon realtime-icon">
              <Zap :size="32" />
            </div>
            <h3 class="feature-title">Temps Réel</h3>
            <p class="feature-description">
              Analyse en temps réel avec suivi de progression et notifications
            </p>
          </div>
          
          <div class="feature-card animate-fade-in stagger-3">
            <div class="feature-icon analytics-icon">
              <BarChart3 :size="32" />
            </div>
            <h3 class="feature-title">Analytics</h3>
            <p class="feature-description">
              Métriques détaillées et visualisations interactives
            </p>
          </div>
          
          <div class="feature-card animate-fade-in stagger-4">
            <div class="feature-icon reports-icon">
              <FileText :size="32" />
            </div>
            <h3 class="feature-title">Rapports PDF</h3>
            <p class="feature-description">
              Génération automatique de rapports professionnels
            </p>
          </div>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="cta-section">
      <div class="container">
        <div class="cta-content animate-scale-in">
          <h2 class="cta-title">Prêt à analyser votre équipe ?</h2>
          <p class="cta-description">
            Commencez dès maintenant avec votre première analyse gratuite
          </p>
          <button
            @click="scrollToUpload"
            class="cta-button"
          >
            <Play :size="20" />
            Commencer maintenant
          </button>
        </div>
      </div>
    </section>

    <!-- Floating Action Button -->
    <button
      v-if="showFab"
      @click="scrollToTop"
      class="fab animate-bounce-gentle"
      title="Retour en haut"
    >
      <ArrowUp :size="20" />
    </button>

    <!-- Toast Notifications -->
    <div class="toast-container">
      <div
        v-for="toast in toasts"
        :key="toast.id"
        class="toast"
        :class="`toast-${toast.type}`"
      >
        <div class="toast-icon">
          <CheckCircle v-if="toast.type === 'success'" :size="20" />
          <AlertCircle v-else-if="toast.type === 'error'" :size="20" />
          <Info v-else-if="toast.type === 'info'" :size="20" />
          <AlertTriangle v-else :size="20" />
        </div>
        <div class="toast-content">
          <p class="toast-message">{{ toast.message }}</p>
        </div>
        <button @click="removeToast(toast.id)" class="toast-close">
          <X :size="16" />
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useAppStore } from '@/stores'
import UploadZone from '@/components/UploadZone.vue'
import AnalysisDashboard from '@/components/AnalysisDashboard.vue'
import VideoPlayer from '@/components/VideoPlayer.vue'
import PlayerCard from '@/components/PlayerCard.vue'
import ReportViewer from '@/components/ReportViewer.vue'
import type { PlayerResult, VideoAnnotation } from '@/types'
import {
  Upload,
  ArrowRight,
  Eye,
  Activity,
  Target,
  Cpu,
  Zap,
  BarChart3,
  FileText,
  Play,
  ArrowUp,
  CheckCircle,
  AlertCircle,
  Info,
  AlertTriangle,
  X
} from 'lucide-vue-next'

// Store
const store = useAppStore()

// State
const currentJobId = ref<string | null>(null)
const currentVideoSrc = ref<string | null>(null)
const currentVideoTitle = ref<string | null>(null)
const currentPdfUrl = ref<string | null>(null)
const currentReportTitle = ref<string | null>(null)
const currentReportDate = ref<string | null>(null)
const selectedPlayerId = ref<number | null>(null)
const showVideoPlayer = ref(false)
const showReportViewer = ref(false)
const showFab = ref(false)
const toasts = ref<Array<{
  id: number
  type: 'success' | 'error' | 'info' | 'warning'
  message: string
}>>([])

// Mock data
const playersData = ref<PlayerResult[]>([])
const videoAnnotations = ref<VideoAnnotation[]>([])
const timelineEvents = ref<any[]>([])

// Computed
const isAnalyzing = computed(() => store.isAnalyzing)
const hasResults = computed(() => store.hasResults)

// Methods
function handleUploadStarted(jobId: string) {
  currentJobId.value = jobId
  showToast('success', 'Upload démarré avec succès')
}

function handleUploadCompleted(data: { job_id: string; video_id: number }) {
  showToast('success', 'Vidéo uploadée avec succès!')
  // Mock video URL - in real app, this would come from the API
  currentVideoSrc.value = '/mock-video.mp4'
  currentVideoTitle.value = `Analyse vidéo #${data.video_id}`
  showVideoPlayer.value = true
}

function handleAnalysisStarted(jobId: string) {
  currentJobId.value = jobId
  showToast('info', 'Analyse démarrée - veuillez patienter')
  
  // Mock analysis completion after delay
  setTimeout(() => {
    mockAnalysisComplete()
  }, 5000)
}

function handlePlayerSelected(player: PlayerResult) {
  selectedPlayerId.value = player.player_id
  showToast('info', `Joueur ${player.jersey_number} sélectionné`)
}

function handleMomentSelected(timestamp: number) {
  showToast('info', `Navigation vers ${formatTime(timestamp)}`)
}

function handlePlayerCardClick(player: PlayerResult) {
  selectedPlayerId.value = player.player_id
}

function handlePlayerDetails(player: PlayerResult) {
  showToast('info', `Détails du joueur ${player.jersey_number}`)
}

function handlePlayerHeatmap(player: PlayerResult) {
  showToast('info', `Heatmap du joueur ${player.jersey_number}`)
}

function handlePlayerCompare(player: PlayerResult) {
  showToast('info', `Comparaison du joueur ${player.jersey_number}`)
}

function handleAnnotationSelected(annotation: VideoAnnotation) {
  showToast('info', `Annotation sélectionnée: ${annotation.title}`)
}

function handleVideoTimeUpdate(currentTime: number) {
  // Handle video time updates
}

function handleReportDownload(jobId: string) {
  showToast('success', 'Téléchargement du rapport démarré')
}

function handleReportShare(jobId: string) {
  showToast('info', 'Lien de partage copié dans le presse-papier')
}

function scrollToUpload() {
  document.getElementById('upload-section')?.scrollIntoView({ behavior: 'smooth' })
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

function showToast(type: 'success' | 'error' | 'info' | 'warning', message: string) {
  const id = Date.now()
  toasts.value.push({ id, type, message })
  
  // Auto remove after 5 seconds
  setTimeout(() => {
    removeToast(id)
  }, 5000)
}

function removeToast(id: number) {
  const index = toasts.value.findIndex(toast => toast.id === id)
  if (index > -1) {
    toasts.value.splice(index, 1)
  }
}

function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${minutes}:${secs.toString().padStart(2, '0')}`
}

function getPlayerEvolution(playerId: number) {
  // Mock evolution data
  return [
    { timestamp: 1, score: 75 },
    { timestamp: 2, score: 78 },
    { timestamp: 3, score: 82 },
    { timestamp: 4, score: 85 },
    { timestamp: 5, score: 88 }
  ]
}

function mockAnalysisComplete() {
  // Mock completed analysis data
  playersData.value = [
    {
      player_id: 1,
      jersey_number: 10,
      team: 'home',
      position: 'Attaquant',
      scores: {
        overall: 85,
        biomechanics: 88,
        technical: 82,
        tactical: 87
      },
      metrics: {
        distance_covered: 9500,
        top_speed: 28.5,
        average_speed: 12.3,
        passes_completed: 45,
        pass_accuracy: 89,
        shots: 6,
        tackles: 2,
        interceptions: 1
      },
      performance: {
        strengths: ['Vitesse de course', 'Précision des passes'],
        weaknesses: ['Jeu aérien', 'Défense'],
        recommendations: ['Travailler les duels aériens', 'Améliorer le repli défensif']
      },
      feedback: 'Excellente performance offensive avec une bonne vision du jeu.'
    },
    {
      player_id: 2,
      jersey_number: 7,
      team: 'away',
      position: 'Milieu',
      scores: {
        overall: 79,
        biomechanics: 82,
        technical: 75,
        tactical: 81
      },
      metrics: {
        distance_covered: 11200,
        top_speed: 26.8,
        average_speed: 11.9,
        passes_completed: 67,
        pass_accuracy: 92,
        shots: 2,
        tackles: 8,
        interceptions: 5
      },
      performance: {
        strengths: ['Récupération de balle', 'Distribution'],
        weaknesses: ['Finition', 'Vitesse'],
        recommendations: ['Améliorer la frappe', 'Renforcer l\'explosivité']
      },
      feedback: 'Très bon travail défensif et distribution précise du jeu.'
    }
  ]

  // Mock video annotations
  videoAnnotations.value = [
    {
      id: '1',
      timestamp: 120,
      type: 'goal',
      title: 'But!',
      description: 'Superbe frappe du joueur #10',
      players: [10],
      position: { x: 75, y: 45 }
    },
    {
      id: '2',
      timestamp: 180,
      type: 'foul',
      title: 'Faute',
      description: 'Faute du joueur #7',
      players: [7],
      position: { x: 35, y: 60 }
    }
  ]

  // Mock timeline events
  timelineEvents.value = [
    { id: '1', timestamp: 120, type: 'goal', title: 'But - Joueur #10' },
    { id: '2', timestamp: 180, type: 'foul', title: 'Faute - Joueur #7' },
    { id: '3', timestamp: 240, type: 'substitution', title: 'Changement' }
  ]

  showToast('success', 'Analyse terminée avec succès!')
  
  // Mock PDF report
  setTimeout(() => {
    currentPdfUrl.value = '/mock-report.pdf'
    currentReportTitle.value = `Rapport d'analyse - ${currentVideoTitle.value}`
    currentReportDate.value = new Date().toISOString()
    showReportViewer.value = true
    showToast('info', 'Rapport PDF généré')
  }, 2000)
}

function handleScroll() {
  showFab.value = window.scrollY > 500
}

// Lifecycle
onMounted(() => {
  window.addEventListener('scroll', handleScroll)
})

onUnmounted(() => {
  window.removeEventListener('scroll', handleScroll)
})
</script>

<style scoped>
.home-page {
  @apply min-h-screen;
}

.hero-section {
  @apply relative py-20 lg:py-32 overflow-hidden;
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);
}

.hero-content {
  @apply container mx-auto px-4 flex flex-col lg:flex-row items-center gap-12;
}

.hero-text {
  @apply flex-1 text-center lg:text-left;
}

.hero-title {
  @apply text-4xl lg:text-6xl font-bold text-gray-900 dark:text-white mb-6;
  @apply bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent;
}

.hero-subtitle {
  @apply text-lg lg:text-xl text-gray-600 dark:text-gray-300 mb-8 leading-relaxed;
}

.hero-features {
  @apply flex flex-col sm:flex-row gap-4 justify-center lg:justify-start;
}

.feature-item {
  @apply flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300;
}

.hero-action {
  @apply flex-1 text-center lg:text-right;
}

.hero-cta-button {
  @apply inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-blue-600 to-green-600;
  @apply text-white font-semibold rounded-xl shadow-lg hover:shadow-xl;
  @apply transform hover:scale-105 transition-all duration-300 mb-8;
}

.hero-stats {
  @apply flex gap-8 justify-center lg:justify-end;
}

.stat-item {
  @apply text-center;
}

.stat-number {
  @apply block text-2xl font-bold text-gray-900 dark:text-white;
}

.stat-label {
  @apply text-sm text-gray-600 dark:text-gray-400;
}

.upload-section,
.analysis-section,
.video-section,
.players-section,
.report-section,
.features-section {
  @apply py-16 lg:py-24;
}

.container {
  @apply max-w-7xl mx-auto px-4;
}

.section-header {
  @apply text-center mb-12;
}

.section-title {
  @apply text-3xl lg:text-4xl font-bold text-gray-900 dark:text-white mb-4;
}

.section-description {
  @apply text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto;
}

.video-player-wrapper,
.report-viewer-wrapper {
  @apply bg-white dark:bg-gray-800 rounded-2xl shadow-lg p-6;
}

.players-grid {
  @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6;
}

.features-section {
  @apply bg-gray-50 dark:bg-gray-800;
}

.features-grid {
  @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8;
}

.feature-card {
  @apply bg-white dark:bg-gray-700 rounded-xl p-6 text-center shadow-soft hover:shadow-medium;
  @apply transform hover:scale-105 transition-all duration-300;
}

.feature-icon {
  @apply w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center;
}

.ai-icon {
  @apply bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400;
}

.realtime-icon {
  @apply bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400;
}

.analytics-icon {
  @apply bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400;
}

.reports-icon {
  @apply bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400;
}

.feature-title {
  @apply text-xl font-semibold text-gray-900 dark:text-white mb-2;
}

.feature-description {
  @apply text-gray-600 dark:text-gray-300;
}

.cta-section {
  @apply py-16 bg-gradient-to-r from-blue-600 to-green-600;
}

.cta-content {
  @apply container mx-auto px-4 text-center text-white;
}

.cta-title {
  @apply text-3xl lg:text-4xl font-bold mb-4;
}

.cta-description {
  @apply text-lg mb-8 opacity-90;
}

.cta-button {
  @apply inline-flex items-center gap-3 px-8 py-4 bg-white text-blue-600;
  @apply font-semibold rounded-xl shadow-lg hover:shadow-xl;
  @apply transform hover:scale-105 transition-all duration-300;
}

.fab {
  @apply fixed bottom-6 right-6 w-12 h-12 bg-blue-600 hover:bg-blue-700 text-white;
  @apply rounded-full shadow-lg hover:shadow-xl flex items-center justify-center;
  @apply transform hover:scale-110 transition-all duration-300 z-50;
}

.toast-container {
  @apply fixed top-4 right-4 space-y-2 z-50;
}

.toast {
  @apply flex items-center gap-3 bg-white dark:bg-gray-800 rounded-lg shadow-lg border p-4;
  @apply transform transition-all duration-300 max-w-sm;
  animation: slideInRight 0.3s ease-out;
}

.toast-success {
  @apply border-green-200 dark:border-green-700;
}

.toast-error {
  @apply border-red-200 dark:border-red-700;
}

.toast-info {
  @apply border-blue-200 dark:border-blue-700;
}

.toast-warning {
  @apply border-yellow-200 dark:border-yellow-700;
}

.toast-icon {
  @apply flex-shrink-0;
}

.toast-success .toast-icon {
  @apply text-green-500;
}

.toast-error .toast-icon {
  @apply text-red-500;
}

.toast-info .toast-icon {
  @apply text-blue-500;
}

.toast-warning .toast-icon {
  @apply text-yellow-500;
}

.toast-content {
  @apply flex-1;
}

.toast-message {
  @apply text-sm font-medium text-gray-900 dark:text-white;
}

.toast-close {
  @apply text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 flex-shrink-0;
}

/* Animations */
@keyframes slideInRight {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}
</style>