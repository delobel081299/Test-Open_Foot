<template>
  <div 
    class="player-card"
    :class="[
      `player-card--${player.team}`,
      { 'player-card--selected': isSelected }
    ]"
    @click="$emit('click', player)"
  >
    <!-- Card Header -->
    <div class="card-header">
      <div class="player-avatar">
        <img
          v-if="playerImage"
          :src="playerImage"
          :alt="`Joueur ${player.jersey_number}`"
          class="avatar-image"
        />
        <div v-else class="avatar-placeholder">
          <User :size="24" />
        </div>
        
        <!-- Jersey Number Badge -->
        <div class="jersey-badge">
          {{ player.jersey_number }}
        </div>
      </div>

      <div class="player-info">
        <h3 class="player-name">
          {{ playerName || `Joueur ${player.jersey_number}` }}
        </h3>
        <p class="player-position">{{ player.position }}</p>
        <div class="team-indicator">
          <div class="team-color" :class="`team-color--${player.team}`"></div>
          <span class="team-label">{{ teamLabel }}</span>
        </div>
      </div>

      <!-- Overall Score -->
      <div class="overall-score">
        <div class="score-circle" :class="getScoreClass(player.scores.overall)">
          <span class="score-value">{{ player.scores.overall }}</span>
        </div>
        <span class="score-label">Score global</span>
      </div>
    </div>

    <!-- Performance Radar Chart -->
    <div class="radar-container">
      <div class="radar-chart">
        <Radar
          :data="radarData"
          :options="radarOptions"
          :height="200"
        />
      </div>
    </div>

    <!-- Key Stats -->
    <div class="key-stats">
      <div class="stat-item">
        <div class="stat-icon biomechanics">
          <Activity :size="16" />
        </div>
        <div class="stat-content">
          <span class="stat-label">Biomécanique</span>
          <div class="stat-bar">
            <div 
              class="stat-fill biomechanics-fill"
              :style="{ width: `${player.scores.biomechanics}%` }"
            ></div>
          </div>
          <span class="stat-value">{{ player.scores.biomechanics }}/100</span>
        </div>
      </div>

      <div class="stat-item">
        <div class="stat-icon technical">
          <Target :size="16" />
        </div>
        <div class="stat-content">
          <span class="stat-label">Technique</span>
          <div class="stat-bar">
            <div 
              class="stat-fill technical-fill"
              :style="{ width: `${player.scores.technical}%` }"
            ></div>
          </div>
          <span class="stat-value">{{ player.scores.technical }}/100</span>
        </div>
      </div>

      <div class="stat-item">
        <div class="stat-icon tactical">
          <MapPin :size="16" />
        </div>
        <div class="stat-content">
          <span class="stat-label">Tactique</span>
          <div class="stat-bar">
            <div 
              class="stat-fill tactical-fill"
              :style="{ width: `${player.scores.tactical}%` }"
            ></div>
          </div>
          <span class="stat-value">{{ player.scores.tactical }}/100</span>
        </div>
      </div>
    </div>

    <!-- Performance Metrics (if available) -->
    <div v-if="player.metrics" class="performance-metrics">
      <div class="metrics-header">
        <BarChart3 :size="16" />
        <span>Métriques de performance</span>
      </div>
      
      <div class="metrics-grid">
        <div class="metric-item">
          <div class="metric-icon">
            <MapPin :size="12" />
          </div>
          <div class="metric-content">
            <span class="metric-value">{{ formatDistance(player.metrics.distance_covered) }}</span>
            <span class="metric-label">Distance</span>
          </div>
        </div>

        <div class="metric-item">
          <div class="metric-icon">
            <Zap :size="12" />
          </div>
          <div class="metric-content">
            <span class="metric-value">{{ player.metrics.top_speed.toFixed(1) }}</span>
            <span class="metric-label">Vitesse max</span>
          </div>
        </div>

        <div class="metric-item">
          <div class="metric-icon">
            <Target :size="12" />
          </div>
          <div class="metric-content">
            <span class="metric-value">{{ player.metrics.pass_accuracy }}%</span>
            <span class="metric-label">Précision</span>
          </div>
        </div>

        <div class="metric-item">
          <div class="metric-icon">
            <Activity :size="12" />
          </div>
          <div class="metric-content">
            <span class="metric-value">{{ player.metrics.shots }}</span>
            <span class="metric-label">Tirs</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Evolution Chart (if historical data available) -->
    <div v-if="showEvolution && evolutionData.length" class="evolution-section">
      <div class="evolution-header">
        <TrendingUp :size="16" />
        <span>Évolution</span>
        <button
          @click.stop="toggleEvolution"
          class="evolution-toggle"
        >
          <ChevronDown 
            :size="16"
            :class="{ 'rotate-180': showEvolutionChart }"
            class="transition-transform"
          />
        </button>
      </div>
      
      <div v-if="showEvolutionChart" class="evolution-chart">
        <Line
          :data="evolutionChartData"
          :options="evolutionChartOptions"
          :height="120"
        />
      </div>
    </div>

    <!-- Performance Insights -->
    <div v-if="player.performance" class="performance-insights">
      <!-- Strengths -->
      <div v-if="player.performance.strengths.length" class="insights-section">
        <div class="insights-header strengths-header">
          <TrendingUp :size="14" />
          <span>Points forts</span>
        </div>
        <div class="insights-list">
          <div
            v-for="strength in player.performance.strengths.slice(0, 2)"
            :key="strength"
            class="insight-item strength-item"
          >
            {{ strength }}
          </div>
        </div>
      </div>

      <!-- Weaknesses -->
      <div v-if="player.performance.weaknesses.length" class="insights-section">
        <div class="insights-header weaknesses-header">
          <TrendingDown :size="14" />
          <span>À améliorer</span>
        </div>
        <div class="insights-list">
          <div
            v-for="weakness in player.performance.weaknesses.slice(0, 2)"
            :key="weakness"
            class="insight-item weakness-item"
          >
            {{ weakness }}
          </div>
        </div>
      </div>
    </div>

    <!-- Action Buttons -->
    <div class="card-actions">
      <button
        @click.stop="$emit('viewDetails', player)"
        class="action-button primary"
      >
        <Eye :size="14" />
        Détails
      </button>
      
      <button
        @click.stop="$emit('viewHeatmap', player)"
        class="action-button secondary"
      >
        <MapPin :size="14" />
        Heatmap
      </button>
      
      <button
        @click.stop="$emit('compare', player)"
        class="action-button secondary"
      >
        <BarChart3 :size="14" />
        Comparer
      </button>
    </div>

    <!-- Hover Overlay -->
    <div class="hover-overlay">
      <div class="hover-content">
        <div class="hover-score">
          <span class="hover-score-value">{{ player.scores.overall }}</span>
          <span class="hover-score-label">Score global</span>
        </div>
        <div class="hover-actions">
          <button class="hover-action">
            <Eye :size="16" />
          </button>
          <button class="hover-action">
            <BarChart3 :size="16" />
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Radar, Line } from 'vue-chartjs'
import type { PlayerResult, RadarChartData } from '@/types'
import {
  User,
  Activity,
  Target,
  MapPin,
  BarChart3,
  Zap,
  TrendingUp,
  TrendingDown,
  ChevronDown,
  Eye
} from 'lucide-vue-next'

// Props
interface Props {
  player: PlayerResult
  isSelected?: boolean
  showEvolution?: boolean
  evolutionData?: Array<{ timestamp: number; score: number }>
  playerImage?: string
}

const props = withDefaults(defineProps<Props>(), {
  isSelected: false,
  showEvolution: false,
  evolutionData: () => []
})

// Emits
const emit = defineEmits<{
  click: [player: PlayerResult]
  viewDetails: [player: PlayerResult]
  viewHeatmap: [player: PlayerResult]
  compare: [player: PlayerResult]
}>()

// State
const showEvolutionChart = ref(false)

// Computed
const playerName = computed(() => {
  // Could be enhanced to fetch actual player names from a database
  return `Joueur ${props.player.jersey_number}`
})

const teamLabel = computed(() => {
  return props.player.team === 'home' ? 'Domicile' : 'Extérieur'
})

const radarData = computed((): RadarChartData => ({
  labels: ['Biomécanique', 'Technique', 'Tactique', 'Endurance', 'Précision'],
  datasets: [{
    label: 'Performance',
    data: [
      props.player.scores.biomechanics,
      props.player.scores.technical,
      props.player.scores.tactical,
      // Mock additional data - would come from metrics in real app
      props.player.metrics?.distance_covered ? Math.min(100, props.player.metrics.distance_covered / 100) : 75,
      props.player.metrics?.pass_accuracy || 80
    ],
    backgroundColor: getRadarBackgroundColor(),
    borderColor: getRadarBorderColor(),
    pointBackgroundColor: getRadarBorderColor(),
    pointBorderColor: '#fff',
    pointHoverBackgroundColor: '#fff',
    pointHoverBorderColor: getRadarBorderColor()
  }]
}))

const evolutionChartData = computed(() => ({
  labels: props.evolutionData.map((_, index) => `Match ${index + 1}`),
  datasets: [{
    label: 'Score global',
    data: props.evolutionData.map(item => item.score),
    borderColor: getRadarBorderColor(),
    backgroundColor: getRadarBackgroundColor(),
    tension: 0.4,
    fill: true
  }]
}))

// Chart Options
const radarOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: false
    },
    tooltip: {
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      titleColor: 'white',
      bodyColor: 'white',
      borderColor: getRadarBorderColor(),
      borderWidth: 1
    }
  },
  scales: {
    r: {
      beginAtZero: true,
      max: 100,
      ticks: {
        display: false
      },
      grid: {
        color: 'rgba(255, 255, 255, 0.1)'
      },
      angleLines: {
        color: 'rgba(255, 255, 255, 0.1)'
      },
      pointLabels: {
        color: 'rgba(255, 255, 255, 0.7)',
        font: {
          size: 10
        }
      }
    }
  },
  elements: {
    line: {
      borderWidth: 2
    },
    point: {
      radius: 3,
      hoverRadius: 5
    }
  }
}

const evolutionChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: false
    }
  },
  scales: {
    y: {
      beginAtZero: true,
      max: 100,
      display: false
    },
    x: {
      display: false
    }
  },
  elements: {
    point: {
      radius: 0
    }
  }
}

// Methods
function getScoreClass(score: number): string {
  if (score >= 90) return 'score-excellent'
  if (score >= 80) return 'score-very-good'
  if (score >= 70) return 'score-good'
  if (score >= 60) return 'score-average'
  return 'score-poor'
}

function getRadarBackgroundColor(): string {
  const colors = {
    home: 'rgba(34, 197, 94, 0.2)',
    away: 'rgba(239, 68, 68, 0.2)'
  }
  return colors[props.player.team as keyof typeof colors] || colors.home
}

function getRadarBorderColor(): string {
  const colors = {
    home: 'rgba(34, 197, 94, 1)',
    away: 'rgba(239, 68, 68, 1)'
  }
  return colors[props.player.team as keyof typeof colors] || colors.home
}

function formatDistance(meters: number): string {
  if (meters >= 1000) {
    return `${(meters / 1000).toFixed(1)}km`
  }
  return `${meters}m`
}

function toggleEvolution() {
  showEvolutionChart.value = !showEvolutionChart.value
}
</script>

<style scoped>
.player-card {
  @apply bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700;
  @apply hover:shadow-lg hover:scale-[1.02] transition-all duration-300 cursor-pointer relative overflow-hidden;
}

.player-card--home {
  @apply hover:border-green-300 dark:hover:border-green-600;
}

.player-card--away {
  @apply hover:border-red-300 dark:hover:border-red-600;
}

.player-card--selected {
  @apply ring-2 ring-blue-500 shadow-lg scale-[1.02];
}

.card-header {
  @apply flex items-start space-x-4 p-4 pb-2;
}

.player-avatar {
  @apply relative flex-shrink-0;
}

.avatar-image {
  @apply w-12 h-12 rounded-full object-cover;
}

.avatar-placeholder {
  @apply w-12 h-12 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center text-gray-500 dark:text-gray-400;
}

.jersey-badge {
  @apply absolute -top-1 -right-1 w-6 h-6 bg-blue-600 text-white text-xs font-bold rounded-full flex items-center justify-center;
}

.player-info {
  @apply flex-1 space-y-1;
}

.player-name {
  @apply font-semibold text-gray-900 dark:text-white text-sm;
}

.player-position {
  @apply text-xs text-gray-600 dark:text-gray-400 uppercase tracking-wide;
}

.team-indicator {
  @apply flex items-center space-x-2;
}

.team-color {
  @apply w-3 h-3 rounded-full;
}

.team-color--home {
  @apply bg-green-500;
}

.team-color--away {
  @apply bg-red-500;
}

.team-label {
  @apply text-xs text-gray-500 dark:text-gray-400;
}

.overall-score {
  @apply text-center space-y-1 flex-shrink-0;
}

.score-circle {
  @apply w-10 h-10 rounded-full flex items-center justify-center font-bold text-white text-sm;
}

.score-excellent {
  @apply bg-green-500;
}

.score-very-good {
  @apply bg-blue-500;
}

.score-good {
  @apply bg-yellow-500;
}

.score-average {
  @apply bg-orange-500;
}

.score-poor {
  @apply bg-red-500;
}

.score-value {
  @apply font-bold;
}

.score-label {
  @apply text-xs text-gray-500 dark:text-gray-400;
}

.radar-container {
  @apply px-4 py-2;
}

.radar-chart {
  @apply h-48 bg-gray-50 dark:bg-gray-900/50 rounded-lg p-2;
}

.key-stats {
  @apply px-4 pb-2 space-y-3;
}

.stat-item {
  @apply flex items-center space-x-3;
}

.stat-icon {
  @apply w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0;
}

.stat-icon.biomechanics {
  @apply bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400;
}

.stat-icon.technical {
  @apply bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400;
}

.stat-icon.tactical {
  @apply bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400;
}

.stat-content {
  @apply flex-1 space-y-1;
}

.stat-label {
  @apply text-xs font-medium text-gray-700 dark:text-gray-300;
}

.stat-bar {
  @apply w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden;
}

.stat-fill {
  @apply h-full transition-all duration-500 rounded-full;
}

.biomechanics-fill {
  @apply bg-gradient-to-r from-purple-400 to-purple-600;
}

.technical-fill {
  @apply bg-gradient-to-r from-blue-400 to-blue-600;
}

.tactical-fill {
  @apply bg-gradient-to-r from-green-400 to-green-600;
}

.stat-value {
  @apply text-xs font-semibold text-gray-900 dark:text-white;
}

.performance-metrics {
  @apply px-4 pb-2 space-y-3;
}

.metrics-header {
  @apply flex items-center space-x-2 text-xs font-medium text-gray-700 dark:text-gray-300;
}

.metrics-grid {
  @apply grid grid-cols-2 gap-3;
}

.metric-item {
  @apply flex items-center space-x-2;
}

.metric-icon {
  @apply text-gray-500 dark:text-gray-400;
}

.metric-content {
  @apply space-y-0.5;
}

.metric-value {
  @apply block text-sm font-semibold text-gray-900 dark:text-white;
}

.metric-label {
  @apply text-xs text-gray-600 dark:text-gray-400;
}

.evolution-section {
  @apply px-4 pb-2 space-y-3;
}

.evolution-header {
  @apply flex items-center justify-between text-xs font-medium text-gray-700 dark:text-gray-300;
}

.evolution-toggle {
  @apply text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors;
}

.evolution-chart {
  @apply h-20;
}

.performance-insights {
  @apply px-4 pb-2 space-y-3;
}

.insights-section {
  @apply space-y-2;
}

.insights-header {
  @apply flex items-center space-x-2 text-xs font-medium;
}

.strengths-header {
  @apply text-green-700 dark:text-green-400;
}

.weaknesses-header {
  @apply text-red-700 dark:text-red-400;
}

.insights-list {
  @apply space-y-1;
}

.insight-item {
  @apply text-xs px-2 py-1 rounded-md;
}

.strength-item {
  @apply bg-green-50 dark:bg-green-900/20 text-green-800 dark:text-green-300;
}

.weakness-item {
  @apply bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-300;
}

.card-actions {
  @apply flex space-x-2 px-4 pb-4;
}

.action-button {
  @apply flex items-center space-x-1 px-3 py-1.5 text-xs font-medium rounded-md transition-colors;
}

.action-button.primary {
  @apply bg-blue-600 hover:bg-blue-700 text-white;
}

.action-button.secondary {
  @apply border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700;
}

.hover-overlay {
  @apply absolute inset-0 bg-black bg-opacity-80 flex items-center justify-center opacity-0 transition-opacity duration-300 rounded-xl;
}

.player-card:hover .hover-overlay {
  @apply opacity-100;
}

.hover-content {
  @apply text-center space-y-4;
}

.hover-score {
  @apply space-y-1;
}

.hover-score-value {
  @apply text-3xl font-bold text-white;
}

.hover-score-label {
  @apply text-sm text-gray-300;
}

.hover-actions {
  @apply flex space-x-3;
}

.hover-action {
  @apply p-3 bg-white bg-opacity-20 hover:bg-opacity-30 text-white rounded-full transition-colors;
}
</style>