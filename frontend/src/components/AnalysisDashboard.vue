<template>
  <div class="analysis-dashboard">
    <!-- Header with Status -->
    <div class="dashboard-header">
      <div class="header-content">
        <div class="header-info">
          <h1 class="dashboard-title">
            Analyse de la vidéo
            <span v-if="results?.video_id" class="video-id">
              #{{ results.video_id }}
            </span>
          </h1>
          <p v-if="results?.generated_at" class="generated-time">
            Généré le {{ formatDateTime(results.generated_at) }}
          </p>
        </div>
        
        <div class="header-actions">
          <!-- Export Actions -->
          <div class="export-menu">
            <button
              @click="showExportMenu = !showExportMenu"
              class="export-button"
            >
              <Download :size="16" />
              Exporter
              <ChevronDown 
                :size="16" 
                :class="{ 'rotate-180': showExportMenu }"
                class="transition-transform"
              />
            </button>
            
            <div v-if="showExportMenu" class="export-dropdown">
              <button @click="exportPDF" class="export-option">
                <FileText :size="16" />
                Rapport PDF
              </button>
              <button @click="exportCSV" class="export-option">
                <Table :size="16" />
                Données CSV
              </button>
              <button @click="exportJSON" class="export-option">
                <Code :size="16" />
                Données JSON
              </button>
            </div>
          </div>
          
          <!-- Refresh Button -->
          <button @click="refreshData" class="refresh-button">
            <RotateCcw :size="16" />
            Actualiser
          </button>
        </div>
      </div>

      <!-- Analysis Progress (if still running) -->
      <div v-if="isAnalyzing" class="analysis-progress">
        <div class="progress-info">
          <div class="progress-text">
            <span class="progress-stage">{{ analysisStage }}</span>
            <span class="progress-percentage">{{ analysisProgress }}%</span>
          </div>
          <div v-if="analysisETA" class="progress-eta">
            Temps restant: {{ formatTime(analysisETA) }}
          </div>
        </div>
        
        <div class="progress-bar">
          <div 
            class="progress-fill"
            :style="{ width: `${analysisProgress}%` }"
          ></div>
        </div>
      </div>
    </div>

    <!-- Results Content -->
    <div v-if="results" class="dashboard-content">
      <!-- Summary Cards -->
      <div class="summary-cards">
        <div class="summary-card">
          <div class="card-icon players-icon">
            <Users :size="24" />
          </div>
          <div class="card-content">
            <h3 class="card-title">Joueurs analysés</h3>
            <p class="card-value">{{ results.summary.total_players_analyzed }}</p>
          </div>
        </div>

        <div class="summary-card">
          <div class="card-icon time-icon">
            <Clock :size="24" />
          </div>
          <div class="card-content">
            <h3 class="card-title">Durée vidéo</h3>
            <p class="card-value">{{ formatDuration(results.summary.video_duration) }}</p>
          </div>
        </div>

        <div class="summary-card">
          <div class="card-icon score-icon">
            <Trophy :size="24" />
          </div>
          <div class="card-content">
            <h3 class="card-title">Score global</h3>
            <p class="card-value">{{ results.overall_scores.overall_score }}/100</p>
          </div>
        </div>

        <div class="summary-card">
          <div class="card-icon frames-icon">
            <Film :size="24" />
          </div>
          <div class="card-content">
            <h3 class="card-title">Images analysées</h3>
            <p class="card-value">{{ results.summary.frames_processed.toLocaleString() }}</p>
          </div>
        </div>
      </div>

      <!-- Tabs Navigation -->
      <div class="tabs-navigation">
        <button
          v-for="tab in tabs"
          :key="tab.id"
          @click="setActiveTab(tab.id)"
          :class="[
            'tab-button',
            { 'tab-button--active': activeTab === tab.id }
          ]"
        >
          <component :is="tab.icon" :size="16" />
          <span>{{ tab.label }}</span>
          <span v-if="tab.badge" class="tab-badge">{{ tab.badge }}</span>
        </button>
      </div>

      <!-- Tab Content -->
      <div class="tab-content">
        <!-- Overview Tab -->
        <div v-if="activeTab === 'overview'" class="tab-panel">
          <div class="overview-grid">
            <!-- Overall Performance Chart -->
            <div class="chart-card">
              <h3 class="chart-title">Performance générale</h3>
              <div class="chart-container">
                <Radar
                  :data="overallPerformanceData"
                  :options="chartOptions.radar"
                />
              </div>
            </div>

            <!-- Team Comparison -->
            <div class="chart-card">
              <h3 class="chart-title">Comparaison équipes</h3>
              <div class="chart-container">
                <Bar
                  :data="teamComparisonData"
                  :options="chartOptions.bar"
                />
              </div>
            </div>

            <!-- Key Metrics -->
            <div class="metrics-card">
              <h3 class="card-title">Métriques clés</h3>
              <div class="metrics-list">
                <div class="metric-item">
                  <span class="metric-label">Cohésion d'équipe</span>
                  <div class="metric-bar">
                    <div 
                      class="metric-fill team-cohesion"
                      :style="{ width: `${results.overall_scores.team_cohesion}%` }"
                    ></div>
                  </div>
                  <span class="metric-value">{{ results.overall_scores.team_cohesion }}/100</span>
                </div>

                <div class="metric-item">
                  <span class="metric-label">Efficacité tactique</span>
                  <div class="metric-bar">
                    <div 
                      class="metric-fill tactical-efficiency"
                      :style="{ width: `${results.overall_scores.tactical_efficiency}%` }"
                    ></div>
                  </div>
                  <span class="metric-value">{{ results.overall_scores.tactical_efficiency }}/100</span>
                </div>

                <div class="metric-item">
                  <span class="metric-label">Performance individuelle</span>
                  <div class="metric-bar">
                    <div 
                      class="metric-fill individual-performance"
                      :style="{ width: `${results.overall_scores.individual_performance}%` }"
                    ></div>
                  </div>
                  <span class="metric-value">{{ results.overall_scores.individual_performance }}/100</span>
                </div>
              </div>
            </div>

            <!-- Key Moments -->
            <div v-if="results.key_moments" class="moments-card">
              <h3 class="card-title">Moments clés</h3>
              <div class="moments-list">
                <div
                  v-for="moment in results.key_moments.slice(0, 5)"
                  :key="moment.timestamp"
                  class="moment-item"
                  @click="seekToMoment(moment.timestamp)"
                >
                  <div class="moment-time">{{ formatDuration(moment.timestamp) }}</div>
                  <div class="moment-content">
                    <h4 class="moment-title">{{ moment.type }}</h4>
                    <p class="moment-description">{{ moment.description }}</p>
                  </div>
                  <div class="moment-impact" :class="getImpactClass(moment.score_impact)">
                    {{ moment.score_impact > 0 ? '+' : '' }}{{ moment.score_impact }}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Players Tab -->
        <div v-if="activeTab === 'players'" class="tab-panel">
          <!-- Players Filters -->
          <div class="filters-section">
            <div class="filters-row">
              <div class="filter-group">
                <label class="filter-label">Équipe:</label>
                <select v-model="playersFilter.team" @change="applyPlayersFilter" class="filter-select">
                  <option value="">Toutes</option>
                  <option value="home">Domicile</option>
                  <option value="away">Extérieur</option>
                </select>
              </div>

              <div class="filter-group">
                <label class="filter-label">Score minimum:</label>
                <input
                  v-model.number="playersFilter.minScore"
                  @input="applyPlayersFilter"
                  type="range"
                  min="0"
                  max="100"
                  class="filter-range"
                />
                <span class="filter-value">{{ playersFilter.minScore }}</span>
              </div>

              <div class="filter-group">
                <label class="filter-label">Recherche:</label>
                <input
                  v-model="playersFilter.search"
                  @input="applyPlayersFilter"
                  type="text"
                  placeholder="Numéro ou nom..."
                  class="filter-input"
                />
              </div>
            </div>
          </div>

          <!-- Players Grid -->
          <div class="players-grid">
            <PlayerCard
              v-for="player in filteredPlayers"
              :key="player.player_id"
              :player="player"
              @click="selectPlayer(player)"
            />
          </div>

          <!-- Pagination -->
          <div v-if="results.pagination" class="pagination">
            <button
              @click="goToPage(results.pagination.current_page - 1)"
              :disabled="!results.pagination.has_prev"
              class="pagination-button"
            >
              <ChevronLeft :size="16" />
              Précédent
            </button>
            
            <span class="pagination-info">
              Page {{ results.pagination.current_page }} sur {{ results.pagination.total_pages }}
            </span>
            
            <button
              @click="goToPage(results.pagination.current_page + 1)"
              :disabled="!results.pagination.has_next"
              class="pagination-button"
            >
              Suivant
              <ChevronRight :size="16" />
            </button>
          </div>
        </div>

        <!-- Team Analysis Tab -->
        <div v-if="activeTab === 'team'" class="tab-panel">
          <div class="team-analysis-grid">
            <!-- Formation Analysis -->
            <div class="chart-card full-width">
              <h3 class="chart-title">Analyse des formations</h3>
              <div class="formation-comparison">
                <div class="team-formation">
                  <h4>Équipe domicile</h4>
                  <div class="formation-display">
                    {{ results.team_statistics?.home_team?.formation || '4-4-2' }}
                  </div>
                  <div class="formation-stats">
                    <div class="stat-item">
                      <span>Possession:</span>
                      <span>{{ results.team_statistics?.home_team?.possession || 0 }}%</span>
                    </div>
                    <div class="stat-item">
                      <span>Précision passes:</span>
                      <span>{{ results.team_statistics?.home_team?.pass_accuracy || 0 }}%</span>
                    </div>
                  </div>
                </div>

                <div class="vs-divider">VS</div>

                <div class="team-formation">
                  <h4>Équipe extérieur</h4>
                  <div class="formation-display">
                    {{ results.team_statistics?.away_team?.formation || '4-3-3' }}
                  </div>
                  <div class="formation-stats">
                    <div class="stat-item">
                      <span>Possession:</span>
                      <span>{{ results.team_statistics?.away_team?.possession || 0 }}%</span>
                    </div>
                    <div class="stat-item">
                      <span>Précision passes:</span>
                      <span>{{ results.team_statistics?.away_team?.pass_accuracy || 0 }}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Team Performance Chart -->
            <div class="chart-card">
              <h3 class="chart-title">Performance comparative</h3>
              <div class="chart-container">
                <Line
                  :data="teamPerformanceData"
                  :options="chartOptions.line"
                />
              </div>
            </div>

            <!-- Territorial Map -->
            <div class="chart-card">
              <h3 class="chart-title">Contrôle territorial</h3>
              <div class="territorial-map">
                <!-- Placeholder for heatmap -->
                <div class="heatmap-placeholder">
                  <MapPin :size="48" class="text-gray-400" />
                  <p>Carte de chaleur à implémenter</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Technical Analysis Tab -->
        <div v-if="activeTab === 'technical'" class="tab-panel">
          <div class="technical-grid">
            <!-- Actions Distribution -->
            <div class="chart-card">
              <h3 class="chart-title">Répartition des actions</h3>
              <div class="chart-container">
                <Doughnut
                  :data="actionsDistributionData"
                  :options="chartOptions.doughnut"
                />
              </div>
            </div>

            <!-- Technical Skills Radar -->
            <div class="chart-card">
              <h3 class="chart-title">Compétences techniques</h3>
              <div class="chart-container">
                <Radar
                  :data="technicalSkillsData"
                  :options="chartOptions.radar"
                />
              </div>
            </div>

            <!-- Performance Over Time -->
            <div class="chart-card full-width">
              <h3 class="chart-title">Évolution de la performance</h3>
              <div class="chart-container">
                <Line
                  :data="performanceOverTimeData"
                  :options="chartOptions.line"
                />
              </div>
            </div>
          </div>
        </div>

        <!-- Tactical Analysis Tab -->
        <div v-if="activeTab === 'tactical'" class="tab-panel">
          <div class="tactical-grid">
            <div class="chart-card">
              <h3 class="chart-title">Positionnement tactique</h3>
              <div class="tactical-positioning">
                <!-- Placeholder for tactical positioning -->
                <div class="positioning-placeholder">
                  <Target :size="48" class="text-gray-400" />
                  <p>Analyse tactique à implémenter</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Empty State -->
    <div v-else-if="!isAnalyzing" class="empty-state">
      <div class="empty-content">
        <BarChart3 :size="64" class="text-gray-400" />
        <h3 class="empty-title">Aucune analyse disponible</h3>
        <p class="empty-description">
          Uploadez une vidéo et démarrez l'analyse pour voir les résultats ici.
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { useAppStore } from '@/stores'
import { Radar, Bar, Line, Doughnut } from 'vue-chartjs'
import PlayerCard from './PlayerCard.vue'
import type { AnalysisResults, PlayerResult, ChartDataPoint } from '@/types'
import {
  Download,
  ChevronDown,
  FileText,
  Table,
  Code,
  RotateCcw,
  Users,
  Clock,
  Trophy,
  Film,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  MapPin,
  Target
} from 'lucide-vue-next'

// Props
interface Props {
  jobId?: string
}

const props = defineProps<Props>()

// Emits
const emit = defineEmits<{
  playerSelected: [player: PlayerResult]
  momentSelected: [timestamp: number]
}>()

// Store
const store = useAppStore()

// State
const showExportMenu = ref(false)
const activeTab = ref('overview')
const playersFilter = reactive({
  team: '',
  minScore: 0,
  search: ''
})

// Computed
const results = computed(() => store.analysisResults)
const isAnalyzing = computed(() => store.isAnalyzing)
const analysisProgress = computed(() => store.analysisProgress)
const analysisStage = computed(() => 
  store.currentAnalysis?.current_stage || 'En cours...'
)
const analysisETA = computed(() => 
  store.currentAnalysis?.eta_seconds
)

const tabs = computed(() => [
  {
    id: 'overview',
    label: 'Vue d\'ensemble',
    icon: 'BarChart3',
    badge: null
  },
  {
    id: 'players',
    label: 'Joueurs',
    icon: 'Users',
    badge: results.value?.summary.total_players_analyzed || null
  },
  {
    id: 'team',
    label: 'Équipes',
    icon: 'Target',
    badge: null
  },
  {
    id: 'technical',
    label: 'Technique',
    icon: 'Activity',
    badge: results.value?.summary.actions_classified || null
  },
  {
    id: 'tactical',
    label: 'Tactique',
    icon: 'MapPin',
    badge: null
  }
])

const filteredPlayers = computed(() => {
  if (!results.value?.players) return []
  
  return results.value.players.filter(player => {
    const matchesTeam = !playersFilter.team || player.team === playersFilter.team
    const matchesScore = player.scores.overall >= playersFilter.minScore
    const matchesSearch = !playersFilter.search || 
      player.jersey_number.toString().includes(playersFilter.search) ||
      player.position.toLowerCase().includes(playersFilter.search.toLowerCase())
    
    return matchesTeam && matchesScore && matchesSearch
  })
})

// Chart Options
const chartOptions = {
  radar: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
      }
    },
    scales: {
      r: {
        beginAtZero: true,
        max: 100
      }
    }
  },
  bar: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
      }
    }
  },
  line: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
      }
    }
  },
  doughnut: {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
      }
    }
  }
}

// Chart Data
const overallPerformanceData = computed(() => ({
  labels: ['Biomécanique', 'Technique', 'Tactique', 'Cohésion', 'Efficacité'],
  datasets: [{
    label: 'Performance globale',
    data: [
      results.value?.overall_scores.overall_score || 0,
      results.value?.overall_scores.overall_score || 0,
      results.value?.overall_scores.tactical_efficiency || 0,
      results.value?.overall_scores.team_cohesion || 0,
      results.value?.overall_scores.individual_performance || 0
    ],
    backgroundColor: 'rgba(59, 130, 246, 0.2)',
    borderColor: 'rgba(59, 130, 246, 1)',
    pointBackgroundColor: 'rgba(59, 130, 246, 1)'
  }]
}))

const teamComparisonData = computed(() => ({
  labels: ['Possession', 'Passes', 'Tirs', 'Distance'],
  datasets: [
    {
      label: 'Domicile',
      data: [
        results.value?.team_statistics?.home_team?.possession || 0,
        results.value?.team_statistics?.home_team?.pass_accuracy || 0,
        results.value?.team_statistics?.home_team?.shots || 0,
        results.value?.team_statistics?.home_team?.distance_covered || 0
      ],
      backgroundColor: 'rgba(16, 185, 129, 0.8)'
    },
    {
      label: 'Extérieur',
      data: [
        results.value?.team_statistics?.away_team?.possession || 0,
        results.value?.team_statistics?.away_team?.pass_accuracy || 0,
        results.value?.team_statistics?.away_team?.shots || 0,
        results.value?.team_statistics?.away_team?.distance_covered || 0
      ],
      backgroundColor: 'rgba(239, 68, 68, 0.8)'
    }
  ]
}))

const teamPerformanceData = computed(() => ({
  labels: ['0-15min', '15-30min', '30-45min', '45-60min', '60-75min', '75-90min'],
  datasets: [
    {
      label: 'Domicile',
      data: [75, 78, 82, 80, 76, 79],
      borderColor: 'rgba(16, 185, 129, 1)',
      backgroundColor: 'rgba(16, 185, 129, 0.1)'
    },
    {
      label: 'Extérieur',
      data: [72, 74, 76, 78, 81, 83],
      borderColor: 'rgba(239, 68, 68, 1)',
      backgroundColor: 'rgba(239, 68, 68, 0.1)'
    }
  ]
}))

const actionsDistributionData = computed(() => ({
  labels: ['Passes', 'Tirs', 'Tacles', 'Dribbles', 'Centres'],
  datasets: [{
    data: [45, 15, 20, 12, 8],
    backgroundColor: [
      'rgba(59, 130, 246, 0.8)',
      'rgba(16, 185, 129, 0.8)',
      'rgba(245, 158, 11, 0.8)',
      'rgba(139, 92, 246, 0.8)',
      'rgba(236, 72, 153, 0.8)'
    ]
  }]
}))

const technicalSkillsData = computed(() => ({
  labels: ['Précision', 'Vitesse', 'Contrôle', 'Puissance', 'Finition'],
  datasets: [{
    label: 'Compétences techniques',
    data: [85, 78, 92, 76, 82],
    backgroundColor: 'rgba(139, 92, 246, 0.2)',
    borderColor: 'rgba(139, 92, 246, 1)',
    pointBackgroundColor: 'rgba(139, 92, 246, 1)'
  }]
}))

const performanceOverTimeData = computed(() => ({
  labels: Array.from({ length: 90 }, (_, i) => `${i + 1}min`),
  datasets: [{
    label: 'Performance moyenne',
    data: Array.from({ length: 90 }, () => Math.random() * 20 + 70),
    borderColor: 'rgba(59, 130, 246, 1)',
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
    tension: 0.4
  }]
}))

// Methods
function setActiveTab(tabId: string) {
  activeTab.value = tabId
  store.setSelectedTab(tabId)
}

function selectPlayer(player: PlayerResult) {
  emit('playerSelected', player)
}

function seekToMoment(timestamp: number) {
  emit('momentSelected', timestamp)
}

function applyPlayersFilter() {
  // Filters are applied through computed property
  // Could add API call for server-side filtering here
}

async function goToPage(page: number) {
  if (!props.jobId) return
  
  try {
    await store.loadResults(props.jobId, {
      page,
      ...playersFilter
    })
  } catch (error) {
    console.error('Error loading page:', error)
  }
}

async function refreshData() {
  if (!props.jobId) return
  
  try {
    await store.loadResults(props.jobId)
  } catch (error) {
    console.error('Error refreshing data:', error)
  }
}

async function exportPDF() {
  if (!props.jobId) return
  
  try {
    await store.generateReport(props.jobId, {
      template: 'standard',
      language: 'fr',
      include_charts: true,
      include_heatmaps: true
    })
  } catch (error) {
    console.error('Error generating PDF:', error)
  }
  showExportMenu.value = false
}

function exportCSV() {
  if (!results.value) return
  
  // Generate CSV data
  const csvData = results.value.players.map(player => ({
    'Numéro': player.jersey_number,
    'Équipe': player.team,
    'Position': player.position,
    'Score Global': player.scores.overall,
    'Biomécanique': player.scores.biomechanics,
    'Technique': player.scores.technical,
    'Tactique': player.scores.tactical,
    ...(player.metrics && {
      'Distance': player.metrics.distance_covered,
      'Vitesse max': player.metrics.top_speed,
      'Précision passes': player.metrics.pass_accuracy
    })
  }))
  
  downloadCSV(csvData, `football_analysis_${props.jobId}.csv`)
  showExportMenu.value = false
}

function exportJSON() {
  if (!results.value) return
  
  const jsonData = JSON.stringify(results.value, null, 2)
  downloadFile(jsonData, `football_analysis_${props.jobId}.json`, 'application/json')
  showExportMenu.value = false
}

function getImpactClass(impact: number): string {
  if (impact > 0) return 'positive-impact'
  if (impact < 0) return 'negative-impact'
  return 'neutral-impact'
}

// Utility functions
function formatDateTime(dateString: string): string {
  return new Date(dateString).toLocaleString('fr-FR')
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

function downloadCSV(data: any[], filename: string) {
  if (!data.length) return
  
  const headers = Object.keys(data[0])
  const csv = [
    headers.join(','),
    ...data.map(row => headers.map(header => `"${row[header] || ''}"`).join(','))
  ].join('\n')
  
  downloadFile(csv, filename, 'text/csv')
}

function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

// Lifecycle
onMounted(() => {
  if (props.jobId && !results.value) {
    store.loadResults(props.jobId)
  }
})

watch(() => props.jobId, (newJobId) => {
  if (newJobId) {
    store.loadResults(newJobId)
  }
})
</script>

<style scoped>
.analysis-dashboard {
  @apply w-full h-full flex flex-col space-y-6;
}

.dashboard-header {
  @apply bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700;
}

.header-content {
  @apply flex justify-between items-start;
}

.header-info {
  @apply space-y-1;
}

.dashboard-title {
  @apply text-2xl font-bold text-gray-900 dark:text-white;
}

.video-id {
  @apply text-lg text-blue-600 dark:text-blue-400 font-mono;
}

.generated-time {
  @apply text-sm text-gray-500 dark:text-gray-400;
}

.header-actions {
  @apply flex items-center space-x-4;
}

.export-menu {
  @apply relative;
}

.export-button {
  @apply flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors;
}

.export-dropdown {
  @apply absolute right-0 top-full mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 py-2 z-50;
}

.export-option {
  @apply flex items-center space-x-2 w-full px-4 py-2 text-left text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors;
}

.refresh-button {
  @apply flex items-center space-x-2 px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-lg transition-colors;
}

.analysis-progress {
  @apply mt-6 pt-6 border-t border-gray-200 dark:border-gray-700;
}

.progress-info {
  @apply flex justify-between items-center mb-2;
}

.progress-text {
  @apply flex items-center space-x-3;
}

.progress-stage {
  @apply font-medium text-gray-700 dark:text-gray-200;
}

.progress-percentage {
  @apply text-lg font-bold text-blue-600 dark:text-blue-400;
}

.progress-eta {
  @apply text-sm text-gray-500 dark:text-gray-400;
}

.progress-bar {
  @apply w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden;
}

.progress-fill {
  @apply h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-300;
}

.dashboard-content {
  @apply space-y-6;
}

.summary-cards {
  @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6;
}

.summary-card {
  @apply bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700 flex items-center space-x-4;
}

.card-icon {
  @apply p-3 rounded-lg;
}

.players-icon {
  @apply bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400;
}

.time-icon {
  @apply bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400;
}

.score-icon {
  @apply bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400;
}

.frames-icon {
  @apply bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400;
}

.card-content {
  @apply space-y-1;
}

.card-title {
  @apply text-sm font-medium text-gray-600 dark:text-gray-300;
}

.card-value {
  @apply text-2xl font-bold text-gray-900 dark:text-white;
}

.tabs-navigation {
  @apply flex space-x-1 bg-gray-100 dark:bg-gray-700 rounded-lg p-1;
}

.tab-button {
  @apply flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-colors relative;
  @apply text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white;
}

.tab-button--active {
  @apply bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 shadow-sm;
}

.tab-badge {
  @apply px-2 py-0.5 text-xs font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded-full;
}

.tab-content {
  @apply bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-200 dark:border-gray-700;
}

.tab-panel {
  @apply space-y-6;
}

.overview-grid,
.technical-grid,
.tactical-grid {
  @apply grid grid-cols-1 lg:grid-cols-2 gap-6;
}

.team-analysis-grid {
  @apply space-y-6;
}

.chart-card {
  @apply bg-gray-50 dark:bg-gray-900/50 rounded-lg p-6 space-y-4;
}

.chart-card.full-width {
  @apply lg:col-span-2;
}

.chart-title {
  @apply text-lg font-semibold text-gray-900 dark:text-white;
}

.chart-container {
  @apply h-64;
}

.metrics-card,
.moments-card {
  @apply bg-gray-50 dark:bg-gray-900/50 rounded-lg p-6 space-y-4;
}

.metrics-list {
  @apply space-y-4;
}

.metric-item {
  @apply flex items-center space-x-4;
}

.metric-label {
  @apply text-sm font-medium text-gray-700 dark:text-gray-300 min-w-[8rem];
}

.metric-bar {
  @apply flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden;
}

.metric-fill {
  @apply h-full transition-all duration-500;
}

.metric-fill.team-cohesion {
  @apply bg-gradient-to-r from-blue-500 to-blue-600;
}

.metric-fill.tactical-efficiency {
  @apply bg-gradient-to-r from-green-500 to-green-600;
}

.metric-fill.individual-performance {
  @apply bg-gradient-to-r from-purple-500 to-purple-600;
}

.metric-value {
  @apply text-sm font-semibold text-gray-900 dark:text-white min-w-[3rem] text-right;
}

.moments-list {
  @apply space-y-3 max-h-80 overflow-y-auto;
}

.moment-item {
  @apply flex items-center space-x-4 p-3 bg-white dark:bg-gray-800 rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors;
}

.moment-time {
  @apply text-sm font-mono text-blue-600 dark:text-blue-400 min-w-[4rem];
}

.moment-content {
  @apply flex-1 space-y-1;
}

.moment-title {
  @apply font-medium text-gray-900 dark:text-white;
}

.moment-description {
  @apply text-sm text-gray-600 dark:text-gray-300;
}

.moment-impact {
  @apply text-sm font-bold px-2 py-1 rounded;
}

.positive-impact {
  @apply text-green-700 dark:text-green-400 bg-green-100 dark:bg-green-900/30;
}

.negative-impact {
  @apply text-red-700 dark:text-red-400 bg-red-100 dark:bg-red-900/30;
}

.neutral-impact {
  @apply text-gray-700 dark:text-gray-400 bg-gray-100 dark:bg-gray-900/30;
}

.filters-section {
  @apply bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 mb-6;
}

.filters-row {
  @apply flex flex-wrap gap-4;
}

.filter-group {
  @apply flex items-center space-x-2;
}

.filter-label {
  @apply text-sm font-medium text-gray-700 dark:text-gray-300;
}

.filter-select,
.filter-input {
  @apply px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700;
}

.filter-range {
  @apply w-24;
}

.filter-value {
  @apply text-sm font-mono text-gray-600 dark:text-gray-300 min-w-[2rem] text-right;
}

.players-grid {
  @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6;
}

.pagination {
  @apply flex justify-center items-center space-x-4 mt-8;
}

.pagination-button {
  @apply flex items-center space-x-2 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors;
}

.pagination-info {
  @apply text-sm text-gray-600 dark:text-gray-300;
}

.formation-comparison {
  @apply flex items-center justify-between space-x-8;
}

.team-formation {
  @apply text-center space-y-4;
}

.formation-display {
  @apply text-3xl font-bold text-blue-600 dark:text-blue-400;
}

.formation-stats {
  @apply space-y-2;
}

.stat-item {
  @apply flex justify-between text-sm;
}

.vs-divider {
  @apply text-2xl font-bold text-gray-400 dark:text-gray-500;
}

.territorial-map,
.heatmap-placeholder,
.positioning-placeholder {
  @apply h-64 flex flex-col items-center justify-center text-center space-y-4;
}

.empty-state {
  @apply flex items-center justify-center min-h-[400px];
}

.empty-content {
  @apply text-center space-y-4;
}

.empty-title {
  @apply text-xl font-semibold text-gray-700 dark:text-gray-200;
}

.empty-description {
  @apply text-gray-500 dark:text-gray-400 max-w-md;
}
</style>