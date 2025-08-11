<template>
  <div class="video-player-container">
    <!-- Video Player -->
    <div class="video-player-wrapper" :class="{ 'fullscreen': isFullscreen }">
      <div
        ref="playerContainer"
        class="video-player"
        @mousemove="handleMouseMove"
        @mouseleave="hideControls"
        @click="togglePlayPause"
      >
        <!-- Main Video Element -->
        <video
          ref="videoElement"
          class="video-element"
          :src="videoSrc"
          @loadedmetadata="handleVideoLoaded"
          @timeupdate="handleTimeUpdate"
          @play="handlePlay"
          @pause="handlePause"
          @ended="handleEnded"
          @seeking="handleSeeking"
          @seeked="handleSeeked"
          playsinline
        />

        <!-- Annotations Overlay -->
        <div class="annotations-overlay" v-if="showAnnotations && currentAnnotations.length">
          <div
            v-for="annotation in currentAnnotations"
            :key="annotation.id"
            class="annotation"
            :class="[
              `annotation--${annotation.type}`,
              { 'annotation--active': annotation.id === selectedAnnotationId }
            ]"
            :style="getAnnotationPosition(annotation)"
            @click.stop="selectAnnotation(annotation)"
          >
            <div class="annotation-marker">
              <component :is="getAnnotationIcon(annotation.type)" :size="16" />
            </div>
            
            <div class="annotation-tooltip" v-if="annotation.id === selectedAnnotationId">
              <div class="tooltip-header">
                <h4 class="tooltip-title">{{ annotation.title }}</h4>
                <button @click.stop="closeAnnotation" class="tooltip-close">
                  <X :size="14" />
                </button>
              </div>
              <p class="tooltip-description">{{ annotation.description }}</p>
              <div v-if="annotation.players?.length" class="tooltip-players">
                <span class="players-label">Joueurs:</span>
                <span class="players-list">{{ annotation.players.join(', ') }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Loading Spinner -->
        <div v-if="isLoading" class="loading-overlay">
          <div class="loading-spinner">
            <Loader2 :size="32" class="animate-spin" />
          </div>
        </div>

        <!-- Controls Overlay -->
        <div
          class="controls-overlay"
          :class="{ 'controls-overlay--visible': showControls || !isPlaying }"
        >
          <!-- Main Controls -->
          <div class="main-controls">
            <!-- Play/Pause Button -->
            <button @click="togglePlayPause" class="control-button control-button--primary">
              <Play v-if="!isPlaying" :size="24" />
              <Pause v-else :size="24" />
            </button>

            <!-- Playback Speed -->
            <div class="playback-speed">
              <button
                @click="showSpeedMenu = !showSpeedMenu"
                class="control-button"
                :class="{ 'active': playbackSpeed !== 1 }"
              >
                {{ playbackSpeed }}x
              </button>
              
              <div v-if="showSpeedMenu" class="speed-menu">
                <button
                  v-for="speed in playbackSpeeds"
                  :key="speed"
                  @click="setPlaybackSpeed(speed)"
                  class="speed-option"
                  :class="{ 'active': playbackSpeed === speed }"
                >
                  {{ speed }}x
                </button>
              </div>
            </div>

            <!-- Frame Navigation -->
            <div class="frame-controls">
              <button @click="previousFrame" class="control-button" title="Image précédente">
                <SkipBack :size="16" />
              </button>
              <button @click="nextFrame" class="control-button" title="Image suivante">
                <SkipForward :size="16" />
              </button>
            </div>

            <!-- Time Display -->
            <div class="time-display">
              <span class="current-time">{{ formatTime(currentTime) }}</span>
              <span class="time-separator">/</span>
              <span class="total-time">{{ formatTime(duration) }}</span>
            </div>

            <!-- Spacer -->
            <div class="controls-spacer"></div>

            <!-- Annotations Toggle -->
            <button
              @click="toggleAnnotations"
              class="control-button"
              :class="{ 'active': showAnnotations }"
              title="Afficher/Masquer les annotations"
            >
              <Eye v-if="showAnnotations" :size="16" />
              <EyeOff v-else :size="16" />
            </button>

            <!-- Settings -->
            <button
              @click="showSettings = !showSettings"
              class="control-button"
              title="Paramètres"
            >
              <Settings :size="16" />
            </button>

            <!-- Fullscreen -->
            <button @click="toggleFullscreen" class="control-button" title="Plein écran">
              <Maximize v-if="!isFullscreen" :size="16" />
              <Minimize v-else :size="16" />
            </button>
          </div>

          <!-- Progress Bar -->
          <div class="progress-container">
            <!-- Timeline Events -->
            <div class="timeline-events">
              <div
                v-for="event in timelineEvents"
                :key="event.id"
                class="timeline-event"
                :class="`timeline-event--${event.type}`"
                :style="{ left: `${(event.timestamp / duration) * 100}%` }"
                @click="seekTo(event.timestamp)"
                :title="`${formatTime(event.timestamp)} - ${event.title}`"
              >
                <div class="event-marker"></div>
              </div>
            </div>

            <!-- Progress Bar -->
            <div
              ref="progressBar"
              class="progress-bar"
              @click="handleProgressClick"
              @mousedown="handleProgressMouseDown"
            >
              <!-- Background -->
              <div class="progress-bg"></div>
              
              <!-- Buffered -->
              <div
                class="progress-buffered"
                :style="{ width: `${bufferedPercent}%` }"
              ></div>
              
              <!-- Played -->
              <div
                class="progress-played"
                :style="{ width: `${progressPercent}%` }"
              ></div>
              
              <!-- Scrubber -->
              <div
                class="progress-scrubber"
                :style="{ left: `${progressPercent}%` }"
              ></div>
              
              <!-- Preview Tooltip -->
              <div
                v-if="showPreviewTooltip"
                class="preview-tooltip"
                :style="{ left: `${previewPosition}%` }"
              >
                {{ formatTime(previewTime) }}
              </div>
            </div>
          </div>
        </div>

        <!-- Settings Panel -->
        <div v-if="showSettings" class="settings-panel">
          <div class="settings-header">
            <h3>Paramètres</h3>
            <button @click="showSettings = false" class="settings-close">
              <X :size="16" />
            </button>
          </div>
          
          <div class="settings-content">
            <div class="setting-group">
              <label class="setting-label">Volume</label>
              <div class="volume-control">
                <button @click="toggleMute" class="volume-button">
                  <Volume2 v-if="volume > 0.5 && !isMuted" :size="16" />
                  <Volume1 v-else-if="volume > 0 && !isMuted" :size="16" />
                  <VolumeX v-else :size="16" />
                </button>
                <input
                  v-model.number="volume"
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  class="volume-slider"
                  @input="updateVolume"
                />
                <span class="volume-value">{{ Math.round(volume * 100) }}%</span>
              </div>
            </div>

            <div class="setting-group">
              <label class="setting-label">
                <input
                  v-model="autoplay"
                  type="checkbox"
                  class="setting-checkbox"
                />
                Lecture automatique
              </label>
            </div>

            <div class="setting-group">
              <label class="setting-label">
                <input
                  v-model="loop"
                  type="checkbox"
                  class="setting-checkbox"
                />
                Lecture en boucle
              </label>
            </div>

            <div class="setting-group">
              <label class="setting-label">Qualité vidéo</label>
              <select v-model="selectedQuality" class="quality-select">
                <option value="auto">Automatique</option>
                <option value="1080p">1080p</option>
                <option value="720p">720p</option>
                <option value="480p">480p</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Video Info -->
    <div class="video-info">
      <div class="video-meta">
        <h3 class="video-title">{{ videoTitle || 'Analyse vidéo football' }}</h3>
        <div class="video-details">
          <span class="detail-item">
            <Clock :size="14" />
            {{ formatTime(duration) }}
          </span>
          <span class="detail-item">
            <Monitor :size="14" />
            {{ videoResolution }}
          </span>
          <span v-if="fps" class="detail-item">
            <Zap :size="14" />
            {{ fps }} FPS
          </span>
        </div>
      </div>

      <!-- Playback Stats -->
      <div v-if="showStats" class="playback-stats">
        <div class="stat-item">
          <span class="stat-label">Vitesse:</span>
          <span class="stat-value">{{ playbackSpeed }}x</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Image:</span>
          <span class="stat-value">{{ currentFrame }}/{{ totalFrames }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Annotations:</span>
          <span class="stat-value">{{ annotations.length }}</span>
        </div>
      </div>
    </div>

    <!-- Keyboard Shortcuts Help -->
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
            <kbd>Espace</kbd>
            <span>Lecture/Pause</span>
          </div>
          <div class="shortcut-item">
            <kbd>←/→</kbd>
            <span>Avancer/Reculer (5s)</span>
          </div>
          <div class="shortcut-item">
            <kbd>↑/↓</kbd>
            <span>Volume +/-</span>
          </div>
          <div class="shortcut-item">
            <kbd>,/.</kbd>
            <span>Image par image</span>
          </div>
          <div class="shortcut-item">
            <kbd>F</kbd>
            <span>Plein écran</span>
          </div>
          <div class="shortcut-item">
            <kbd>A</kbd>
            <span>Annotations</span>
          </div>
          <div class="shortcut-item">
            <kbd>?</kbd>
            <span>Afficher les raccourcis</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import type { VideoAnnotation, VideoPlayerState } from '@/types'
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Eye,
  EyeOff,
  Settings,
  Maximize,
  Minimize,
  Volume2,
  Volume1,
  VolumeX,
  Clock,
  Monitor,
  Zap,
  X,
  Loader2,
  Target,
  AlertCircle,
  Flag,
  Users,
  CreditCard
} from 'lucide-vue-next'

// Props
interface Props {
  videoSrc: string
  videoTitle?: string
  annotations?: VideoAnnotation[]
  timelineEvents?: any[]
  autoplay?: boolean
  showStats?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  autoplay: false,
  showStats: false,
  annotations: () => [],
  timelineEvents: () => []
})

// Emits
const emit = defineEmits<{
  play: [currentTime: number]
  pause: [currentTime: number]
  seeking: [time: number]
  seeked: [time: number]
  timeUpdate: [currentTime: number]
  annotationSelected: [annotation: VideoAnnotation]
  speedChanged: [speed: number]
  fullscreenChanged: [isFullscreen: boolean]
}>()

// Refs
const videoElement = ref<HTMLVideoElement>()
const playerContainer = ref<HTMLElement>()
const progressBar = ref<HTMLElement>()

// Player State
const isPlaying = ref(false)
const isLoading = ref(false)
const currentTime = ref(0)
const duration = ref(0)
const bufferedPercent = ref(0)
const volume = ref(1)
const isMuted = ref(false)
const previousVolume = ref(1)
const playbackSpeed = ref(1)
const isFullscreen = ref(false)
const videoResolution = ref('')
const fps = ref(0)

// UI State
const showControls = ref(true)
const showAnnotations = ref(true)
const showSettings = ref(false)
const showSpeedMenu = ref(false)
const showShortcuts = ref(false)
const showPreviewTooltip = ref(false)
const selectedAnnotationId = ref<string | null>(null)
const controlsTimeout = ref<NodeJS.Timeout>()

// Preview State
const previewPosition = ref(0)
const previewTime = ref(0)
const isDragging = ref(false)

// Settings
const autoplay = ref(props.autoplay)
const loop = ref(false)
const selectedQuality = ref('auto')

// Constants
const playbackSpeeds = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]

// Computed
const progressPercent = computed(() => 
  duration.value > 0 ? (currentTime.value / duration.value) * 100 : 0
)

const currentFrame = computed(() => 
  fps.value > 0 ? Math.floor(currentTime.value * fps.value) : 0
)

const totalFrames = computed(() => 
  fps.value > 0 ? Math.floor(duration.value * fps.value) : 0
)

const currentAnnotations = computed(() => {
  return props.annotations.filter(annotation => {
    const tolerance = 1 // 1 second tolerance
    return Math.abs(annotation.timestamp - currentTime.value) <= tolerance
  })
})

// Methods
function handleVideoLoaded() {
  if (!videoElement.value) return
  
  duration.value = videoElement.value.duration
  videoResolution.value = `${videoElement.value.videoWidth}x${videoElement.value.videoHeight}`
  
  // Try to detect FPS (approximation)
  fps.value = 25 // Default, would need more sophisticated detection
  
  if (autoplay.value) {
    play()
  }
}

function handleTimeUpdate() {
  if (!videoElement.value || isDragging.value) return
  
  currentTime.value = videoElement.value.currentTime
  updateBuffered()
  emit('timeUpdate', currentTime.value)
}

function handlePlay() {
  isPlaying.value = true
  emit('play', currentTime.value)
  hideControlsAfterDelay()
}

function handlePause() {
  isPlaying.value = false
  emit('pause', currentTime.value)
  showControls.value = true
  clearTimeout(controlsTimeout.value)
}

function handleEnded() {
  isPlaying.value = false
  if (loop.value) {
    seekTo(0)
    play()
  }
}

function handleSeeking() {
  isLoading.value = true
  emit('seeking', currentTime.value)
}

function handleSeeked() {
  isLoading.value = false
  emit('seeked', currentTime.value)
}

function togglePlayPause() {
  if (isPlaying.value) {
    pause()
  } else {
    play()
  }
}

function play() {
  if (videoElement.value) {
    videoElement.value.play()
  }
}

function pause() {
  if (videoElement.value) {
    videoElement.value.pause()
  }
}

function seekTo(time: number) {
  if (videoElement.value) {
    videoElement.value.currentTime = Math.max(0, Math.min(time, duration.value))
  }
}

function previousFrame() {
  if (fps.value > 0) {
    const frameTime = 1 / fps.value
    seekTo(currentTime.value - frameTime)
  }
}

function nextFrame() {
  if (fps.value > 0) {
    const frameTime = 1 / fps.value
    seekTo(currentTime.value + frameTime)
  }
}

function setPlaybackSpeed(speed: number) {
  playbackSpeed.value = speed
  if (videoElement.value) {
    videoElement.value.playbackRate = speed
  }
  showSpeedMenu.value = false
  emit('speedChanged', speed)
}

function updateVolume() {
  if (videoElement.value) {
    videoElement.value.volume = volume.value
    isMuted.value = volume.value === 0
  }
}

function toggleMute() {
  if (isMuted.value) {
    volume.value = previousVolume.value
    isMuted.value = false
  } else {
    previousVolume.value = volume.value
    volume.value = 0
    isMuted.value = true
  }
  updateVolume()
}

function toggleAnnotations() {
  showAnnotations.value = !showAnnotations.value
}

function selectAnnotation(annotation: VideoAnnotation) {
  selectedAnnotationId.value = annotation.id
  emit('annotationSelected', annotation)
}

function closeAnnotation() {
  selectedAnnotationId.value = null
}

function getAnnotationIcon(type: string) {
  const iconMap: Record<string, any> = {
    goal: Target,
    foul: AlertCircle,
    offside: Flag,
    substitution: Users,
    card: CreditCard,
    other: AlertCircle
  }
  return iconMap[type] || AlertCircle
}

function getAnnotationPosition(annotation: VideoAnnotation) {
  if (!annotation.position) return { top: '50%', left: '50%' }
  
  return {
    top: `${annotation.position.y}%`,
    left: `${annotation.position.x}%`
  }
}

async function toggleFullscreen() {
  if (!document.fullscreenElement) {
    await playerContainer.value?.requestFullscreen()
    isFullscreen.value = true
  } else {
    await document.exitFullscreen()
    isFullscreen.value = false
  }
  emit('fullscreenChanged', isFullscreen.value)
}

function handleMouseMove() {
  showControls.value = true
  hideControlsAfterDelay()
}

function hideControls() {
  if (isPlaying.value) {
    showControls.value = false
  }
}

function hideControlsAfterDelay() {
  clearTimeout(controlsTimeout.value)
  if (isPlaying.value) {
    controlsTimeout.value = setTimeout(() => {
      showControls.value = false
    }, 3000)
  }
}

function handleProgressClick(event: MouseEvent) {
  if (!progressBar.value) return
  
  const rect = progressBar.value.getBoundingClientRect()
  const percent = (event.clientX - rect.left) / rect.width
  const time = percent * duration.value
  seekTo(time)
}

function handleProgressMouseDown(event: MouseEvent) {
  isDragging.value = true
  handleProgressDrag(event)
  
  const handleMouseMove = (e: MouseEvent) => handleProgressDrag(e)
  const handleMouseUp = () => {
    isDragging.value = false
    document.removeEventListener('mousemove', handleMouseMove)
    document.removeEventListener('mouseup', handleMouseUp)
  }
  
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
}

function handleProgressDrag(event: MouseEvent) {
  if (!progressBar.value || !isDragging.value) return
  
  const rect = progressBar.value.getBoundingClientRect()
  const percent = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
  const time = percent * duration.value
  seekTo(time)
}

function updateBuffered() {
  if (!videoElement.value) return
  
  const buffered = videoElement.value.buffered
  if (buffered.length > 0) {
    const bufferedEnd = buffered.end(buffered.length - 1)
    bufferedPercent.value = (bufferedEnd / duration.value) * 100
  }
}

function handleKeydown(event: KeyboardEvent) {
  // Prevent default for handled keys
  const handledKeys = [' ', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'f', 'F', 'a', 'A', ',', '.', '?']
  if (handledKeys.includes(event.key)) {
    event.preventDefault()
  }
  
  switch (event.key) {
    case ' ':
      togglePlayPause()
      break
    case 'ArrowLeft':
      seekTo(currentTime.value - 5)
      break
    case 'ArrowRight':
      seekTo(currentTime.value + 5)
      break
    case 'ArrowUp':
      volume.value = Math.min(1, volume.value + 0.1)
      updateVolume()
      break
    case 'ArrowDown':
      volume.value = Math.max(0, volume.value - 0.1)
      updateVolume()
      break
    case 'f':
    case 'F':
      toggleFullscreen()
      break
    case 'a':
    case 'A':
      toggleAnnotations()
      break
    case ',':
      previousFrame()
      break
    case '.':
      nextFrame()
      break
    case '?':
      showShortcuts.value = !showShortcuts.value
      break
  }
}

// Utility Functions
function formatTime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  
  if (h > 0) {
    return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }
  return `${m}:${s.toString().padStart(2, '0')}`
}

// Lifecycle
onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
  
  // Handle fullscreen change
  document.addEventListener('fullscreenchange', () => {
    isFullscreen.value = !!document.fullscreenElement
  })
  
  // Initialize video settings
  if (videoElement.value) {
    videoElement.value.volume = volume.value
    videoElement.value.playbackRate = playbackSpeed.value
  }
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
  clearTimeout(controlsTimeout.value)
})

// Watch for prop changes
watch(() => props.videoSrc, () => {
  currentTime.value = 0
  duration.value = 0
  isPlaying.value = false
})

watch(loop, (newLoop) => {
  if (videoElement.value) {
    videoElement.value.loop = newLoop
  }
})
</script>

<style scoped>
.video-player-container {
  @apply w-full space-y-4;
}

.video-player-wrapper {
  @apply relative bg-black rounded-xl overflow-hidden shadow-2xl;
}

.video-player-wrapper.fullscreen {
  @apply fixed inset-0 z-50 rounded-none;
}

.video-player {
  @apply relative w-full h-full cursor-pointer;
}

.video-element {
  @apply w-full h-full object-contain;
}

.annotations-overlay {
  @apply absolute inset-0 pointer-events-none;
}

.annotation {
  @apply absolute pointer-events-auto cursor-pointer transform -translate-x-1/2 -translate-y-1/2;
}

.annotation-marker {
  @apply w-8 h-8 rounded-full flex items-center justify-center shadow-lg transition-all duration-200;
}

.annotation-marker:hover {
  @apply scale-110;
}

.annotation--goal .annotation-marker {
  @apply bg-green-500 text-white;
}

.annotation--foul .annotation-marker {
  @apply bg-red-500 text-white;
}

.annotation--offside .annotation-marker {
  @apply bg-yellow-500 text-black;
}

.annotation--substitution .annotation-marker {
  @apply bg-blue-500 text-white;
}

.annotation--card .annotation-marker {
  @apply bg-orange-500 text-white;
}

.annotation--other .annotation-marker {
  @apply bg-purple-500 text-white;
}

.annotation--active .annotation-marker {
  @apply ring-4 ring-white ring-opacity-50 scale-125;
}

.annotation-tooltip {
  @apply absolute top-full mt-2 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-90 text-white rounded-lg p-3 min-w-[200px] max-w-[300px] z-10;
}

.tooltip-header {
  @apply flex items-center justify-between mb-2;
}

.tooltip-title {
  @apply font-semibold text-sm;
}

.tooltip-close {
  @apply text-gray-300;
}

.tooltip-close:hover {
  @apply text-white;
}

.tooltip-description {
  @apply text-xs text-gray-200 mb-2;
}

.tooltip-players {
  @apply text-xs;
}

.players-label {
  @apply text-gray-300;
}

.players-list {
  @apply text-white font-medium ml-1;
}

.loading-overlay {
  @apply absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center;
}

.loading-spinner {
  @apply text-white;
}

.controls-overlay {
  @apply absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent opacity-0 transition-opacity duration-300 pointer-events-none;
}

.controls-overlay--visible {
  @apply opacity-100 pointer-events-auto;
}

.main-controls {
  @apply absolute bottom-16 left-0 right-0 flex items-center space-x-4 px-6;
}

.control-button {
  @apply text-white p-2 rounded-md transition-colors;
}

.control-button:hover {
  @apply text-gray-300 bg-white bg-opacity-20;
}

.control-button--primary {
  @apply bg-white bg-opacity-20;
}

.control-button--primary:hover {
  @apply bg-opacity-30;
}

.control-button.active {
  @apply text-blue-400;
}

.playback-speed {
  @apply relative;
}

.speed-menu {
  @apply absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-90 rounded-lg py-2 min-w-[4rem];
}

.speed-option {
  @apply block w-full px-3 py-1 text-white text-sm text-center;
}

.speed-option:hover {
  @apply bg-white bg-opacity-20;
}

.speed-option.active {
  @apply text-blue-400;
}

.frame-controls {
  @apply flex items-center space-x-1;
}

.time-display {
  @apply text-white text-sm font-mono;
}

.time-separator {
  @apply text-gray-400 mx-1;
}

.controls-spacer {
  @apply flex-1;
}

.progress-container {
  @apply absolute bottom-6 left-0 right-0 px-6;
}

.timeline-events {
  @apply relative h-1 mb-2;
}

.timeline-event {
  @apply absolute top-0 w-1 h-full cursor-pointer;
}

.timeline-event--goal {
  @apply bg-green-500;
}

.timeline-event--foul {
  @apply bg-red-500;
}

.timeline-event--offside {
  @apply bg-yellow-500;
}

.timeline-event--substitution {
  @apply bg-blue-500;
}

.timeline-event--card {
  @apply bg-orange-500;
}

.timeline-event--other {
  @apply bg-purple-500;
}

.event-marker {
  @apply w-full h-full rounded-sm shadow-sm;
}

.progress-bar {
  @apply relative h-1 bg-gray-600 rounded-full cursor-pointer;
}

.progress-bg {
  @apply absolute inset-0 bg-gray-600 rounded-full;
}

.progress-buffered {
  @apply absolute top-0 left-0 h-full bg-gray-500 rounded-full;
}

.progress-played {
  @apply absolute top-0 left-0 h-full bg-blue-500 rounded-full;
}

.progress-scrubber {
  @apply absolute top-1/2 w-3 h-3 bg-blue-500 rounded-full transform -translate-y-1/2 -translate-x-1/2 opacity-0 transition-opacity;
}

.progress-bar:hover .progress-scrubber {
  @apply opacity-100;
}

.preview-tooltip {
  @apply absolute bottom-full mb-2 transform -translate-x-1/2 bg-black bg-opacity-90 text-white text-xs px-2 py-1 rounded;
}

.settings-panel {
  @apply absolute top-4 right-4 bg-black bg-opacity-90 text-white rounded-lg p-4 min-w-[250px];
}

.settings-header {
  @apply flex items-center justify-between mb-4;
}

.settings-header h3 {
  @apply font-semibold;
}

.settings-close {
  @apply text-gray-300;
}

.settings-close:hover {
  @apply text-white;
}

.settings-content {
  @apply space-y-4;
}

.setting-group {
  @apply space-y-2;
}

.setting-label {
  @apply block text-sm font-medium flex items-center space-x-2;
}

.setting-checkbox {
  @apply w-4 h-4;
}

.volume-control {
  @apply flex items-center space-x-2;
}

.volume-button {
  @apply text-white;
}

.volume-button:hover {
  @apply text-gray-300;
}

.volume-slider {
  @apply flex-1;
}

.volume-value {
  @apply text-xs font-mono min-w-[3rem] text-right;
}

.quality-select {
  @apply bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm;
}

.video-info {
  @apply bg-white dark:bg-gray-800 rounded-xl p-4 flex items-center justify-between border border-gray-200 dark:border-gray-700;
}

.video-meta {
  @apply space-y-2;
}

.video-title {
  @apply font-semibold text-gray-900 dark:text-white;
}

.video-details {
  @apply flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-300;
}

.detail-item {
  @apply flex items-center space-x-1;
}

.playback-stats {
  @apply space-y-1;
}

.stat-item {
  @apply flex items-center justify-between text-sm min-w-[6rem];
}

.stat-label {
  @apply text-gray-600 dark:text-gray-400;
}

.stat-value {
  @apply text-gray-900 dark:text-white font-medium;
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
  @apply text-gray-400;
}

.shortcuts-close:hover {
  @apply text-gray-600 dark:text-gray-200;
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
</style>