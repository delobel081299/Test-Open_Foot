import { createApp } from 'vue'
import App from './App.vue'
import router from './router/router'
import { createPinia } from 'pinia'
import '@/assets/main.css'
import * as lucide from 'lucide-vue-next'

// Chart.js registration
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  RadialLinearScale,
  ArcElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  RadialLinearScale,
  ArcElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

const app = createApp(App)

// Register Lucide icons globally
for (const [name, component] of Object.entries(lucide)) {
  app.component(name, component)
}

app.use(createPinia())
app.use(router)
app.mount('#app')