import { createRouter, createWebHistory } from 'vue-router'
import Home from '@/views/Home.vue'
import Upload from '@/views/Upload.vue'

const router = createRouter({
    history: createWebHistory(),
    routes: [
        {
            path: '/',
            name: 'Home',
            component: Home,
        },
        {
            path: '/upload',
            name: 'Upload',
            component: Upload,
        },
        {
            path: '/analysis/:videoId?',
            name: 'Analysis',
            component: () => import('@/views/Analysis.vue'),
        },
        {
            path: '/results/:videoId?',
            name: 'Results',
            component: () => import('@/views/Results.vue'),
        },
        {
            path: '/reports/:videoId?',
            name: 'Reports',
            component: () => import('@/views/Reports.vue'),
        },
    ],
});

export default router;
