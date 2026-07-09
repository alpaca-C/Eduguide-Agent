import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    redirect: '/qa',
  },
  {
    path: '/qa',
    name: 'qa',
    component: () => import('@/components/QAPage.vue'),
  },
  {
    path: '/docs',
    name: 'docs',
    component: () => import('@/components/DocsPage.vue'),
  },
]

export default createRouter({
  history: createWebHistory(),
  routes,
})
