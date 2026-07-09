import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useAppStore = defineStore('app', () => {
  // Theme
  const theme = ref(localStorage.getItem('theme') || 'dark')
  function setTheme(t) { theme.value = t; localStorage.setItem('theme', t) }
  function applyTheme() { document.documentElement.setAttribute('data-theme', theme.value) }

  // Toast
  const toasts = ref([])
  let toastId = 0
  function toast(msg, type = 'info') {
    const id = ++toastId
    toasts.value.push({ id, msg, type })
    setTimeout(() => {
      toasts.value = toasts.value.filter(t => t.id !== id)
    }, 3000)
  }

  // Loading overlay
  const loading = ref(false)
  const loadingText = ref('处理中...')
  function showLoading(text = '处理中...') { loadingText.value = text; loading.value = true }
  function hideLoading() { loading.value = false }

  // Settings modal
  const showSettings = ref(false)

  return { theme, setTheme, applyTheme, toasts, toast, loading, loadingText, showLoading, hideLoading, showSettings }
})
