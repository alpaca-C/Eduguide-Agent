<script setup>
import { onMounted } from 'vue'
import { useAppStore } from '@/stores/app'
import NavBar from '@/components/NavBar.vue'
import ToastContainer from '@/components/ToastContainer.vue'
import LoadingOverlay from '@/components/LoadingOverlay.vue'
import SettingsModal from '@/components/SettingsModal.vue'

const app = useAppStore()
onMounted(() => app.applyTheme())
</script>

<template>
  <div class="app">
    <NavBar />
    <main class="main-content">
      <router-view />
    </main>
    <ToastContainer />
    <LoadingOverlay />
    <SettingsModal />
  </div>
</template>
