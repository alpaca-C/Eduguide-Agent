<script setup>
import { useAppStore } from '@/stores/app'
const app = useAppStore()
</script>

<template>
  <div v-if="app.loading" class="loading-overlay">
    <div class="spinner" />
    <div class="loading-text">{{ app.loadingText }}</div>
  </div>
</template>
