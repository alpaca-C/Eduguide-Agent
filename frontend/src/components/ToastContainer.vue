<script setup>
import { useAppStore } from '@/stores/app'
const app = useAppStore()
</script>

<template>
  <div class="toast-container">
    <div
      v-for="t in app.toasts" :key="t.id"
      class="toast" :class="t.type"
    >{{ t.msg }}</div>
  </div>
</template>
