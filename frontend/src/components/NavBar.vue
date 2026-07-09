<script setup>
import { useRoute } from 'vue-router'
import { computed } from 'vue'
import { useAppStore } from '@/stores/app'

const route = useRoute()
const app = useAppStore()
const active = computed(() => route.name)

const items = [
  { id: 'qa', icon: '💬', label: '知识问答' },
  { id: 'docs', icon: '📁', label: '资料管理' },
]
</script>

<template>
  <nav class="nav-bar">
    <router-link
      v-for="item in items" :key="item.id"
      :to="{ name: item.id }"
      class="nav-item"
      :class="{ active: active === item.id }"
    >
      <span class="nav-icon">{{ item.icon }}</span>
      <span class="nav-label">{{ item.label }}</span>
    </router-link>
    <div class="nav-spacer" />
    <button class="nav-item" @click="app.showSettings = !app.showSettings">
      <span class="nav-icon">⚙</span>
      <span class="nav-label">设置</span>
    </button>
  </nav>
</template>
