<script setup>
import { useAppStore } from '@/stores/app'
const app = useAppStore()
</script>

<template>
  <div v-if="app.showSettings" class="settings-overlay" @click.self="app.showSettings = false">
    <div class="settings-modal">
      <div class="settings-header">
        <h3>设置</h3>
        <button class="settings-close" @click="app.showSettings = false">×</button>
      </div>
      <div class="settings-body">
        <div class="settings-row">
          <span>主题</span>
          <div class="theme-toggle">
            <button
              class="theme-option" :class="{ active: app.theme === 'dark' }"
              @click="app.setTheme('dark'); app.applyTheme()"
            >☾ 深色</button>
            <button
              class="theme-option" :class="{ active: app.theme === 'light' }"
              @click="app.setTheme('light'); app.applyTheme()"
            >☀ 浅色</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
