<script setup>
defineProps({
  visible: { type: Boolean, default: false },
  percent: { type: Number, default: 0 },
  text: { type: String, default: '' },
  animating: { type: Boolean, default: false },
})
</script>

<template>
  <div v-if="visible" class="chapter-progress" :class="{ animating }">
    <div class="progress-bar">
      <div class="progress-fill" :style="{ width: percent + '%' }" />
    </div>
    <div class="progress-text">{{ text }}</div>
  </div>
</template>
