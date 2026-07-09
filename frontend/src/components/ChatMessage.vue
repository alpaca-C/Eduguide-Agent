<script setup>
import { renderMarkdown, escapeHtml } from '@/composables/useMarkdown'
import { computed } from 'vue'

const props = defineProps({
  role: { type: String, required: true },
  content: { type: String, required: true },
})

const html = computed(() =>
  props.role === 'assistant' ? renderMarkdown(props.content) : escapeHtml(props.content)
)
</script>

<template>
  <div class="message" :class="role">
    <div class="msg-content" v-html="html" />
  </div>
</template>
