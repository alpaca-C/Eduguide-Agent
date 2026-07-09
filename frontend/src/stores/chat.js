import { defineStore } from 'pinia'
import { ref } from 'vue'
import { get, post, del } from '@/composables/useAPI'

export const useChatStore = defineStore('chat', () => {
  const sessionId = ref('')
  const chatHistory = ref([])
  const sessions = ref([])

  async function refreshSessions() {
    try {
      const data = await get('/sessions')
      sessions.value = data.sessions || []
    } catch { /* ignore */ }
  }

  async function loadSession(sid) {
    try {
      const data = await get(`/sessions/${sid}`)
      sessionId.value = sid
      chatHistory.value = data.messages || []
      return chatHistory.value
    } catch { return [] }
  }

  async function deleteSession(sid) {
    await del(`/sessions/${sid}`)
    if (sessionId.value === sid) newConversation()
    await refreshSessions()
  }

  function newConversation() {
    sessionId.value = ''
    chatHistory.value = []
  }

  return { sessionId, chatHistory, sessions, refreshSessions, loadSession, deleteSession, newConversation }
})
