import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import { get, post, del } from '@/composables/useAPI'

const SESSION_KEY = 'chat_session_id'
const USER_KEY = 'chat_user_id'

function loadFromStorage(key) {
  try { return localStorage.getItem(key) || '' } catch { return '' }
}
function saveToStorage(key, val) {
  try { localStorage.setItem(key, val) } catch { /* ignore */ }
}
function generateUserId() {
  return 'u_' + Math.random().toString(36).slice(2, 10)
}

export const useChatStore = defineStore('chat', () => {
  const sessionId = ref(loadFromStorage(SESSION_KEY))
  // Stable user identifier for cross-session episodic memory
  const userId = ref(loadFromStorage(USER_KEY) || (() => {
    const id = generateUserId(); saveToStorage(USER_KEY, id); return id
  })())
  const chatHistory = ref([])
  const sessions = ref([])

  // Persist session_id to localStorage on change (survives page refresh)
  watch(sessionId, (sid) => { if (sid) saveToStorage(SESSION_KEY, sid) })

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
    localStorage.removeItem(LS_KEY)
  }

  return { sessionId, chatHistory, sessions, refreshSessions, loadSession, deleteSession, newConversation }
})
