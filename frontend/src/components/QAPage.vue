<script setup>
import { ref, nextTick, onMounted, watch } from 'vue'
import { useChatStore } from '@/stores/chat'
import { useAppStore } from '@/stores/app'
import { useDocsStore } from '@/stores/docs'
import { post } from '@/composables/useAPI'
import { escapeHtml as escapeHtmlFn } from '@/composables/useMarkdown'
import { readSSE } from '@/composables/useSSE'
import { renderMarkdown } from '@/composables/useMarkdown'
import ChatMessage from '@/components/ChatMessage.vue'

const chat = useChatStore()
const app = useAppStore()
const docs = useDocsStore()

const messages = ref([])
const input = ref('')
const thinking = ref('')
const sending = ref(false)
const chatContainer = ref(null)
const docFilter = ref([])

onMounted(() => {
  chat.refreshSessions()
})

// Load messages when session changes
watch(() => chat.sessionId, async () => {
  if (chat.sessionId && chat.chatHistory.length) {
    messages.value = chat.chatHistory.map(m => ({ role: m.role, content: m.content }))
  }
})

function scrollDown() {
  nextTick(() => {
    if (chatContainer.value) {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    }
  })
}

async function sendMessage() {
  const question = input.value.trim()
  if (!question || sending.value) return
  input.value = ''
  sending.value = true

  messages.value.push({ role: 'user', content: question })
  thinking.value = '思考中...'
  scrollDown()

  let replyText = ''
  let replyStarted = false
  let finalSid = ''

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        session_id: chat.sessionId,
        doc_filter: docFilter.value,
      }),
    })

    await readSSE(res, (evt) => {
      if (evt.type === 'status') {
        thinking.value = evt.text
      } else if (evt.type === 'reply_start') {
        thinking.value = ''
        replyStarted = true
        messages.value.push({ role: 'assistant', content: '' })
      } else if (evt.type === 'reply_chunk') {
        replyText += evt.text
        // Update the last message (streaming preview — escape HTML)
        const last = messages.value[messages.value.length - 1]
        if (last) last.content = replyText
        scrollDown()
      } else if (evt.type === 'done') {
        finalSid = evt.session_id
        // Final render: swap plain text for rendered markdown
        const last = messages.value[messages.value.length - 1]
        if (last) last.content = replyText
      }
    })

    if (finalSid) chat.sessionId = finalSid
    if (!replyStarted) {
      thinking.value = ''
      messages.value.push({ role: 'assistant', content: '抱歉，无法生成回答。' })
    }

    chat.chatHistory.push(
      { role: 'user', content: question },
      { role: 'assistant', content: replyText || '...' },
    )
  } catch (e) {
    thinking.value = ''
    messages.value.push({ role: 'assistant', content: `错误: ${e.message}` })
  }
  sending.value = false
  scrollDown()
}

function newConversation() {
  chat.newConversation()
  messages.value = []
}

async function loadSession(sid) {
  const msgs = await chat.loadSession(sid)
  messages.value = (msgs || []).map(m => ({ role: m.role, content: m.content }))
}

function onKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}
</script>

<template>
  <div class="qa-layout">
    <!-- Sessions sidebar -->
    <aside class="qa-sessions">
      <div class="qa-sessions-header">
        <h3>对话记录</h3>
        <button class="btn-sm" @click="newConversation">+ 新对话</button>
      </div>
      <div class="session-list">
        <div v-if="!chat.sessions.length" class="session-empty">暂无对话记录</div>
        <div
          v-for="s in chat.sessions" :key="s.session_id"
          class="session-item"
          @click="loadSession(s.session_id)"
        >
          <span class="sess-topic">{{ s.topic || '新对话' }}</span>
          <button class="sess-delete" @click.stop="chat.deleteSession(s.session_id)">×</button>
        </div>
      </div>
    </aside>

    <!-- Chat area -->
    <div class="qa-chat">
      <div class="chat-toolbar">
        <span class="current-session">
          {{ chat.sessionId ? `会话: ${chat.sessionId}` : '新会话' }}
        </span>
      </div>

      <div ref="chatContainer" class="chat-container">
        <div v-if="!messages.length" class="chat-welcome">
          <div class="welcome-icon">📚</div>
          <h3>文档知识图谱问答系统</h3>
          <p>上传资料后，基于资料内容进行智能问答</p>
        </div>

        <template v-for="(msg, i) in messages" :key="i">
          <ChatMessage :role="msg.role" :content="msg.content" />
        </template>

        <!-- Thinking indicator -->
        <div v-if="thinking" class="message assistant thinking">
          <div class="msg-content">
            <span class="thinking-text">{{ thinking }}</span>
          </div>
        </div>
      </div>

      <div class="chat-input-area">
        <textarea
          v-model="input"
          placeholder="基于上传的资料提问..."
          rows="2"
          @keydown="onKeydown"
        />
        <button class="btn-primary" @click="sendMessage" :disabled="sending">发送</button>
      </div>
    </div>
  </div>
</template>
