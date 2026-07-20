<script setup>
import { ref, onMounted } from 'vue'
import { useDocsStore } from '@/stores/docs'
import { useAppStore } from '@/stores/app'
import { readSSE } from '@/composables/useSSE'
import { formatSize } from '@/composables/useMarkdown'
import ProgressBar from '@/components/ProgressBar.vue'

const docs = useDocsStore()
const app = useAppStore()

const abortController = ref(null)  // AbortController for in-flight detection

onMounted(() => docs.loadFiles())

const fileInput = ref(null)

// ── Upload ────────────────────────────────────────────────────────

function onUploadClick() { fileInput.value?.click() }

async function onFileChange(e) {
  if (e.target.files.length) await uploadFiles(e.target.files)
  e.target.value = ''
}

async function uploadFiles(fileList) {
  const formData = new FormData()
  for (const f of fileList) {
    const ext = f.name.split('.').pop().toLowerCase()
    if (!['pdf', 'txt', 'md', 'docx'].includes(ext)) {
      app.toast(`${f.name} 格式不支持`, 'error')
      continue
    }
    formData.append('files', f)
  }
  if (!formData.getAll('files').length) return

  app.showLoading('上传中...')
  try {
    const res = await fetch('/api/files/upload', { method: 'POST', body: formData })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || `上传失败 (HTTP ${res.status})`)
    }
    const data = await res.json()
    docs.files = data.uploaded || []
    app.toast(`已上传 ${data.total} 个文件`, 'success')
  } catch (e) {
    app.toast(`上传失败: ${e.message}`, 'error')
  }
  app.hideLoading()
}

function onDragOver(e) { e.preventDefault(); e.target.classList.add('drag-over') }
function onDragLeave(e) { e.target.classList.remove('drag-over') }
function onDrop(e) {
  e.preventDefault()
  e.target.classList.remove('drag-over')
  if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files)
}

// ── File selection ────────────────────────────────────────────────

async function selectDoc(filename) {
  if (docs.activeDoc === filename) return
  docs.activeDoc = filename
  await docs.loadChapters(filename)
  await docs.loadProcessedLabels()
}

async function deleteFile(filename) {
  if (!confirm(`确定删除文件 "${filename}" 吗？此操作不可恢复。`)) return
  await docs.deleteFile(filename)
  app.toast(`已删除: ${filename}`, 'info')
}

// ── Chapter detection (per-file state, concurrent-safe) ───────────

async function detectChapters() {
  const targetFile = docs.activeDoc
  if (!targetFile) return

  // Guard: don't start if this file is already detecting
  if (docs.detectingState[targetFile]?.detecting) return

  // Normal detection: does NOT clear existing indexed data.
  // Use '重新检测并清除' to wipe and re-detect.
  docs.chapters = []
  docs.startDetection(targetFile)

  // Create abort controller for this detection
  const controller = new AbortController()
  abortController.value = controller

  await new Promise(r => setTimeout(r, 50))

  try {
    const res = await fetch('/api/chapters/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filepaths: [targetFile] }),
      signal: controller.signal,
    })

    await readSSE(res, (evt) => {
      switch (evt.type) {
        case 'progress':
          docs.updateDetection(targetFile, {
            text: `${evt.file}: ${evt.stage} (${evt.file_idx}/${evt.file_total})`,
          })
          break
        case 'file_done':
          docs.updateDetection(targetFile, {
            percent: Math.round((evt.file_idx / evt.file_total) * 100),
            text: `${evt.file}: 完成 (${evt.chapters_found} 个章节)`,
          })
          break
        case 'error':
          app.toast(`${evt.file}: ${evt.msg}`, 'error')
          break
        case 'complete': {
          const chapters = (evt.chapters || []).map(c => ({
            title: c.title || c.label,
            label: c.label,
            filename: c.filename || targetFile,
            text_length: c.text_length || 0,
            selected: false,
          }))
          docs.docChaptersCache[targetFile] = { chapters }
          if (docs.activeDoc === targetFile) {
            docs.chapters = chapters
          }
          docs.finishDetection(targetFile, true)
          app.toast(`检测到 ${evt.total} 个章节`, 'success')
          break
        }
      }
    })
  } catch (e) {
    if (e.name === 'AbortError') {
      // User cancelled — no error toast, state already reset in cancelDetection()
    } else {
      app.toast(`章节检测失败: ${e.message}`, 'error')
      docs.finishDetection(targetFile, false)
    }
  } finally {
    abortController.value = null
  }

  // Persist to backend (non-blocking)
  const cached = docs.docChaptersCache[targetFile]
  if (cached?.chapters?.length) {
    fetch('/api/chapters/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: targetFile, chapters: cached.chapters }),
    }).catch(() => { /* non-critical */ })
  }

  if (docs.activeDoc === targetFile) {
    docs.loadProcessedLabels()
  }
}

async function detectChaptersWithClear() {
  const targetFile = docs.activeDoc
  if (!targetFile) return
  if (docs.detectingState[targetFile]?.detecting) return
  await docs.clearDocumentKnowledge(targetFile)
  detectChapters()
}

function cancelDetection() {
  if (abortController.value) {
    abortController.value.abort()
    abortController.value = null
  }
  // Reset detection state for current file
  const f = docs.activeDoc
  if (f) docs.finishDetection(f, false)
}

function toggleChapter(idx) {
  docs.chapters[idx].selected = !docs.chapters[idx].selected
}

// ── Knowledge processing ──────────────────────────────────────────

async function processDocuments() {
  if (!docs.selectedChapters.length || !docs.activeDoc) return

  app.showLoading('正在处理...')
  let finalData = null

  try {
    const res = await fetch('/api/knowledge/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filepaths: [docs.activeDoc],
        selected_chapters: docs.selectedChapters,
      }),
    })

    await readSSE(res, (evt) => {
      if (evt.type === 'progress') app.loadingText = evt.stage
      else if (evt.type === 'error') app.toast(`处理失败: ${evt.msg}`, 'error')
      else if (evt.type === 'complete') finalData = evt
    })

    if (finalData) {
      docs.processedLabels[docs.activeDoc] = [...docs.selectedChapters]
      app.toast(
        `处理完成！概念: ${finalData.stats.concepts}, 关系: ${finalData.stats.relations}`,
        'success',
      )
    }
  } catch (e) {
    app.toast(`处理失败: ${e.message}`, 'error')
  }
  app.hideLoading()
}
</script>

<template>
  <div class="docs-layout">
    <!-- Sidebar -->
    <aside class="docs-sidebar">
      <h3>已上传资料</h3>
      <label
        class="upload-area"
        @click="onUploadClick"
        @dragover.prevent="onDragOver"
        @dragleave="onDragLeave"
        @drop="onDrop"
      >
        <input ref="fileInput" type="file" multiple accept=".pdf,.txt,.md,.docx" hidden @change="onFileChange" />
        <span class="upload-icon">+</span>
        <span>点击或拖拽上传文件</span>
        <span class="upload-hint">PDF / TXT / MD / DOCX</span>
      </label>

      <div class="doc-list">
        <button
          v-for="f in docs.files" :key="f"
          class="doc-item"
          :class="{ active: docs.activeDoc === f }"
          @click="selectDoc(f)"
        >
          <span class="doc-icon">📄</span>
          <span class="doc-name">{{ f }}</span>
          <!-- Show spinner if this file is being detected -->
          <span v-if="docs.detectingState[f]?.detecting" class="doc-detect-spinner" title="检测中..." />
          <span class="doc-delete" @click.stop="deleteFile(f)">×</span>
        </button>
      </div>
    </aside>

    <!-- Main area -->
    <div class="docs-main">
      <div v-if="!docs.activeDoc" class="docs-empty">
        <div class="empty-icon">📄</div>
        <p>请先上传资料，然后选择一个资料查看章节</p>
      </div>

      <div v-else class="docs-chapters">
        <div class="docs-chapters-header">
          <h3>章节管理: {{ docs.activeDoc }}</h3>
          <div class="docs-chapters-actions">
            <button
              class="btn-sm"
              :disabled="docs.isDetecting"
              @click="detectChapters"
            >
              {{ docs.isDetecting ? '检测中...' : docs.hasChapterCache ? '重新分析章节' : '检测章节' }}
            </button>
            <button
              class="btn-sm btn-danger"
              :disabled="docs.isDetecting"
              @click="detectChaptersWithClear"
              title="清除本书已索引数据后重新检测"
            >
              🗑 清除并重检
            </button>
            <button
              v-if="docs.isDetecting"
              class="btn-sm btn-cancel"
              @click="cancelDetection"
            >
              取消检测
            </button>
            <button class="btn-sm btn-outline" @click="docs.selectAll()">全选</button>
            <button class="btn-sm btn-outline" @click="docs.deselectAll()">取消全选</button>
          </div>
        </div>

        <ProgressBar
          :visible="docs.isDetecting"
          :percent="docs.currentProgress.percent"
          :text="docs.currentProgress.text"
          :animating="docs.currentProgress.animating"
        />

        <div class="chapter-list">
          <div
            v-for="(ch, i) in docs.chapters" :key="ch.label"
            class="chapter-item"
            :class="{ selected: ch.selected }"
            @click="toggleChapter(i)"
          >
            <div class="checkbox" />
            <span class="ch-name">{{ ch.title }}</span>
            <span
              class="ch-badge"
              :class="{ imported: docs.processedLabels[docs.activeDoc]?.includes(ch.label) }"
              @click.stop
            >
              <template v-if="docs.processedLabels[docs.activeDoc]?.includes(ch.label)">
                已导入
                <button
                  class="ch-unimport"
                  title="删除此章节的索引数据"
                  @click.stop="docs.unimportChapter(ch.label).then(() => app.toast('已删除索引: ' + ch.title, 'info')).catch(() => app.toast('删除失败', 'error'))"
                >×</button>
              </template>
              <template v-else>未导入</template>
            </span>
            <span class="ch-len">{{ formatSize(ch.text_length || 0) }}</span>
          </div>
        </div>

        <div class="docs-chapters-footer">
          <button
            class="btn-primary"
            :disabled="!docs.selectedCount"
            @click="processDocuments"
          >
            {{ docs.selectedCount ? `开始处理 (${docs.selectedCount} 个章节)` : '开始处理选中章节' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
