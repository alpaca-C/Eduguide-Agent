import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { get, del } from '@/composables/useAPI'

export const useDocsStore = defineStore('docs', () => {
  const files = ref([])
  const activeDoc = ref(null)
  const chapters = ref([])
  const docChaptersCache = ref({})
  const processedLabels = ref({})

  // ── Per-file detection state ─────────────────────────────────────
  // Keyed by filename. Each file independently tracks its own
  // detection status — switching files shows that file's state.

  const detectingState = ref(/** @type {Record<string, {detecting:boolean,percent:number,text:string,animating:boolean,hasCache:boolean}>} */ ({}))

  function _ensureState(filename) {
    if (!detectingState.value[filename]) {
      detectingState.value[filename] = {
        detecting: false, percent: 0, text: '', animating: false, hasCache: false,
      }
    }
    return detectingState.value[filename]
  }

  function startDetection(filename) {
    const s = _ensureState(filename)
    s.detecting = true
    s.percent = 0
    s.text = '开始检测...'
    s.animating = true
  }

  function updateDetection(filename, patch) {
    const s = _ensureState(filename)
    Object.assign(s, patch)
  }

  function finishDetection(filename, hasCache) {
    const s = _ensureState(filename)
    s.detecting = false
    s.hasCache = hasCache
    s.animating = false
    s.percent = hasCache ? 100 : 0
  }

  // Computed for the currently active document
  const isDetecting = computed(() => {
    const s = detectingState.value[activeDoc.value]
    return s ? s.detecting : false
  })

  const currentProgress = computed(() => {
    const s = detectingState.value[activeDoc.value]
    return s || { percent: 0, text: '', animating: false }
  })

  const hasChapterCache = computed(() => {
    // Check per-file state first, then fall back to in-memory cache
    const s = detectingState.value[activeDoc.value]
    if (s?.hasCache) return true
    return !!docChaptersCache.value[activeDoc.value]?.chapters?.length
  })

  // ── Chapter selection ───────────────────────────────────────────

  const selectedChapters = computed(() =>
    chapters.value.filter(c => c.selected).map(c => c.label)
  )
  const selectedCount = computed(() =>
    chapters.value.filter(c => c.selected).length
  )

  // ── Data loading ────────────────────────────────────────────────

  async function loadFiles() {
    try {
      const data = await get('/files/list')
      files.value = data.files || []
      if (files.value.length === 0) {
        try { await fetch('/api/knowledge/clear', { method: 'DELETE' }) } catch { /* ok */ }
      }
    } catch (e) { console.error('Failed to load files', e) }
  }

  async function deleteFile(filename) {
    await del(`/files/${encodeURIComponent(filename)}`)
    files.value = files.value.filter(f => f !== filename)
    delete detectingState.value[filename]
    if (activeDoc.value === filename) {
      activeDoc.value = null
      chapters.value = []
    }
  }

  async function loadChapters(filename) {
    chapters.value = []
    // 1) In-memory cache first (instant for previously detected files)
    const cached = docChaptersCache.value[filename]
    if (cached?.chapters?.length) {
      chapters.value = cached.chapters.map(c => ({
        title: c.title || c.label,
        label: c.label,
        filename: c.filename || filename,
        text_length: c.text_length || 0,
        imported: c.imported || false,
        selected: false,
      }))
      const s = _ensureState(filename)
      s.hasCache = true
      return true
    }
    // 2) Backend (persisted from previous sessions)
    try {
      const data = await get(`/chapters/${encodeURIComponent(filename)}`)
      if (data.chapters?.length) {
        chapters.value = data.chapters.map(c => ({
          title: c.title || c.label,
          label: c.label,
          filename: c.filename || filename,
          text_length: c.text_length || 0,
          imported: c.imported || false,
          selected: false,
        }))
        docChaptersCache.value[filename] = { chapters: data.chapters }
        const s = _ensureState(filename)
        s.hasCache = true
        return true
      }
    } catch { /* no cache */ }
    return false
  }

  async function unimportChapter(label) {
    try {
      await fetch(`/api/knowledge/chapters/${encodeURIComponent(label)}`, { method: 'DELETE' })
      // Remove from processed labels
      for (const [fn, labels] of Object.entries(processedLabels.value)) {
        processedLabels.value[fn] = labels.filter(l => l !== label)
      }
    } catch (e) {
      console.error('Failed to unimport chapter', label, e)
      throw e
    }
  }

  function selectAll() { chapters.value.forEach(c => c.selected = true) }
  function deselectAll() { chapters.value.forEach(c => c.selected = false) }

  // ── Processed labels ────────────────────────────────────────────

  async function loadProcessedLabels(filename) {
    const fn = filename || activeDoc.value
    if (!fn) return
    try {
      // Always query backend directly for fresh imported status
      const data = await get(`/chapters/${encodeURIComponent(fn)}`)
      const importedLabels = (data.chapters || [])
        .filter(c => c.imported)
        .map(c => c.label)
      processedLabels.value[fn] = importedLabels
    } catch { /* keep existing */ }
  }

  return {
    files, activeDoc, chapters, docChaptersCache, processedLabels,
    detectingState,
    isDetecting, currentProgress, hasChapterCache,
    selectedChapters, selectedCount,
    startDetection, updateDetection, finishDetection,
    loadFiles, deleteFile, loadChapters, unimportChapter, selectAll, deselectAll, loadProcessedLabels,
  }
})
