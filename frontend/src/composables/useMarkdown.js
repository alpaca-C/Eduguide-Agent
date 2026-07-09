import { marked } from 'marked'
import katex from 'katex'
import 'katex/dist/katex.min.css'

// Configure marked
marked.setOptions({
  breaks: true,
  gfm: true,
  headerIds: false,
  mangle: false,
})

const SENTINEL = '\x00MATH'

export function renderMarkdown(text) {
  if (!text) return ''

  // Phase 1: Protect LaTeX
  const mathBlocks = []
  let idx = 0

  // Display math: \[ ... \] and $$ ... $$
  for (const [open, close] of [['\\[', '\\]'], ['$$', '$$']]) {
    let i = 0
    while (i < text.length) {
      const ds = text.indexOf(open, i)
      if (ds === -1) break
      const de = text.indexOf(close, ds + open.length)
      if (de === -1) break
      const formula = text.substring(ds + open.length, de)
      const id = `${SENTINEL}${idx++}${SENTINEL}`
      mathBlocks.push({ id, formula, display: true })
      text = text.substring(0, ds) + id + text.substring(de + close.length)
      i = ds + id.length
    }
  }

  // Inline math: \( ... \)
  let j = 0
  while (j < text.length) {
    const is = text.indexOf('\\(', j)
    if (is === -1) break
    const ie = text.indexOf('\\)', is + 2)
    if (ie === -1) break
    const formula = text.substring(is + 2, ie)
    const id = `${SENTINEL}${idx++}${SENTINEL}`
    mathBlocks.push({ id, formula, display: false })
    text = text.substring(0, is) + id + text.substring(ie + 2)
    j = is + id.length
  }

  // Phase 2: Markdown
  let html
  try {
    html = marked.parse(text)
  } catch (e) {
    console.warn('Markdown parse failed:', e)
    html = escapeHtml(text)
  }

  // Phase 3: Render KaTeX
  for (const block of mathBlocks) {
    let rendered
    try {
      rendered = katex.renderToString(block.formula, {
        displayMode: block.display,
        throwOnError: false,
        trust: false,
      })
    } catch {
      rendered = block.display
        ? `<span class="math-fallback">\\[${escapeText(block.formula)}\\]</span>`
        : `<span class="math-fallback">\\(${escapeText(block.formula)}\\)</span>`
    }
    html = html.replace(block.id, rendered)
  }

  return html
}

export function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>')
}

function escapeText(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

export function formatSize(bytes) {
  if (!bytes || bytes < 1000) return (bytes || 0) + '字'
  if (bytes < 1000000) return (bytes / 1000).toFixed(1) + '千字'
  return (bytes / 1000000).toFixed(1) + '万字'
}
