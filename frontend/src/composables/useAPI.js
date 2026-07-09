import { ref } from 'vue'

// Base path for API requests — configurable via env
const BASE = '/api'

// Shared loading state
export const processing = ref(false)

export async function get(path) {
  return request(path, { method: 'GET' })
}

export async function post(path, body) {
  return request(path, {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export async function del(path) {
  return request(path, { method: 'DELETE' })
}

async function request(path, options = {}) {
  const url = path.startsWith('http') ? path : BASE + path
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  })
  if (!res.ok) {
    let err
    try { err = await res.json() } catch { err = { detail: res.statusText } }
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}
