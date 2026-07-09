export async function readSSE(response, onEvent) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const result = await reader.read()
    if (result.done) break

    buffer += decoder.decode(result.value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed.startsWith('data: ')) continue
      try {
        const evt = JSON.parse(trimmed.slice(6))
        onEvent(evt)
      } catch {
        // Skip malformed SSE events
      }
    }
  }
}
