// Document QA System — Centralized State
// Single source of truth for all UI state. Components read/write through
// this store rather than scattered global variables.
var State = {
    // Document state
    files: [],
    activeDoc: null,
    chapters: [],
    selectedChapters: [],
    docChaptersCache: {},
    processedLabels: {},

    // Chat state
    sessionId: '',
    chatHistory: [],
    processing: false,
};
