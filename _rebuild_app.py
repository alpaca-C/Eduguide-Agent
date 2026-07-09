import os

content = """// Document QA System - Frontend App
const API = '/api';

// ===== State =====
const state = {
    files: [],
    chapters: [],
    selectedChapters: [],
    sessionId: '',
    chatHistory: [],
    processing: false,
};

// ===== DOM Elements =====
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
    uploadArea: $('#uploadArea'),
    fileInput: $('#fileInput'),
    fileList: $('#fileList'),
    chapterSection: $('#chapterSection'),
    detectBtn: $('#detectChaptersBtn'),
    chapterList: $('#chapterList'),
    selectAllBtn: $('#selectAllBtn'),
    deselectAllBtn: $('#deselectAllBtn'),
    processBtn: $('#processBtn'),
    statsSection: $('#statsSection'),
    statsBox: $('#statsBox'),
    sessionList: $('#sessionList'),
    refreshSessionsBtn: $('#refreshSessionsBtn'),
    docFilter: $('#docFilter'),
    refreshFilterBtn: $('#refreshFilterBtn'),
    chatContainer: $('#chatContainer'),
    chatInput: $('#chatInput'),
    sendBtn: $('#sendBtn'),
    statusLog: $('#statusLog'),
    conceptsPanel: $('#conceptsPanel'),
    loadingOverlay: $('#loadingOverlay'),
    loadingText: $('#loadingText'),
    toastContainer: $('#toastContainer'),
};

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
    initUpload();
    initTabs();
    initChat();
    initChapters();
    initSessions();
    initResize();
    loadFiles();
    refreshSessions();
    refreshDocFilter();
});

// ===== Toast =====
function toast(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = 'toast ' + type;
    el.textContent = msg;
    els.toastContainer.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; el.style.transition = '0.3s'; setTimeout(() => el.remove(), 300); }, 3000);
}

// ===== Loading =====
function showLoading(text) {
    els.loadingText.textContent = text || '\u5904\u7406\u4e2d...';
    els.loadingOverlay.style.display = 'flex';
    state.processing = true;
}
function hideLoading() {
    els.loadingOverlay.style.display = 'none';
    state.processing = false;
}

// ===== Load Existing Files =====
async function loadFiles() {
    try {
        const data = await api('/files/list');
        state.files = data.files || [];
        renderFileList();
        renderChapterSection();
    } catch (err) {
        console.error('Failed to load files', err);
    }
}

// ===== API Helpers =====
async function api(path, options = {}) {
    const url = path.startsWith('http') ? path : API + path;
    const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Request failed');
    }
    return res.json();
}

// ===== Upload =====
function initUpload() {
    els.uploadArea.addEventListener('click', () => els.fileInput.click());
    els.fileInput.addEventListener('change', handleFileSelect);

    els.uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); els.uploadArea.classList.add('drag-over'); });
    els.uploadArea.addEventListener('dragleave', () => els.uploadArea.classList.remove('drag-over'));
    els.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        els.uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
    });
}

async function handleFileSelect(e) {
    if (e.target.files.length) await uploadFiles(e.target.files);
    els.fileInput.value = '';
}

async function uploadFiles(fileList) {
    const formData = new FormData();
    for (const f of fileList) {
        const ext = f.name.split('.').pop().toLowerCase();
        if (!['pdf', 'txt', 'md', 'docx'].includes(ext)) {
            toast(f.name + ' \u683c\u5f0f\u4e0d\u652f\u6301', 'error');
            continue;
        }
        formData.append('files', f);
    }

    const entries = formData.getAll('files');
    if (!entries.length) return;

    showLoading('\u4e0a\u4f20\u4e2d...');
    try {
        const res = await fetch(API + '/files/upload', { method: 'POST', body: formData });
        const data = await res.json();
        state.files = data.uploaded;
        renderFileList();
        renderChapterSection();
        toast('\u5df2\u4e0a\u4f20' + data.total + ' \u4e2a\u6587\u4ef6', 'success');
    } catch (err) {
        toast('\u4e0a\u4f20\u5931\u8d25: ' + err.message, 'error');
    }
    hideLoading();
}

function renderFileList() {
    els.fileList.innerHTML = state.files.map((f, i) =>
        '<div class="file-item">' +
        '<span class="file-icon">\ud83d\udcc4</span>' +
        '<span class="file-name">' + f + '</span>' +
        '<span class="file-remove" data-idx="' + i + '">\u00d7</span>' +
        '</div>'
    ).join('');

    $$('.file-remove').forEach(el => {
        el.addEventListener('click', async (e) => {
            e.stopPropagation();
            const idx = parseInt(el.dataset.idx);
            const filename = state.files[idx];
            try {
                await fetch(API + '/files/' + encodeURIComponent(filename), { method: 'DELETE' });
                toast('\u5df2\u5220\u9664: ' + filename, 'info');
            } catch (err) {
                toast('\u5220\u9664\u5931\u8d25: ' + err.message, 'error');
                return;
            }
            state.files.splice(idx, 1);
            renderFileList();
            renderChapterSection();
        });
    });
}"""

print('Part 1 written, size:', len(content))

filepath = r'D:\NOTHING\self-coding\research-agent\frontend\js\app.js'
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('DONE part 1')
