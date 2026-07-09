// Document QA System — Document Management Component
var Docs = {
    init: function() {
        var detectBtn = document.getElementById('detectChaptersBtn');
        var selectAllBtn = document.getElementById('selectAllBtn');
        var deselectAllBtn = document.getElementById('deselectAllBtn');
        var processBtn = document.getElementById('processBtn');
        var uploadArea = document.getElementById('uploadArea');
        var fileInput = document.getElementById('fileInput');

        if (detectBtn) detectBtn.addEventListener('click', this._detectChapters.bind(this));
        if (selectAllBtn) selectAllBtn.addEventListener('click', this._selectAll.bind(this));
        if (deselectAllBtn) deselectAllBtn.addEventListener('click', this._deselectAll.bind(this));
        if (processBtn) processBtn.addEventListener('click', this._processDocuments.bind(this));

        // Upload handlers
        if (uploadArea && fileInput) {
            var self = this;
            uploadArea.addEventListener('click', function() { fileInput.click(); });
            fileInput.addEventListener('change', function(e) { self._handleFileSelect(e); });
            uploadArea.addEventListener('dragover', function(e) { e.preventDefault(); uploadArea.classList.add('drag-over'); });
            uploadArea.addEventListener('dragleave', function() { uploadArea.classList.remove('drag-over'); });
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('drag-over');
                if (e.dataTransfer.files.length) self._uploadFiles(e.dataTransfer.files);
            });
        }
    },

    // ── Upload ──────────────────────────────────────────────────────

    _handleFileSelect: async function(e) {
        if (e.target.files.length) await this._uploadFiles(e.target.files);
        e.target.value = '';
    },

    _uploadFiles: async function(fileList) {
        var formData = new FormData();
        for (var i = 0; i < fileList.length; i++) {
            var f = fileList[i];
            var ext = f.name.split('.').pop().toLowerCase();
            if (['pdf', 'txt', 'md', 'docx'].indexOf(ext) === -1) {
                Toast.show(f.name + ' 格式不支持', 'error');
                continue;
            }
            formData.append('files', f);
        }

        var entries = formData.getAll('files');
        if (!entries.length) return;

        Toast.loading.show('上传中...');
        try {
            var res = await fetch(API.BASE + '/files/upload', { method: 'POST', body: formData });
            if (!res.ok) {
                var errData = await res.json().catch(function() { return {}; });
                throw new Error(errData.detail || '上传失败 (HTTP ' + res.status + ')');
            }
            var data = await res.json();
            State.files = data.uploaded || [];
            this._renderDocList();
            Toast.show('已上传 ' + data.total + ' 个文件', 'success');
        } catch(err) {
            Toast.show('上传失败: ' + err.message, 'error');
        }
        Toast.loading.hide();
    },

    // ── File List ──────────────────────────────────────────────────

    loadFiles: async function() {
        try {
            var data = await API.get('/files/list');
            State.files = data.files || [];
            if (State.files.length === 0) {
                try { await fetch(API.BASE + '/knowledge/clear', { method: 'DELETE' }); } catch(e) {}
            }
            this._renderDocList();
        } catch(err) {
            console.error('Failed to load files', err);
        }
    },

    _renderDocList: function() {
        var list = document.getElementById('docList');
        if (!list) return;
        var self = this;

        list.innerHTML = State.files.map(function(f) {
            var isActive = State.activeDoc === f;
            return '<button class="doc-item' + (isActive ? ' active' : '') + '" data-filename="' + f + '">' +
                '<span class="doc-icon">📄</span>' +
                '<span class="doc-name">' + f + '</span>' +
                '<span class="doc-delete" data-filename="' + f + '">×</span>' +
                '</button>';
        }).join('');

        document.querySelectorAll('.doc-item').forEach(function(el) {
            el.addEventListener('click', function(e) {
                if (e.target.classList.contains('doc-delete')) return;
                self._selectDoc(this.dataset.filename);
            });
        });

        document.querySelectorAll('.doc-delete').forEach(function(el) {
            el.addEventListener('click', async function(e) {
                e.stopPropagation();
                var filename = this.dataset.filename;
                if (!confirm('确定删除文件 "' + filename + '" 吗？此操作不可恢复。')) return;
                try {
                    await API.del('/files/' + encodeURIComponent(filename));
                    State.files = State.files.filter(function(f) { return f !== filename; });
                    if (State.activeDoc === filename) {
                        State.activeDoc = null;
                        State.chapters = [];
                        self._showEmpty();
                    }
                    self._renderDocList();
                    Toast.show('已删除: ' + filename, 'info');
                } catch(err) {
                    Toast.show('删除失败: ' + err.message, 'error');
                }
            });
        });
    },

    _selectDoc: function(filename) {
        State.activeDoc = filename;
        State.selectedChapters = [];
        this._renderDocList();
        this._loadDocChapters(filename);
    },

    // ── Chapters ──────────────────────────────────────────────────

    _showEmpty: function() {
        var empty = document.getElementById('docsEmpty');
        var chapters = document.getElementById('docsChapters');
        if (empty) empty.style.display = 'flex';
        if (chapters) chapters.style.display = 'none';
    },

    _showChapters: function(filename, hasCache) {
        var empty = document.getElementById('docsEmpty');
        var chapters = document.getElementById('docsChapters');
        var title = document.getElementById('docsDocTitle');
        var detectBtn = document.getElementById('detectChaptersBtn');

        if (empty) empty.style.display = 'none';
        if (chapters) chapters.style.display = 'flex';
        if (title) title.textContent = '章节管理: ' + filename;
        if (detectBtn) detectBtn.textContent = hasCache ? '重新分析章节' : '检测章节';

        if (hasCache) {
            this._renderChapterList();
            this._updateProcessBtn();
            this._loadProcessedLabels(filename);
        } else {
            this._detectChapters();
        }
    },

    _loadDocChapters: async function(filename) {
        State.chapters = [];
        State.selectedChapters = [];
        try {
            var data = await API.get('/chapters/' + encodeURIComponent(filename));
            if (data.chapters && data.chapters.length) {
                State.chapters = data.chapters.map(function(c) {
                    return { title: c.title || c.label, label: c.label, text_length: c.text_length || 0, selected: false };
                });
                State.docChaptersCache[filename] = { chapters: data.chapters };
                this._showChapters(filename, true);
                return;
            }
        } catch(e) {}
        this._showChapters(filename, false);
    },

    // ── Chapter Detection ──────────────────────────────────────────

    _detectChapters: async function() {
        var filename = State.activeDoc;
        if (!filename) return;

        var progressEl = document.getElementById('chapterProgress');
        var progressFill = document.getElementById('progressFill');
        var progressText = document.getElementById('progressText');
        var chapterList = document.getElementById('chapterList');
        var detectBtn = document.getElementById('detectChaptersBtn');

        State.chapters = [];
        State.selectedChapters = [];
        if (chapterList) chapterList.innerHTML = '';
        this._updateProcessBtn();

        progressEl.style.display = 'block';
        progressEl.classList.add('animating');
        progressFill.style.width = '0%';
        progressText.textContent = '开始检测...';
        if (detectBtn) { detectBtn.disabled = true; detectBtn.textContent = '检测中...'; }

        await new Promise(function(r) { setTimeout(r, 50); });

        try {
            var res = await fetch(API.BASE + '/chapters/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filepaths: [filename] }),
            });

            var self = this;
            await SSE.readStream(res, function(evt) {
                self._handleChapterSSE(evt, progressEl, progressFill, progressText);
            });
        } catch(err) {
            Toast.show('章节检测失败: ' + err.message, 'error');
        }

        progressEl.style.display = 'none';
        progressEl.classList.remove('animating');
        if (detectBtn) { detectBtn.disabled = false; detectBtn.textContent = '重新分析章节'; }

        State.docChaptersCache[filename] = { chapters: State.chapters };
        try {
            await API.post('/chapters/save', { filename: filename, chapters: State.chapters });
        } catch(e) {}

        this._loadProcessedLabels(filename);
    },

    _handleChapterSSE: function(evt, progressEl, progressFill, progressText) {
        switch (evt.type) {
            case 'progress':
                progressText.textContent = evt.file + ': ' + evt.stage + ' (' + evt.file_idx + '/' + evt.file_total + ')';
                break;
            case 'file_done':
                var dpct = Math.round((evt.file_idx / evt.file_total) * 100);
                progressFill.style.width = dpct + '%';
                progressText.textContent = evt.file + ': 完成 (' + evt.chapters_found + ' 个章节)';
                break;
            case 'error':
                Toast.show(evt.file + ': ' + evt.msg, 'error');
                break;
            case 'complete':
                progressFill.style.width = '100%';
                progressEl.classList.remove('animating');
                progressText.textContent = '检测完成，共' + evt.total + ' 个章节';
                State.chapters = (evt.chapters || []).map(function(c) {
                    return { title: c.title || c.label, label: c.label, text_length: c.text_length || 0, selected: false };
                });
                State.selectedChapters = [];
                Toast.show('检测到 ' + evt.total + ' 个章节', 'success');
                this._renderChapterList();
                this._updateProcessBtn();
                break;
        }
    },

    // ── Chapter List ───────────────────────────────────────────────

    _selectAll: function() {
        State.chapters.forEach(function(c) { c.selected = true; });
        this._renderChapterList();
        this._updateProcessBtn();
    },

    _deselectAll: function() {
        State.chapters.forEach(function(c) { c.selected = false; });
        this._renderChapterList();
        this._updateProcessBtn();
    },

    _renderChapterList: function() {
        var list = document.getElementById('chapterList');
        if (!list) return;

        var filename = State.activeDoc;
        var processed = State.processedLabels[filename] || [];
        var self = this;

        list.innerHTML = State.chapters.map(function(ch, i) {
            var label = ch.label || ch.title;
            var isProcessed = processed.indexOf(label) !== -1;
            var badge = isProcessed ? '<span class="ch-badge imported">已导入</span>' : '<span class="ch-badge">未导入</span>';
            return '<div class="chapter-item' + (ch.selected ? ' selected' : '') + '" data-idx="' + i + '" title="' + label + '">' +
                '<div class="checkbox"></div>' +
                '<span class="ch-name">' + ch.title + '</span>' +
                badge +
                '<span class="ch-len">' + Markdown.formatSize(ch.text_length || 0) + '</span>' +
                '</div>';
        }).join('');

        document.querySelectorAll('.chapter-item').forEach(function(el) {
            el.addEventListener('click', function() {
                var idx = parseInt(this.dataset.idx);
                State.chapters[idx].selected = !State.chapters[idx].selected;
                self._renderChapterList();
                self._updateProcessBtn();
            });
        });
    },

    _updateProcessBtn: function() {
        var selected = State.chapters.filter(function(c) { return c.selected; });
        State.selectedChapters = selected.map(function(c) { return c.label; });
        var btn = document.getElementById('processBtn');
        if (btn) {
            btn.disabled = selected.length === 0;
            btn.textContent = selected.length ? '开始处理 (' + selected.length + ' 个章节)' : '开始处理选中章节';
        }
    },

    _loadProcessedLabels: async function(filename) {
        var fn = filename || State.activeDoc;
        if (!fn) return;
        try {
            var data = await API.get('/knowledge/documents');
            var docs = data.documents || [];
            var hasDoc = docs.some(function(d) { return d.indexOf(fn) !== -1; });
            if (hasDoc && (!State.processedLabels[fn] || !State.processedLabels[fn].length)) {
                State.processedLabels[fn] = State.selectedChapters;
            }
        } catch(err) {
            // Keep existing labels on error
        }
        this._renderChapterList();
        this._updateProcessBtn();
    },

    // ── Knowledge Processing ───────────────────────────────────────

    _processDocuments: async function() {
        if (!State.selectedChapters.length || !State.activeDoc) return;

        Toast.loading.show('正在处理...');
        try {
            var res = await fetch(API.BASE + '/knowledge/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filepaths: [State.activeDoc],
                    selected_chapters: State.selectedChapters,
                    session_id: State.sessionId,
                }),
            });

            var finalData = null;
            var self = this;
            await SSE.readStream(res, function(evt) {
                if (evt.type === 'progress') {
                    document.getElementById('loadingText').textContent = evt.stage;
                } else if (evt.type === 'error') {
                    Toast.show('处理失败: ' + evt.msg, 'error');
                } else if (evt.type === 'complete') {
                    finalData = evt;
                }
            });

            if (finalData) {
                State.processedLabels[State.activeDoc] = State.selectedChapters.slice();
                self._renderChapterList();
                self._updateProcessBtn();
                Toast.show('处理完成！概念: ' + finalData.stats.concepts + ', 关系: ' + finalData.stats.relations, 'success');
            }
        } catch(err) {
            Toast.show('处理失败: ' + err.message, 'error');
        }
        Toast.loading.hide();
    },
};
