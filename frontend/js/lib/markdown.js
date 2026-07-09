// Document QA System — Markdown + LaTeX Rendering
// Three-phase pipeline: protect LaTeX → parse Markdown → render KaTeX.
var Markdown = {
    render: function(text) {
        if (!text) return '';

        // Step 1: Protect LaTeX math from markdown parsing
        var mathBlocks = [];
        var idx = 0;

        // Protect display math: \[ ... \]
        while (text.indexOf('\\[') !== -1) {
            var ds = text.indexOf('\\[');
            var de = text.indexOf('\\]', ds + 2);
            if (de === -1) break;
            var formula = text.substring(ds + 2, de);
            var id = '\x00MATH' + (idx++) + '\x00';
            mathBlocks.push({ id: id, formula: formula, display: true });
            text = text.substring(0, ds) + id + text.substring(de + 2);
        }

        // Protect display math (alternate): $$ ... $$
        while (text.indexOf('$$') !== -1) {
            var dds = text.indexOf('$$');
            var dde = text.indexOf('$$', dds + 2);
            if (dde === -1) break;
            var formula2 = text.substring(dds + 2, dde);
            var id2 = '\x00MATH' + (idx++) + '\x00';
            mathBlocks.push({ id: id2, formula: formula2, display: true });
            text = text.substring(0, dds) + id2 + text.substring(dde + 2);
        }

        // Protect inline math: \( ... \)
        while (text.indexOf('\\(') !== -1) {
            var is = text.indexOf('\\(');
            var ie = text.indexOf('\\)', is + 2);
            if (ie === -1) break;
            var formula3 = text.substring(is + 2, ie);
            var id3 = '\x00MATH' + (idx++) + '\x00';
            mathBlocks.push({ id: id3, formula: formula3, display: false });
            text = text.substring(0, is) + id3 + text.substring(ie + 2);
        }

        // Step 2: Parse markdown
        var html;
        if (typeof marked !== 'undefined') {
            try {
                html = marked.parse(text);
            } catch(e) {
                console.warn('Markdown parse failed, falling back to plain text:', e);
                html = this._escapeHtml(text);
            }
        } else {
            html = this._escapeHtml(text);
        }

        // Step 3: Restore math — render with KaTeX if available
        for (var j = 0; j < mathBlocks.length; j++) {
            var block = mathBlocks[j];
            var rendered;
            if (typeof katex !== 'undefined') {
                try {
                    rendered = katex.renderToString(block.formula, {
                        displayMode: block.display,
                        throwOnError: false,
                        trust: false,
                    });
                } catch(e) {
                    rendered = block.display
                        ? '<span class="math-fallback">\\[' + this._escape(block.formula) + '\\]</span>'
                        : '<span class="math-fallback">\\(' + this._escape(block.formula) + '\\)</span>';
                }
            } else {
                rendered = block.display
                    ? '<span class="math-fallback">\\[' + this._escape(block.formula) + '\\]</span>'
                    : '<span class="math-fallback">\\(' + this._escape(block.formula) + '\\)</span>';
            }
            html = html.replace(block.id, rendered);
        }

        return html;
    },

    // Simple HTML escape for plain text (used during streaming before full render)
    escape: function(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\n/g, '<br>');
    },

    _escapeHtml: function(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\n/g, '<br>');
    },

    _escape: function(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    },

    formatSize: function(bytes) {
        if (!bytes || bytes < 1000) return (bytes || 0) + '字';
        if (bytes < 1000000) return (bytes / 1000).toFixed(1) + '千字';
        return (bytes / 1000000).toFixed(1) + '万字';
    },
};
