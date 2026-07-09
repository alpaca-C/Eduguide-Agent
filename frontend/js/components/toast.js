// Document QA System — Toast Notification Component
var Toast = {
    container: null,

    init: function() {
        this.container = document.getElementById('toastContainer');
    },

    show: function(msg, type) {
        type = type || 'info';
        var el = document.createElement('div');
        el.className = 'toast ' + type;
        el.textContent = msg;
        this.container.appendChild(el);
        setTimeout(function() {
            el.style.opacity = '0';
            el.style.transition = '0.3s';
            setTimeout(function() { el.remove(); }, 300);
        }, 3000);
    },

    loading: {
        show: function(text) {
            document.getElementById('loadingText').textContent = text || '处理中...';
            document.getElementById('loadingOverlay').style.display = 'flex';
            State.processing = true;
        },
        hide: function() {
            document.getElementById('loadingOverlay').style.display = 'none';
            State.processing = false;
        },
    },
};
