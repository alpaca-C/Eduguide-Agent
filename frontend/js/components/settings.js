// Document QA System — Settings & Theme Component
var Settings = {
    init: function() {
        var btn = document.getElementById('settingsBtn');
        var overlay = document.getElementById('settingsOverlay');
        var closeBtn = document.getElementById('settingsClose');
        var themeToggle = document.getElementById('themeToggle');

        if (btn && overlay) {
            btn.addEventListener('click', function() { overlay.style.display = 'flex'; });
        }
        if (closeBtn && overlay) {
            closeBtn.addEventListener('click', function() { overlay.style.display = 'none'; });
        }
        if (overlay) {
            overlay.addEventListener('click', function(e) {
                if (e.target === overlay) overlay.style.display = 'none';
            });
        }

        var savedTheme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);

        if (themeToggle) {
            var self = this;
            themeToggle.querySelectorAll('.theme-option').forEach(function(opt) {
                opt.classList.toggle('active', opt.dataset.theme === savedTheme);
                opt.addEventListener('click', function() {
                    var theme = this.dataset.theme;
                    document.documentElement.setAttribute('data-theme', theme);
                    localStorage.setItem('theme', theme);
                    themeToggle.querySelectorAll('.theme-option').forEach(function(o) {
                        o.classList.toggle('active', o.dataset.theme === theme);
                    });
                });
            });
        }
    },
};
