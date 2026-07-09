// Document QA System — App Entry Point
// Orchestrates component initialization, panel switching, and document filter.
var App = {
    init: function() {
        // Init all components
        Toast.init();
        Settings.init();
        Sessions.init();
        Docs.init();
        Chat.init();

        // Navigation
        this._initNav();

        // Initial data load
        Docs.loadFiles();
        Sessions.refresh();
    },

    _initNav: function() {
        var self = this;
        document.querySelectorAll('.nav-item[data-panel]').forEach(function(item) {
            item.addEventListener('click', function() {
                self.switchPanel(this.dataset.panel);
            });
        });
    },

    switchPanel: function(panelId) {
        document.querySelectorAll('.nav-item[data-panel]').forEach(function(item) {
            item.classList.toggle('active', item.dataset.panel === panelId);
        });
        document.querySelectorAll('.panel').forEach(function(panel) {
            panel.classList.toggle('active', panel.id === 'panel-' + panelId);
        });
        if (panelId === 'docs') {
            Docs._renderDocList();
        } else if (panelId === 'qa') {
            Sessions.refresh();
        }
    },
};

// Boot on DOM ready
document.addEventListener('DOMContentLoaded', function() {
    App.init();
});
