// Document QA System — EventBus
// Lightweight pub/sub for decoupled component communication.
// Components emit events; other components subscribe without direct coupling.
var Events = {
    _listeners: {},

    on: function(event, callback) {
        if (!this._listeners[event]) this._listeners[event] = [];
        this._listeners[event].push(callback);
    },

    off: function(event, callback) {
        var list = this._listeners[event];
        if (!list) return;
        this._listeners[event] = list.filter(function(cb) { return cb !== callback; });
    },

    emit: function(event, data) {
        var list = this._listeners[event];
        if (!list) return;
        list.forEach(function(cb) { cb(data); });
    },
};
