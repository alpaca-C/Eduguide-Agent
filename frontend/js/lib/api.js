// Document QA System — API Helper
// Centralized fetch wrapper with error handling.
var API = {
    BASE: '/api',

    get: function(path) {
        return this._request(path, { method: 'GET' });
    },

    post: function(path, body) {
        return this._request(path, {
            method: 'POST',
            body: JSON.stringify(body),
        });
    },

    del: function(path) {
        return this._request(path, { method: 'DELETE' });
    },

    _request: async function(path, options) {
        var url = path.startsWith('http') ? path : this.BASE + path;
        var res = await fetch(url, Object.assign({}, options, {
            headers: Object.assign(
                { 'Content-Type': 'application/json' },
                options.headers || {}
            ),
        }));
        if (!res.ok) {
            var err = await res.json().catch(function() {
                return { detail: res.statusText };
            });
            throw new Error(err.detail || 'Request failed');
        }
        return res.json();
    },
};
