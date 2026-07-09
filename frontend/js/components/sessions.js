// Document QA System — Sessions Component
var Sessions = {
    init: function() {
        // Session list is rendered on panel switch
    },

    refresh: async function() {
        try {
            var data = await API.get('/sessions');
            this._render(data.sessions || []);
        } catch(e) {
            console.warn('Failed to load sessions', e);
        }
    },

    _render: function(sessions) {
        var list = document.getElementById('sessionList');
        if (!list) return;
        var self = this;
        list.innerHTML = sessions.map(function(s) {
            return '<div class="session-item" data-sid="' + s.session_id + '" onclick="Sessions._load(\'' + s.session_id + '\')">' +
                   '<span>' + (s.topic || '新对话') + '</span>' +
                   '<button class="btn-sm" onclick="event.stopPropagation();Sessions._delete(\'' + s.session_id + '\')">删除</button>' +
                   '</div>';
        }).join('');
    },

    _load: async function(sid) {
        try {
            var data = await API.get('/sessions/' + sid);
            State.sessionId = sid;
            State.chatHistory = data.messages || [];
            this._updateLabel();
            var chatContainer = document.getElementById('chatContainer');
            chatContainer.innerHTML = '';
            (data.messages || []).forEach(function(m) {
                Chat.addMessage(m.role, m.content);
            });
            App.switchPanel('qa');
        } catch(e) {
            Toast.show('加载会话失败', 'error');
        }
    },

    _delete: async function(sid) {
        try {
            await API.del('/sessions/' + sid);
            if (State.sessionId === sid) Chat.newConversation();
            this.refresh();
        } catch(e) {
            console.warn('Failed to delete session', e);
        }
    },

    _updateLabel: function() {
        var label = document.getElementById('currentSessionLabel');
        if (label) label.textContent = State.sessionId ? '会话: ' + State.sessionId : '新会话';
    },
};
