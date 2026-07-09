// Document QA System — Chat Component
var Chat = {
    init: function() {
        var sendBtn = document.getElementById('sendBtn');
        var chatInput = document.getElementById('chatInput');
        var newChatBtn = document.getElementById('newChatBtn');

        if (sendBtn) sendBtn.addEventListener('click', this.sendMessage.bind(this));
        if (chatInput) {
            var self = this;
            chatInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    self.sendMessage();
                }
            });
        }
        if (newChatBtn) newChatBtn.addEventListener('click', this.newConversation.bind(this));
    },

    sendMessage: async function() {
        var input = document.getElementById('chatInput');
        var question = input.value.trim();
        if (!question || State.processing) return;

        input.value = '';
        input.style.height = 'auto';

        var chatContainer = document.getElementById('chatContainer');
        this.addMessage('user', question);

        // Thinking indicator
        var thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'message assistant thinking';
        thinkingDiv.innerHTML = '<div class="msg-content">思考中...</div>';
        chatContainer.appendChild(thinkingDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        State.processing = true;

        // Gather doc filter from checkboxes
        var filterDocs = [];
        var checkboxes = document.querySelectorAll('#docFilter input[type=checkbox]:checked');
        checkboxes.forEach(function(cb) { filterDocs.push(cb.value); });

        try {
            var res = await fetch(API.BASE + '/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: question,
                    session_id: State.sessionId,
                    doc_filter: filterDocs,
                }),
            });

            var replyStarted = false;
            var replyText = '';
            var replyDiv = null;
            var finalSid = '';
            var self = this;

            await SSE.readStream(res, function(evt) {
                if (evt.type === 'status') {
                    thinkingDiv.innerHTML = '<div class="msg-content">' + Markdown.escape(evt.text) + '</div>';
                } else if (evt.type === 'reply_start') {
                    if (thinkingDiv) { thinkingDiv.remove(); thinkingDiv = null; }
                    replyDiv = document.createElement('div');
                    replyDiv.className = 'message assistant';
                    replyDiv.innerHTML = '<div class="msg-content"></div>';
                    chatContainer.appendChild(replyDiv);
                    replyStarted = true;
                } else if (evt.type === 'reply_chunk' && replyDiv) {
                    replyText += evt.text;
                    replyDiv.querySelector('.msg-content').innerHTML = Markdown.escape(replyText);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                } else if (evt.type === 'done') {
                    finalSid = evt.session_id;
                    if (replyDiv && replyText) {
                        replyDiv.querySelector('.msg-content').innerHTML = Markdown.render(replyText);
                    }
                }
            });

            if (finalSid) { State.sessionId = finalSid; Sessions._updateLabel(); }
            if (!replyStarted) {
                if (thinkingDiv) thinkingDiv.remove();
                self.addMessage('assistant', '抱歉，无法生成回答。');
            }
            try {
                State.chatHistory.push(
                    { role: 'user', content: question },
                    { role: 'assistant', content: replyText || '...' }
                );
            } catch(e) {}
        } catch(err) {
            if (thinkingDiv) thinkingDiv.remove();
            this.addMessage('assistant', '错误: ' + err.message);
        }
        State.processing = false;
    },

    addMessage: function(role, content) {
        var chatContainer = document.getElementById('chatContainer');
        var div = document.createElement('div');
        div.className = 'message ' + role;
        var formatted = role === 'assistant' ? Markdown.render(content) : Markdown.escape(content);
        div.innerHTML = '<div class="msg-content">' + formatted + '</div>';
        chatContainer.appendChild(div);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    },

    newConversation: function() {
        State.sessionId = '';
        State.chatHistory = [];
        document.getElementById('chatContainer').innerHTML = '';
        Sessions._updateLabel();
    },
};
