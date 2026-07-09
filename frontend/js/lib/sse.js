// Document QA System — SSE Stream Reader
// Utility to read Server-Sent Events from a fetch response body.
var SSE = {
    readStream: async function(response, onEvent) {
        var reader = response.body.getReader();
        var decoder = new TextDecoder();
        var buffer = '';

        while (true) {
            var result = await reader.read();
            if (result.done) break;

            buffer += decoder.decode(result.value, { stream: true });
            var lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (var i = 0; i < lines.length; i++) {
                var line = lines[i].trim();
                if (!line.startsWith('data: ')) continue;
                try {
                    var evt = JSON.parse(line.slice(6));
                    onEvent(evt);
                } catch(e) {
                    // Skip malformed events (streaming edge cases)
                }
            }
        }
    },
};
