class HumanReasoningApp {
    constructor() {
        this.websocket = null;
        this.currentSessionId = null;
        this.currentRound = 1;
        this.currentQuestion = '';
        this.currentGroundTruth = '';
        this.initializeEventListeners();
        this.loadDatasets();
    }

    initializeEventListeners() {
        document.getElementById('sendButton').addEventListener('click', this.sendMessage.bind(this));
        document.getElementById('messageInput').addEventListener('keypress', this.onKeyPress.bind(this));
        document.getElementById('submitAnswerButton').addEventListener('click', this.submitFinalAnswer.bind(this));
        document.getElementById('answerInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.submitFinalAnswer();
            }
        });
    }

    async loadDatasets() {
        try {
            const response = await fetch('/api/datasets');
            const datasets = await response.json();
            const spubenchKey = 'spubench_500';
            if (datasets[spubenchKey]) {
                // Get all available samples (remove limit or set high limit)
                const response2 = await fetch(`/api/dataset/${spubenchKey}/samples?limit=1000`);
                const data = await response2.json();
                if (data.samples && data.samples.length > 0) {
                    this.samples = data.samples;
                    this.populateChallengeDropdown(data.samples);
                    this.displaySamples(data.samples);
                }
            }
        } catch (error) {
            console.error('Failed to load datasets:', error);
        }
    }

    populateChallengeDropdown(samples) {
        const select = document.getElementById('challengeSelect');
        select.innerHTML = '';
        samples.forEach((sample, idx) => {
            const option = document.createElement('option');
            option.value = idx;
            option.textContent = `Challenge ${idx + 1}`;
            select.appendChild(option);
        });
        select.disabled = false;
        select.onchange = (e) => {
            const idx = parseInt(e.target.value);
            const sample = this.samples[idx];
            this.startSession(sample.question, sample.path, sample.ground_truth);
        };
    }

    displaySamples(samples) {
        if (samples.length > 0) {
            // Show the first sample by default
            const sample = samples[0];
            this.startSession(sample.question, sample.path, sample.ground_truth);
        }
    }

    async startSession(question, imagePath, groundTruth) {
        document.getElementById('sendButton').disabled = true;
        document.getElementById('chatMessages').innerHTML = '';
        document.getElementById('answerInput').value = '';
        document.getElementById('submitAnswerButton').disabled = false;
        this.currentQuestion = question;
        this.currentGroundTruth = groundTruth;
        document.getElementById('questionDisplay').innerHTML = `<div>${question}</div>`;
        try {
            const response = await fetch('/api/start_session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: '',
                    question: question,
                    image_path: imagePath,
                    ground_truth: groundTruth
                })
            });
            if (!response.ok) {
                throw new Error('Failed to start session.');
            }
            const data = await response.json();
            if (!data.session_id) {
                throw new Error('No session_id returned from server.');
            }
            this.currentSessionId = data.session_id;
            this.connectWebSocket();
        } catch (error) {
            document.getElementById('questionDisplay').innerHTML = `<span style='color:red'>Failed to start session: ${error.message}</span>`;
            document.getElementById('sendButton').disabled = true;
            console.error('Failed to start session:', error);
        }
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.currentSessionId}`;
        this.websocket = new WebSocket(wsUrl);
        this.websocket.onopen = () => {
            document.getElementById('connectionStatus').textContent = 'Ready';
            document.getElementById('connectionStatus').style.color = '#4caf50';
            document.getElementById('sendButton').disabled = false;
        };
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        this.websocket.onclose = () => {
            document.getElementById('connectionStatus').textContent = 'Disconnected';
            document.getElementById('connectionStatus').style.color = '#f44336';
            document.getElementById('sendButton').disabled = true;
        };
        this.websocket.onerror = (error) => {
            document.getElementById('connectionStatus').textContent = 'Error';
            document.getElementById('connectionStatus').style.color = '#f44336';
            document.getElementById('sendButton').disabled = true;
            console.error('WebSocket error:', error);
        };
    }

    handleWebSocketMessage(data) {
        if (data.type === 'vision_response') {
            this.addMessage('vision', data.content);
            this.currentRound++;
        } else if (data.type === 'final_answer') {
            this.showResults(data);
        }
    }

    sendMessage() {
        const input = document.getElementById('messageInput');
        const message = input.value.trim();
        if (!message || !this.websocket) return;
        this.addMessage('human', message);
        this.websocket.send(JSON.stringify({ message: message }));
        input.value = '';
    }

    addMessage(role, content) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    submitFinalAnswer() {
    const answer = document.getElementById('answerInput').value.trim();
    if (!answer) return;
    
    // Send answer directly without "The answer is:" formatting
    this.addMessage('human', answer);
    if (this.websocket) {
        this.websocket.send(JSON.stringify({ message: answer }));
    }
    document.getElementById('submitAnswerButton').disabled = true;
    }

    showResults(data) {
        // Show result summary in chat area
        const messagesContainer = document.getElementById('chatMessages');
        const resultClass = data.correct ? 'correct' : 'incorrect';
        const resultIcon = data.correct ? '✅' : '❌';
        const resultDiv = document.createElement('div');
        resultDiv.className = `result-summary ${resultClass}`;
        resultDiv.innerHTML = `
            <h3>${resultIcon} ${data.correct ? 'Correct!' : 'Incorrect'}</h3>
            <p><strong>Your Answer:</strong> ${data.answer}</p>
            <p><strong>Ground Truth:</strong> ${data.ground_truth}</p>
            <p><strong>Rounds Taken:</strong> ${data.rounds}</p>
        `;
        messagesContainer.appendChild(resultDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    onKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }
}

const app = new HumanReasoningApp();
