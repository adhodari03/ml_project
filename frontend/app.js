/**
 * NexusAI v2 — Frontend Application Logic
 * Enhanced ML pipeline visualization, all models integrated.
 */

const API_BASE = 'http://localhost:8000';

// State
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let currentPipeline = [];
let messageHistory = [];
let currentAgentOverride = null; // Used to bypass auto-routing
let currentController = null; // Used for aborting fetch requests

// Voice state
let voiceTimerInterval = null;
let voiceSeconds = 0;
let finalVoiceBlob = null;
let isVoicePaused = false;

// Agent color mapping
const AGENT_COLORS = {
    'General Chat':   { bg: 'var(--accent-dim)',        color: 'var(--accent)' },
    'Task Manager':   { bg: 'var(--accent-blue-dim)',   color: 'var(--accent-blue)' },
    'Receipt Parser': { bg: 'var(--accent-orange-dim)', color: 'var(--accent-orange)' },
    'Document Q&A':   { bg: 'var(--accent-green-dim)',  color: 'var(--accent-green)' },
    'Code Debugger':  { bg: 'var(--accent-cyan-dim)',   color: 'var(--accent-cyan)' },
    'Study Buddy':    { bg: 'var(--accent-purple-dim)', color: 'var(--accent-purple)' },
    'Personal Finance':{ bg: 'var(--accent-pink-dim)',   color: 'var(--accent-pink)' },
    'Whisper':        { bg: 'var(--accent-pink-dim)',    color: 'var(--accent-pink)' },
    'System':         { bg: 'rgba(248,113,113,0.12)',    color: 'var(--accent-red)' },
};

// Pipeline stage metadata for visualization
const STAGE_META = {
    // LLM stages
    'tokenization':         { icon: '🔤', color: 'var(--accent)',       label: 'Tokenization' },
    'embedding':            { icon: '📐', color: 'var(--accent-blue)',  label: 'Embedding Lookup' },
    'transformer_layers':   { icon: '🧠', color: 'var(--accent-cyan)',  label: 'Transformer Forward Pass' },
    'output_generation':    { icon: '📝', color: 'var(--accent-green)', label: 'Output Sampling' },
    // Orchestrator stages
    'intent_classification':{ icon: '🎯', color: 'var(--accent-orange)',label: 'Intent Classification' },
    'rag_retrieval':        { icon: '🔍', color: 'var(--accent-green)', label: 'RAG Context Retrieval' },
    'agent_routing':        { icon: '🔀', color: 'var(--accent)',       label: 'Agent Routing' },
    'memory_storage':       { icon: '💾', color: 'var(--accent-pink)',  label: 'Memory Storage' },
    // RAG stages
    'text_embedding':       { icon: '📐', color: 'var(--accent-blue)',  label: 'Text Embedding' },
    'vector_indexing':      { icon: '📊', color: 'var(--accent-cyan)',  label: 'HNSW Indexing' },
    'query_embedding':      { icon: '🔎', color: 'var(--accent-blue)',  label: 'Query Embedding' },
    'similarity_search':    { icon: '📏', color: 'var(--accent-green)', label: 'Cosine Similarity Search' },
    'document_chunking':    { icon: '✂️', color: 'var(--accent-orange)',label: 'Document Chunking' },
    'batch_embedding_indexing': { icon: '📦', color: 'var(--accent-cyan)', label: 'Batch Embed & Index' },
    'rag_grounding':        { icon: '⚓', color: 'var(--accent-green)', label: 'RAG Grounding' },
    // Whisper stages
    'audio_preprocessing':  { icon: '🎵', color: 'var(--accent-pink)',  label: 'Audio → Mel Spectrogram' },
    'whisper_encoder':      { icon: '🎙️', color: 'var(--accent-pink)',  label: 'Whisper Encoder' },
    'whisper_decoder':      { icon: '📝', color: 'var(--accent-pink)',  label: 'Whisper Decoder' },
    'post_processing':      { icon: '✨', color: 'var(--accent-green)', label: 'Post-Processing' },
    // Vision stages
    'image_preprocessing':  { icon: '🖼️', color: 'var(--accent-orange)',label: 'Image Preprocessing' },
    'clip_visual_encoder':  { icon: '👁️', color: 'var(--accent-orange)',label: 'CLIP Visual Encoder' },
    'visual_projection':    { icon: '🔗', color: 'var(--accent-blue)',  label: 'Visual → LLM Projection' },
    'multimodal_decoding':  { icon: '🧠', color: 'var(--accent-cyan)',  label: 'Multimodal Decoding' },
    // Agent stages
    'task_execution':       { icon: '✅', color: 'var(--accent-blue)',  label: 'Task Execution' },
};

// ─── Init ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadTasks();
    setupDragDrop();
    generateAttentionMatrix();
    updateRAGView([]);  // Show default RAG explanation
    setInterval(checkHealth, 30000);

    // Enter key for Prompt Lab input
    document.getElementById('promptLabInput')?.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); runPromptLab(); }
    });
    // Enter key for Model Comparison input
    document.getElementById('modelCompareInput')?.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); runModelComparison(); }
    });
});

// ─── Health Check ────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');

        if (data.status === 'ok' && data.components?.llm?.ollama_running) {
            dot.classList.remove('offline');
            const models = data.components?.llm?.available_models || [];
            text.textContent = `Connected · ${models.length} models`;
            if (data.components?.memory) updateMemoryStats(data.components.memory);
        } else {
            dot.classList.add('offline');
            text.textContent = 'Ollama Offline';
        }
    } catch (e) {
        document.getElementById('statusDot').classList.add('offline');
        document.getElementById('statusText').textContent = 'Server Offline';
    }
}

// ─── Send Message ────────────────────────────────────────────
function stopMessage() {
    if (currentController) {
        currentController.abort();
        currentController = null;
    }
}

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message) return;

    const welcome = document.getElementById('welcomeScreen');
    if (welcome) welcome.style.display = 'none';

    addMessage(message, 'user');
    input.value = '';
    autoResize(input);

    const typingId = showTyping();
    document.getElementById('sendBtn').style.display = 'none';
    const stopBtn = document.getElementById('stopBtn');
    if (stopBtn) stopBtn.style.display = 'inline-block';

    currentController = new AbortController();

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                message, 
                agent_override: currentAgentOverride 
            }),
            signal: currentController.signal
        });
        const data = await res.json();

        removeTyping(typingId);
        addMessage(data.response, 'assistant', {
            agent: data.agent, intent: data.intent,
            confidence: data.confidence, time: data.total_time_ms
        });

        if (data.pipeline) {
            updatePipeline(data.pipeline);
            updateTokenFlow(message, data.response);
            updateRAGView(data.pipeline);
        }
        if (data.memory_stats) updateMemoryStats(data.memory_stats);
        highlightAgent(data.intent);
        if (data.intent === 'task_management') loadTasks();
    } catch (e) {
        if (e.name === 'AbortError') {
            removeTyping(typingId);
            addMessage('Generation stopped by user.', 'assistant', { agent: 'System' });
        } else {
            removeTyping(typingId);
            addMessage(`Connection error: ${e.message}. Is the server running?`, 'assistant', { agent: 'System' });
        }
    } finally {
        document.getElementById('sendBtn').style.display = 'inline-block';
        document.getElementById('sendBtn').disabled = false;
        const stopBtn = document.getElementById('stopBtn');
        if (stopBtn) stopBtn.style.display = 'none';
    }
}

function quickPrompt(text) {
    const input = document.getElementById('messageInput');
    input.value = text;
    input.focus();
}

// ─── Chat Messages ───────────────────────────────────────────
function addMessage(text, role, meta = {}) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `message ${role}`;

    const msgIndex = messageHistory.length;
    const avatar = role === 'user' ? '👤' : '🤖';
    const agentColors = AGENT_COLORS[meta.agent] || AGENT_COLORS['General Chat'];

    let badge = '';
    if (role === 'assistant' && meta.agent) {
        badge = `<div class="message-agent-badge" style="background:${agentColors.bg};color:${agentColors.color}">
            ${meta.agent}${meta.confidence ? ` · ${Math.round(meta.confidence * 100)}%` : ''}
        </div>`;
    }

    let metaHtml = '';
    if (meta.time || meta.intent || (role === 'assistant' && meta.pipeline)) {
        let btnStr = '';
        if (role === 'assistant' && meta.pipeline) {
            btnStr = `<button class="ml-explain-btn" id="mlExplainBtn-${msgIndex}" onclick="explainML(${msgIndex})">🧠 Explain ML Operations</button>`;
        }
        metaHtml = `<div class="message-meta">
            ${meta.time ? `<span>⏱️ ${meta.time.toFixed(0)}ms</span>` : ''}
            ${meta.intent ? `<span>· ${meta.intent.replace(/_/g, ' ')}</span>` : ''}
            ${btnStr}
        </div>`;
    }

    div.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            ${badge}
            <div class="message-text">${formatText(text)}</div>
            ${metaHtml}
        </div>`;

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    messageHistory.push({ text, role, meta });
}

function formatText(text) {
    if (!text) return '';
    let s = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    s = s.replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    return s;
}

// ─── Dynamic ML Explanation ───────────────────────────────────
async function explainML(index) {
    const btn = document.getElementById(`mlExplainBtn-${index}`);
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `🧠 Analyzing pipeline data...`;
    }

    // Try to get preceding user query constraint
    let userQuery = "Unknown context";
    if (index > 0 && messageHistory[index - 1].role === 'user') {
        userQuery = messageHistory[index - 1].text;
    }
    
    const targetMsg = messageHistory[index];
    const pipelineData = targetMsg.meta.pipeline;
    const systemPromptText = `You are an ML Architecture specialized explainer. The user asked: "${userQuery}". The pipeline array generated this telemetry: ${JSON.stringify(pipelineData)}. Please write a 3-paragraph explanation of the methodologies used, how the tokens or embeddings were mapped/transformed, what architectural layers executed this query (e.g. LLaVA, Whisper, Llama 3, ChromaDB/RAG vector search), and how they structurally interacted under the hood to generate the final response. Make it highly technical and insightful. Use bullet points and code markdown if necessary.`;

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message: systemPromptText, agent_override: 'general_chat' })
        });
        const data = await res.json();
        
        if (btn && btn.parentElement && btn.parentElement.parentElement) {
            const expDiv = document.createElement('div');
            expDiv.className = 'ml-explanation-box';
            expDiv.innerHTML = `<strong>Pipeline Methodology Analysis</strong><br/>` + formatText(data.response);
            btn.parentElement.parentElement.appendChild(expDiv);
        }
        if (btn) btn.style.display = 'none'; // hide after success
    } catch (e) {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `⚠️ Error connecting to NexusAI`;
        }
    }
}

// ─── Typing Indicator ────────────────────────────────────────
function showTyping() {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    const id = 'typing-' + Date.now();
    div.id = id;
    div.className = 'message assistant';
    div.innerHTML = `<div class="message-avatar">🤖</div>
        <div class="message-content"><div class="typing-indicator">
            <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
        </div></div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeTyping(id) { document.getElementById(id)?.remove(); }

// ─── Voice Recording (Whisper) ───────────────────────────────
async function toggleVoice() {
    if (isRecording) return; // Prevent double trigger
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        audioChunks = [];
        
        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        
        mediaRecorder.onstop = () => {
            finalVoiceBlob = new Blob(audioChunks, { type: 'audio/webm' });
            stream.getTracks().forEach(t => t.stop());
            
            // Switch to review mode
            document.getElementById('voiceRecordingBar').style.display = 'none';
            document.getElementById('voiceReviewBar').style.display = 'flex';
            
            const player = document.getElementById('voicePlayer');
            player.src = URL.createObjectURL(finalVoiceBlob);
        };
        
        // Reset states
        voiceSeconds = 0;
        isVoicePaused = false;
        document.getElementById('voiceTime').textContent = '00:00';
        document.getElementById('recordingPulse').className = 'recording-pulse active';
        document.getElementById('voicePauseBtn').textContent = '⏸️';
        
        // UI toggle
        document.getElementById('textInputContainer').style.display = 'none';
        document.getElementById('voiceContainer').style.display = 'flex';
        document.getElementById('voiceRecordingBar').style.display = 'flex';
        document.getElementById('voiceReviewBar').style.display = 'none';
        
        // Start recording
        mediaRecorder.start();
        isRecording = true;
        
        voiceTimerInterval = setInterval(() => {
            if (!isVoicePaused) {
                voiceSeconds++;
                const mins = String(Math.floor(voiceSeconds / 60)).padStart(2, '0');
                const secs = String(voiceSeconds % 60).padStart(2, '0');
                document.getElementById('voiceTime').textContent = `${mins}:${secs}`;
            }
        }, 1000);
        
    } catch (e) {
        addMessage('Microphone access denied. Please allow permissions.', 'assistant', { agent: 'System' });
    }
}

function pauseResumeVoice() {
    if (!mediaRecorder || !isRecording) return;
    
    const pulse = document.getElementById('recordingPulse');
    const pauseBtn = document.getElementById('voicePauseBtn');
    
    if (isVoicePaused) {
        mediaRecorder.resume();
        pulse.className = 'recording-pulse active';
        pauseBtn.textContent = '⏸️';
        pauseBtn.title = 'Pause';
    } else {
        mediaRecorder.pause();
        pulse.className = 'recording-pulse paused';
        pauseBtn.textContent = '▶️';
        pauseBtn.title = 'Resume';
    }
    isVoicePaused = !isVoicePaused;
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        clearInterval(voiceTimerInterval);
    }
}

function discardVoice() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop(); // Stops tracks
        isRecording = false;
    }
    clearInterval(voiceTimerInterval);
    
    // Reset UI back to text input
    document.getElementById('voiceContainer').style.display = 'none';
    document.getElementById('textInputContainer').style.display = 'flex';
    document.getElementById('voicePlayer').src = '';
    
    audioChunks = [];
    finalVoiceBlob = null;
}

async function submitVoice() {
    if (!finalVoiceBlob) return;
    
    const blobToProcess = finalVoiceBlob;
    discardVoice(); // Resets and closes UI
    
    await sendVoice(blobToProcess);
}

async function sendVoice(blob) {
    const welcome = document.getElementById('welcomeScreen');
    if (welcome) welcome.style.display = 'none';
    addMessage('🎤 [Voice message sent to Whisper ASR]', 'user');
    const typingId = showTyping();

    try {
        const fd = new FormData();
        fd.append('audio', blob, 'recording.webm');
        const res = await fetch(`${API_BASE}/api/voice`, { method: 'POST', body: fd });
        const data = await res.json();
        removeTyping(typingId);

        if (data.transcription) {
            addMessage(`📝 Whisper transcription: "${data.transcription}"`, 'assistant', { agent: 'Whisper' });
        }
        addMessage(data.response, 'assistant', {
            agent: data.agent, intent: data.intent,
            confidence: data.confidence, time: data.total_time_ms
        });
        if (data.pipeline) { updatePipeline(data.pipeline); updateRAGView(data.pipeline); }
        if (data.memory_stats) updateMemoryStats(data.memory_stats);
    } catch (e) {
        removeTyping(typingId);
        addMessage(`Voice error: ${e.message}`, 'assistant', { agent: 'System' });
    }
}

// ─── Image Upload (LLaVA) ────────────────────────────────────
async function uploadImage(event) {
    const file = event.target.files[0];
    if (!file) return;
    const welcome = document.getElementById('welcomeScreen');
    if (welcome) welcome.style.display = 'none';

    let userMessage = document.getElementById('messageInput').value.trim();
    if (!userMessage) {
        userMessage = 'Describe this image in detail.';
    } else {
        document.getElementById('messageInput').value = '';
        autoResize(document.getElementById('messageInput'));
    }

    addMessage(`🖼️ Uploaded: ${file.name}  — "${userMessage}"`, 'user');
    const typingId = showTyping();

    try {
        const fd = new FormData();
        fd.append('image', file);
        fd.append('message', userMessage);
        const res = await fetch(`${API_BASE}/api/image`, { method: 'POST', body: fd });
        const data = await res.json();
        removeTyping(typingId);

        addMessage(data.response, 'assistant', {
            agent: data.agent || 'Receipt Parser',
            intent: data.intent, confidence: data.confidence, time: data.total_time_ms
        });
        if (data.pipeline) { updatePipeline(data.pipeline); updateRAGView(data.pipeline); }
    } catch (e) {
        removeTyping(typingId);
        addMessage(`Image error: ${e.message}`, 'assistant', { agent: 'System' });
    }
    event.target.value = '';
}

// ─── Document Upload (RAG) ───────────────────────────────────
async function uploadDocument(event) {
    const file = event.target.files[0];
    if (!file) return;

    addMessage(`📄 Uploading "${file.name}" for RAG ingestion`, 'user');
    const typingId = showTyping();

    try {
        const fd = new FormData();
        fd.append('document', file);
        const res = await fetch(`${API_BASE}/api/document`, { method: 'POST', body: fd });
        const data = await res.json();
        removeTyping(typingId);

        if (data.status === 'success') {
            addMessage(
                `✅ "${data.filename}" ingested into RAG store\n\n` +
                `• ${data.chunks_stored} chunks embedded as 768-dim vectors\n` +
                `• Stored in ChromaDB HNSW index\n` +
                `• Ready for semantic Q&A — ask anything about this document`, 
                'assistant', { agent: 'Document Q&A', time: data.total_time_ms }
            );
            if (data.pipeline) { updatePipeline(data.pipeline); updateRAGView(data.pipeline); }
        } else {
            addMessage(`❌ ${data.error}`, 'assistant', { agent: 'System' });
        }
    } catch (e) {
        removeTyping(typingId);
        addMessage(`Upload error: ${e.message}`, 'assistant', { agent: 'System' });
    }
    event.target.value = '';
}

// ─── Drag & Drop ─────────────────────────────────────────────
function setupDragDrop() {
    const overlay = document.getElementById('uploadOverlay');
    let counter = 0;
    document.addEventListener('dragenter', e => { e.preventDefault(); counter++; overlay.classList.add('active'); });
    document.addEventListener('dragleave', e => { e.preventDefault(); counter--; if (!counter) overlay.classList.remove('active'); });
    document.addEventListener('dragover', e => e.preventDefault());
    document.addEventListener('drop', e => {
        e.preventDefault(); counter = 0; overlay.classList.remove('active');
        const file = e.dataTransfer.files[0];
        if (!file) return;
        if (file.type.startsWith('image/')) {
            uploadImage({ target: { files: [file], value: '' } });
        } else {
            uploadDocument({ target: { files: [file], value: '' } });
        }
    });
}

// ═══════════════════════════════════════════════════════════════
//  ML PIPELINE VISUALIZER
// ═══════════════════════════════════════════════════════════════

function updatePipeline(pipeline) {
    currentPipeline = pipeline;
    const el = document.getElementById('pipelineView');
    if (!pipeline?.length) {
        el.innerHTML = `<div class="empty-state"><div class="empty-state-icon">🔬</div>
            <div class="empty-state-text">No pipeline data yet.</div></div>`;
        return;
    }

    el.innerHTML = pipeline.map((step, i) => {
        const meta = STAGE_META[step.stage] || { icon: '⚙️', color: 'var(--text-muted)', label: step.stage };
        const details = step.details || {};
        const expanded = i < 4 ? 'expanded' : '';

        return `
        <div class="pipeline-step ${expanded}" onclick="this.classList.toggle('expanded')">
            <div class="pipeline-step-header">
                <div class="pipeline-step-left">
                    <div class="pipeline-step-icon" style="background:${meta.color}15;color:${meta.color}">${meta.icon}</div>
                    <span class="pipeline-step-name">${meta.label}</span>
                </div>
                <span class="pipeline-step-time">${step.duration_ms != null ? step.duration_ms.toFixed(1) + 'ms' : ''}</span>
            </div>
            <div class="pipeline-step-body">
                <div class="pipeline-step-desc">${step.description || ''}</div>
                ${Object.keys(details).length ? `<div class="pipeline-details">
                    ${Object.entries(details).map(([k,v]) => `
                        <div class="detail-row">
                            <span class="detail-key">${k.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</span>
                            <span class="detail-value">${fmtVal(v)}</span>
                        </div>`).join('')}
                </div>` : ''}
            </div>
        </div>`;
    }).join('');

    switchVizTab('pipeline');
}

function fmtVal(v) {
    if (v == null) return '—';
    if (typeof v === 'number') return v.toLocaleString();
    if (Array.isArray(v)) return v.length > 4 ? `[${v.length} items]` : v.join(', ');
    if (typeof v === 'object') return JSON.stringify(v);
    return String(v);
}

// ─── Attention Matrix ────────────────────────────────────────
function generateAttentionMatrix(tokens) {
    const el = document.getElementById('attentionMatrix');
    const toks = tokens || ['<s>', 'The', 'trans', 'former', 'uses', 'self', 'attention'];
    const n = toks.length;

    el.style.gridTemplateColumns = `50px repeat(${n}, 1fr)`;
    el.style.gridTemplateRows = `24px repeat(${n}, 1fr)`;

    let html = '<div></div>';
    for (let j = 0; j < n; j++) html += `<div class="attention-label">${toks[j]}</div>`;

    for (let i = 0; i < n; i++) {
        html += `<div class="attention-label" style="display:flex;align-items:center;justify-content:flex-end;padding-right:3px">${toks[i]}</div>`;
        for (let j = 0; j < n; j++) {
            let w = 0;
            if (j <= i) {
                const d = Math.abs(i - j);
                w = Math.min(1, Math.max(0, 1 - d * 0.13) + (j === 0 ? 0.25 : 0) + (i === j ? 0.18 : 0) + Math.random() * 0.12);
            }
            const bg = j > i ? 'rgba(255,255,255,0.015)' : `rgba(129,140,248,${w * 0.75})`;
            html += `<div class="attention-cell" style="background:${bg}" title="${toks[i]}→${toks[j]}: ${w.toFixed(3)}"></div>`;
        }
    }
    el.innerHTML = html;
}

// ─── Token Flow ──────────────────────────────────────────────
function updateTokenFlow(input, output) {
    const el = document.getElementById('tokenFlowContainer');
    const inToks = tokenize(input);
    const outToks = tokenize(output).slice(0, 12);

    el.innerHTML = `
        <div class="token-flow-stage" style="border-left-color:var(--accent)">
            <div class="token-flow-label" style="color:var(--accent)">Input</div>
            <div class="token-flow-content">
                <div style="font-size:12px;color:var(--text-secondary);margin-bottom:4px">"${input.slice(0,70)}${input.length>70?'...':''}"</div>
            </div>
        </div>
        <div class="token-flow-stage" style="border-left-color:var(--accent-orange)">
            <div class="token-flow-label" style="color:var(--accent-orange)">Tokenize</div>
            <div class="token-flow-content">
                <div class="token-flow-tokens">
                    ${inToks.slice(0,10).map(t => `<span class="token">${t}</span>`).join('')}
                    ${inToks.length > 10 ? `<span class="token" style="opacity:0.4">+${inToks.length-10}</span>` : ''}
                </div>
                <div style="font-size:10px;color:var(--text-muted);margin-top:6px">BPE tokenizer → ${inToks.length} tokens → integer IDs</div>
            </div>
        </div>
        <div class="token-flow-stage" style="border-left-color:var(--accent-blue)">
            <div class="token-flow-label" style="color:var(--accent-blue)">Embed</div>
            <div class="token-flow-content">
                <div class="token-flow-tokens">
                    ${inToks.slice(0,4).map(() => {
                        const v1 = (Math.random()*2-1).toFixed(2), v2 = (Math.random()*2-1).toFixed(2);
                        return `<span class="token embedding" style="font-size:9px">[${v1}, ${v2}, ...×4096]</span>`;
                    }).join('')}
                </div>
                <div style="font-size:10px;color:var(--text-muted);margin-top:6px">Embedding lookup (128,256 × 4096) + RoPE positional encoding</div>
            </div>
        </div>
        <div class="token-flow-stage" style="border-left-color:var(--accent-cyan)">
            <div class="token-flow-label" style="color:var(--accent-cyan)">Transform</div>
            <div class="token-flow-content">
                <div style="font-size:11px;color:var(--text-secondary);line-height:1.6">
                    × 32 layers: RMSNorm → Multi-Head Self-Attention (GQA) → Residual → RMSNorm → SwiGLU FFN → Residual
                </div>
            </div>
        </div>
        <div class="token-flow-stage" style="border-left-color:var(--accent-green)">
            <div class="token-flow-label" style="color:var(--accent-green)">Output</div>
            <div class="token-flow-content">
                <div class="token-flow-tokens">
                    ${outToks.map(t => `<span class="token output">${t}</span>`).join('')}
                    ${outToks.length >= 12 ? '<span class="token output" style="opacity:0.4">...</span>' : ''}
                </div>
                <div style="font-size:10px;color:var(--text-muted);margin-top:6px">Linear head → softmax → temperature sampling → detokenize</div>
            </div>
        </div>`;

    if (inToks.length > 2) generateAttentionMatrix(inToks.slice(0, 8));
}

function tokenize(text) {
    if (!text) return [];
    return text.split(/(\s+|[.,!?;:'"()\[\]{}])/).filter(t => t.trim());
}

// ─── RAG Pipeline View ──────────────────────────────────────
function updateRAGView(pipeline) {
    const el = document.getElementById('ragPipeline');

    const ragStages = pipeline.filter(s =>
        ['text_embedding','vector_indexing','query_embedding','similarity_search',
         'document_chunking','batch_embedding_indexing','rag_retrieval','rag_grounding'].includes(s.stage));

    if (!ragStages.length) {
        el.innerHTML = `
        <div class="rag-stage">
            <div class="rag-stage-title" style="color:var(--accent-green)">📚 Retrieval-Augmented Generation (RAG)</div>
            <div style="font-size:12px;color:var(--text-secondary);line-height:1.8">
                RAG extends the LLM beyond its training data by retrieving relevant information at inference time.<br><br>
                <strong style="color:var(--accent-orange)">1. Ingest</strong> — Document → chunk (500 chars, 50 overlap) → embed each chunk via nomic-embed-text → store 768-dim vectors in ChromaDB<br>
                <strong style="color:var(--accent-blue)">2. Query</strong> — User question → embed into same vector space<br>
                <strong style="color:var(--accent-green)">3. Retrieve</strong> — Cosine similarity search in HNSW index → top-5 chunks<br>
                <strong style="color:var(--accent)">4. Inject</strong> — Prepend retrieved chunks to LLM prompt as context<br>
                <strong style="color:var(--accent-cyan)">5. Generate</strong> — LLM answers grounded in retrieved evidence<br><br>
                <strong>Why RAG?</strong> Reduces hallucination. Enables knowledge updates without retraining. Extends effective context beyond the model's fixed window.
            </div>
        </div>
        <div class="rag-stage">
            <div class="rag-stage-title" style="color:var(--accent-blue)">📐 Cosine Similarity</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:var(--accent-cyan);text-align:center;padding:10px;background:rgba(0,0,0,0.2);border-radius:8px;">
                sim(u, v) = (u · v) / (‖u‖ × ‖v‖)
            </div>
            <div style="font-size:11px;color:var(--text-muted);margin-top:8px;line-height:1.6">
                Measures angle between embedding vectors. 1.0 = identical meaning, 0 = unrelated.
                Used in HNSW index for O(log n) approximate nearest neighbor search.
            </div>
        </div>
        <div class="rag-stage">
            <div class="rag-stage-title" style="color:var(--accent-cyan)">🗄️ ChromaDB + HNSW</div>
            <div style="font-size:11px;color:var(--text-muted);line-height:1.6">
                ChromaDB stores vectors in an HNSW (Hierarchical Navigable Small World) graph.
                HNSW builds a multi-layer proximity graph where each layer acts as a skip list.
                Search starts at the top (sparse) layer and navigates down to the bottom (dense) layer,
                achieving O(log n) query time vs O(n) for brute force.
            </div>
        </div>`;
        return;
    }

    el.innerHTML = ragStages.map(stage => {
        const meta = STAGE_META[stage.stage] || { icon: '⚙️', color: 'var(--text-muted)', label: stage.stage };
        const details = stage.details || {};

        let extra = '';
        if (stage.stage === 'similarity_search' && details.distances) {
            extra = (Array.isArray(details.distances) ? details.distances : []).slice(0,5).map((d,i) => {
                const sim = Math.max(0, 1 - d);
                return `<div style="margin-top:5px">
                    <div style="display:flex;justify-content:space-between;font-size:10px">
                        <span style="color:var(--text-muted)">Result ${i+1}</span>
                        <span style="color:var(--accent-green);font-family:'JetBrains Mono',monospace">${(sim*100).toFixed(1)}%</span>
                    </div>
                    <div class="rag-similarity-bar"><div class="rag-similarity-fill" style="width:${sim*100}%"></div></div>
                </div>`;
            }).join('');
        }
        if (stage.stage === 'document_chunking' && details.num_chunks) {
            const count = Math.min(details.num_chunks, 15);
            extra = `<div class="rag-chunks" style="margin-top:8px">
                ${Array.from({length:count},(_,i) => `<span class="rag-chunk">Chunk ${i+1}</span>`).join('')}
                ${details.num_chunks > 15 ? `<span class="rag-chunk" style="opacity:0.4">+${details.num_chunks-15}</span>` : ''}
            </div>`;
        }

        return `<div class="rag-stage">
            <div class="rag-stage-title" style="color:${meta.color}">
                ${meta.icon} ${meta.label}
                <span style="font-size:10px;color:var(--text-muted);font-weight:400;margin-left:auto">${stage.duration_ms?.toFixed(1)||''}ms</span>
            </div>
            <div style="font-size:11px;color:var(--text-secondary);line-height:1.6">${stage.description||''}</div>
            ${Object.keys(details).length ? `<div class="pipeline-details" style="margin-top:8px">
                ${Object.entries(details).filter(([k])=>k!=='distances').map(([k,v]) => `
                    <div class="detail-row"><span class="detail-key">${k.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</span>
                    <span class="detail-value">${fmtVal(v)}</span></div>`).join('')}
            </div>` : ''}
            ${extra}
        </div>`;
    }).join('');
}

// ─── Tab Switching ───────────────────────────────────────────
function switchVizTab(tab) {
    document.querySelectorAll('.viz-tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.viz-tab[data-tab="${tab}"]`)?.classList.add('active');

    const views = {
        pipeline:'pipelineView', architecture:'architectureView',
        attention:'attentionView', tokens:'tokenFlowView', rag:'ragView',
        embeddings:'embeddingsView', promptlab:'promptlabView', models:'modelsView'
    };
    Object.entries(views).forEach(([k,id]) => {
        const el = document.getElementById(id);
        if (el) el.style.display = k === tab ? 'block' : 'none';
    });

    // Auto-load embeddings when switching to that tab
    if (tab === 'embeddings' && !document.getElementById('embeddingCanvas').hasChildNodes()) {
        loadEmbeddings();
    }
}

// ─── Sidebar ─────────────────────────────────────────────────
function highlightAgent(intent) {
    document.querySelectorAll('.agent-card').forEach(c => {
        c.classList.toggle('active', c.dataset.agent === intent);
    });
}

function updateMemoryStats(s) {
    document.getElementById('memConversations').textContent = s.conversations || 0;
    document.getElementById('memDocuments').textContent = s.documents || 0;
    document.getElementById('memTasks').textContent = s.tasks || 0;
}

async function loadTasks() {
    try {
        const res = await fetch(`${API_BASE}/api/tasks`);
        const data = await res.json();
        const el = document.getElementById('taskList');
        if (!data.tasks?.length) {
            el.innerHTML = `<div style="font-size:11px;color:var(--text-muted)">No tasks yet.</div>`;
            return;
        }
        el.innerHTML = data.tasks.map(t => `
            <div class="task-item ${t.status==='completed'?'completed':''}">
                <div class="task-checkbox ${t.status==='completed'?'checked':''}"></div>
                <span>${t.text||'Untitled'}</span>
            </div>`).join('');
    } catch(e) {}
}

// ─── Input Helpers ───────────────────────────────────────────
function handleKeydown(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } }
function autoResize(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 120) + 'px'; }

function startNewChat() {
    // Reset UI only
    document.getElementById('chatMessages').innerHTML = `
        <div class="welcome-screen" id="welcomeScreen">
            <div class="welcome-icon">🤖</div>
            <h1 class="welcome-title" id="welcomeTitle">Welcome to NexusAI</h1>
            <p class="welcome-desc" id="welcomeDesc">Ask anything — NexusAI routes your request to the right agent automatically.</p>
        </div>`;
    messageHistory = [];
    currentAgentOverride = null;
    selectAgent('general_chat'); // Reset to general routing visually
}

async function clearDatabase() {
    // Call backend to wipe ChromaDB Vector DB
    if (!confirm("Are you sure you want to delete all memory and context from the backend?")) return;
    
    try {
        const res = await fetch(`${API_BASE}/api/memory/clear`, { method: 'POST' });
        const data = await res.json();
        if (data.status === 'ok') {
            alert('Backend memory wiped successfully!');
            startNewChat();
            updateMemoryStats({conversations: 0, documents: 0, tasks: 0});
        } else {
            alert('Error clearing memory: ' + data.error);
        }
    } catch(e) {
        alert('Server unreachable to clear memory.');
    }
}

function selectAgent(agentIntent) {
    currentAgentOverride = agentIntent === 'general_chat' ? null : agentIntent;
    
    // UI Updates
    document.querySelectorAll('.agent-card').forEach(c => {
        c.classList.toggle('active', c.dataset.agent === agentIntent);
    });

    const titles = {
        'general_chat': 'Welcome to NexusAI',
        'task_management': 'Task Manager Agent',
        'receipt_parsing': 'Receipt Parser Agent',
        'document_qa': 'Document Q&A Agent',
        'code_debugging': 'Code Debugger Agent',
        'study_buddy': 'Study Buddy & Tutor',
        'personal_finance': 'Personal Finance Agent'
    };

    const descs = {
        'general_chat': 'Ask anything — NexusAI routes your request to the right agent automatically.',
        'task_management': 'I am locked to Task Management. Tell me what tasks to create, list, or track.',
        'receipt_parsing': 'I am locked to Receipt Parsing. Send me text and I will extract line items and totals.',
        'document_qa': 'I am locked to Document Analysis. I will only answer based on retrieved vector memory.',
        'code_debugging': 'I am locked to Code Debugging. Paste your code and error to get explanations and fixes.',
        'study_buddy': 'I am locked to Socratic Tutoring. I will ask guiding questions and explain core concepts without giving direct answers.',
        'personal_finance': 'I am locked to Personal Finance. I handle multi-step actions like receipt parsing, tip calculation, expense logging, and checking memory for past spending.'
    };

    const titleEl = document.getElementById('welcomeTitle');
    const descEl = document.getElementById('welcomeDesc');
    
    if (titleEl && descEl) {
        titleEl.textContent = titles[agentIntent] || 'NexusAI';
        descEl.textContent = descs[agentIntent] || 'Ready for input.';
    }
}

// ═══════════════════════════════════════════════════════════════
//  TIER 2: EMBEDDING SPACE VISUALIZATION (t-SNE)
// ═══════════════════════════════════════════════════════════════

async function loadEmbeddings() {
    const canvas = document.getElementById('embeddingCanvas');
    const stats = document.getElementById('embeddingStats');
    const btn = document.getElementById('loadEmbBtn');

    btn.textContent = '⏳ Computing t-SNE...';
    btn.disabled = true;
    canvas.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-muted);font-size:12px">Running t-SNE dimensionality reduction...</div>';

    try {
        const res = await fetch(`${API_BASE}/api/embeddings/visualize`);
        const data = await res.json();

        if (!data.points || data.points.length === 0) {
            canvas.innerHTML = `<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;text-align:center;padding:20px">
                <div style="font-size:28px;margin-bottom:10px">📊</div>
                <div style="font-size:12px;color:var(--text-muted);line-height:1.7">${data.message || 'No embeddings yet. Send some messages first!'}</div>
            </div>`;
            stats.innerHTML = '';
            btn.textContent = '🔄 Refresh Visualization';
            btn.disabled = false;
            return;
        }

        // Draw scatter plot using CSS-positioned divs (no canvas dependency)
        const pad = 30;
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;

        let html = '';

        // Grid lines
        html += '<div style="position:absolute;top:50%;left:0;right:0;height:1px;background:rgba(255,255,255,0.04)"></div>';
        html += '<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:rgba(255,255,255,0.04)"></div>';

        // Axis labels
        html += '<div style="position:absolute;bottom:4px;right:8px;font-size:9px;color:var(--text-muted)">t-SNE dim 1 →</div>';
        html += '<div style="position:absolute;top:4px;left:8px;font-size:9px;color:var(--text-muted)">↑ t-SNE dim 2</div>';

        // Plot points
        data.points.forEach((pt, i) => {
            const x = pad + ((pt.x + 1) / 2) * (w - 2 * pad);
            const y = pad + ((1 - (pt.y + 1) / 2)) * (h - 2 * pad);
            const truncText = pt.text.length > 40 ? pt.text.slice(0,40) + '...' : pt.text;

            html += `<div class="emb-point" style="
                position:absolute;
                left:${x}px; top:${y}px;
                width:10px; height:10px;
                border-radius:50%;
                background:${pt.color};
                transform:translate(-50%,-50%);
                opacity:0.8;
                cursor:pointer;
                transition:all 0.15s ease;
                z-index:1;
                animation: tokenPop 0.3s ease ${i * 0.02}s both;
            " title="[${pt.category}] ${truncText}"
               onmouseenter="this.style.transform='translate(-50%,-50%) scale(2)';this.style.opacity='1';this.style.zIndex='10'"
               onmouseleave="this.style.transform='translate(-50%,-50%) scale(1)';this.style.opacity='0.8';this.style.zIndex='1'"
            ></div>`;
        });

        // Legend
        const cats = {};
        data.points.forEach(p => { cats[p.category] = p.color; });
        html += '<div style="position:absolute;bottom:8px;left:8px;display:flex;gap:12px">';
        for (const [cat, color] of Object.entries(cats)) {
            html += `<div style="display:flex;align-items:center;gap:4px;font-size:10px;color:var(--text-muted)">
                <div style="width:8px;height:8px;border-radius:50%;background:${color}"></div>
                ${cat}
            </div>`;
        }
        html += '</div>';

        canvas.innerHTML = html;

        // Stats
        stats.innerHTML = `
        <div class="pipeline-details">
            <div class="detail-row"><span class="detail-key">Total Vectors</span><span class="detail-value">${data.total_vectors}</span></div>
            <div class="detail-row"><span class="detail-key">Original Dimensions</span><span class="detail-value">${data.dimensions_original}</span></div>
            <div class="detail-row"><span class="detail-key">Algorithm</span><span class="detail-value">${data.algorithm}</span></div>
            <div class="detail-row"><span class="detail-key">Perplexity</span><span class="detail-value">${data.perplexity}</span></div>
        </div>
        <div style="margin-top:10px;padding:12px;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-md)">
            <div style="font-size:10px;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px">HOW t-SNE WORKS</div>
            <div style="font-size:11px;color:var(--text-secondary);line-height:1.65">
                t-SNE (t-distributed Stochastic Neighbor Embedding) converts pairwise distances in high-dimensional space
                to probabilities: nearby points get high probability, distant points get low probability.
                It then finds a 2D arrangement that minimizes KL divergence between the high-D and low-D probability distributions.
                The heavy-tailed t-distribution in 2D prevents the "crowding problem" — allowing clear cluster separation.
            </div>
        </div>`;

    } catch (e) {
        canvas.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--accent-red);font-size:12px">Error: ${e.message}</div>`;
    }

    btn.textContent = '🔄 Refresh Visualization';
    btn.disabled = false;
}

// ═══════════════════════════════════════════════════════════════
//  TIER 2: PROMPT ENGINEERING LAB
// ═══════════════════════════════════════════════════════════════

async function runPromptLab() {
    const input = document.getElementById('promptLabInput');
    const message = input.value.trim();
    if (!message) return;

    const el = document.getElementById('promptLabResults');
    const btn = document.getElementById('promptLabBtn');
    btn.textContent = '⏳ Analyzing...';
    btn.disabled = true;
    el.innerHTML = '<div style="text-align:center;padding:20px;color:var(--text-muted);font-size:12px">Running prompt through intent classification and RAG retrieval...</div>';

    try {
        const res = await fetch(`${API_BASE}/api/prompt-lab`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        const data = await res.json();

        if (data.error) {
            el.innerHTML = `<div style="color:var(--accent-red);font-size:12px">Error: ${data.error}</div>`;
            btn.textContent = 'Analyze';
            btn.disabled = false;
            return;
        }

        const stageColors = ['var(--accent-orange)', 'var(--accent-green)', 'var(--accent)'];

        el.innerHTML = data.stages.map((stage, i) => {
            return `
            <div class="rag-stage" style="animation:stepSlide 0.3s ease ${i*0.1}s both">
                <div class="rag-stage-title" style="color:${stageColors[i]};font-size:13px">
                    ${stage.name}
                </div>
                <div style="font-size:11px;color:var(--text-secondary);line-height:1.65;margin-bottom:10px">
                    ${stage.description}
                </div>

                ${stage.system_prompt ? `
                <div style="margin-bottom:8px">
                    <div style="font-size:10px;font-weight:600;color:var(--accent-cyan);text-transform:uppercase;margin-bottom:4px">System Prompt</div>
                    <pre style="background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-secondary);white-space:pre-wrap;max-height:160px;overflow-y:auto;user-select:text">${escHtml(stage.system_prompt)}</pre>
                </div>` : ''}

                ${stage.assembled_user_prompt ? `
                <div style="margin-bottom:8px">
                    <div style="font-size:10px;font-weight:600;color:var(--accent-green);text-transform:uppercase;margin-bottom:4px">Assembled Prompt → LLM</div>
                    <pre style="background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-secondary);white-space:pre-wrap;max-height:160px;overflow-y:auto;user-select:text">${escHtml(stage.assembled_user_prompt)}</pre>
                </div>` : ''}

                ${stage.context_text ? `
                <div style="margin-bottom:8px">
                    <div style="font-size:10px;font-weight:600;color:var(--accent-blue);text-transform:uppercase;margin-bottom:4px">RAG Context (${stage.context_items} items${stage.top_similarity ? ', top sim: ' + (stage.top_similarity * 100).toFixed(1) + '%' : ''})</div>
                    <pre style="background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-secondary);white-space:pre-wrap;max-height:120px;overflow-y:auto;user-select:text">${escHtml(stage.context_text)}</pre>
                </div>` : ''}

                ${stage.result ? `
                <div class="pipeline-details">
                    ${Object.entries(stage.result).map(([k,v]) => `
                        <div class="detail-row"><span class="detail-key">${k.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</span>
                        <span class="detail-value">${fmtVal(v)}</span></div>`).join('')}
                </div>` : ''}
            </div>`;
        }).join('') + `
        <div class="pipeline-details" style="margin-top:8px">
            <div class="detail-row"><span class="detail-key">Classified Intent</span><span class="detail-value">${data.intent}</span></div>
            <div class="detail-row"><span class="detail-key">Total Prompt Length</span><span class="detail-value">${data.total_prompt_chars.toLocaleString()} chars</span></div>
        </div>`;

    } catch (e) {
        el.innerHTML = `<div style="color:var(--accent-red);font-size:12px">Error: ${e.message}</div>`;
    }

    btn.textContent = 'Analyze';
    btn.disabled = false;
}

function escHtml(s) {
    return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ═══════════════════════════════════════════════════════════════
//  TIER 2: MODEL COMPARISON BENCHMARK
// ═══════════════════════════════════════════════════════════════

async function runModelComparison() {
    const input = document.getElementById('modelCompareInput');
    const message = input.value.trim();
    if (!message) return;

    const el = document.getElementById('modelCompareResults');
    const btn = document.getElementById('modelCompareBtn');
    btn.textContent = '⏳ Running...';
    btn.disabled = true;
    el.innerHTML = '<div style="text-align:center;padding:20px;color:var(--text-muted);font-size:12px">Sending same prompt to all installed models...<br>This may take a minute.</div>';

    try {
        const res = await fetch(`${API_BASE}/api/models/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        const data = await res.json();

        if (data.error) {
            el.innerHTML = `<div style="color:var(--accent-red);font-size:12px">Error: ${data.error}</div>`;
            btn.textContent = '⚡ Run';
            btn.disabled = false;
            return;
        }

        const maxLatency = Math.max(...data.results.map(r => r.latency_ms), 1);
        const maxTps = Math.max(...data.results.map(r => r.tokens_per_second), 1);

        el.innerHTML = `
        <div style="font-size:11px;color:var(--text-muted);margin-bottom:10px">Tested ${data.models_tested} models with: "${data.prompt.slice(0,60)}..."</div>
        ` + data.results.map((r, i) => {
            const latencyPct = (r.latency_ms / maxLatency) * 100;
            const tpsPct = (r.tokens_per_second / maxTps) * 100;
            const modelColors = ['var(--accent)', 'var(--accent-orange)', 'var(--accent-cyan)', 'var(--accent-green)'];
            const color = modelColors[i] || 'var(--accent)';

            return `
            <div class="rag-stage" style="animation:stepSlide 0.3s ease ${i*0.15}s both">
                <div class="rag-stage-title" style="color:${color};font-size:13px;margin-bottom:8px">
                    🤖 ${r.model}
                    <span style="font-size:10px;color:${r.status === 'success' ? 'var(--accent-green)' : 'var(--accent-red)'};font-weight:400;margin-left:auto">
                        ${r.status}
                    </span>
                </div>

                <!-- Latency Bar -->
                <div style="margin-bottom:6px">
                    <div style="display:flex;justify-content:space-between;font-size:10px;margin-bottom:3px">
                        <span style="color:var(--text-muted)">Latency</span>
                        <span style="color:${color};font-family:'JetBrains Mono',monospace">${(r.latency_ms/1000).toFixed(1)}s</span>
                    </div>
                    <div class="rag-similarity-bar"><div class="rag-similarity-fill" style="width:${latencyPct}%;background:${color}"></div></div>
                </div>

                <!-- Throughput Bar -->
                <div style="margin-bottom:6px">
                    <div style="display:flex;justify-content:space-between;font-size:10px;margin-bottom:3px">
                        <span style="color:var(--text-muted)">Throughput</span>
                        <span style="color:${color};font-family:'JetBrains Mono',monospace">${r.tokens_per_second} tok/s</span>
                    </div>
                    <div class="rag-similarity-bar"><div class="rag-similarity-fill" style="width:${tpsPct}%;background:${color}"></div></div>
                </div>

                <div class="pipeline-details" style="margin-top:6px">
                    <div class="detail-row"><span class="detail-key">Tokens Generated</span><span class="detail-value">${r.tokens_generated}</span></div>
                    <div class="detail-row"><span class="detail-key">Response Length</span><span class="detail-value">${r.response_length} chars</span></div>
                </div>

                ${r.response ? `
                <details style="margin-top:8px">
                    <summary style="font-size:10px;color:var(--text-muted);cursor:pointer">View Response</summary>
                    <pre style="background:rgba(0,0,0,0.3);border:1px solid var(--border);border-radius:var(--radius-sm);padding:10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-secondary);white-space:pre-wrap;max-height:120px;overflow-y:auto;margin-top:6px;user-select:text">${escHtml(r.response)}</pre>
                </details>` : ''}
            </div>`;
        }).join('');

    } catch (e) {
        el.innerHTML = `<div style="color:var(--accent-red);font-size:12px">Error: ${e.message}</div>`;
    }

    btn.disabled = false;
}

// ═══════════════════════════════════════════════════════════════
//  EVALUATION DASHBOARD
// ═══════════════════════════════════════════════════════════════

function openEvaluationModal() {
    document.getElementById('evalModal').style.display = 'flex';
}

function closeEvaluationModal() {
    document.getElementById('evalModal').style.display = 'none';
}

async function runEvaluation() {
    const content = document.getElementById('evalContent');
    const btn = document.getElementById('runEvalBtn');
    
    if (btn) btn.disabled = true;
    
    content.innerHTML = `
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height: 100%; min-height: 300px;">
            <div class="spinner"></div>
            <div style="color:var(--text-secondary); font-size:14px; margin-bottom: 8px;">Running System Evaluation...</div>
            <div style="color:var(--text-muted); font-size:12px;">This evaluates retrieval, routing, and coherence. It may take 15-30 seconds.</div>
        </div>
    `;

    try {
        const res = await fetch(`${API_BASE}/api/evaluation/run`);
        const data = await res.json();
        
        if (data.status === 'error') throw new Error(data.error);
        
        const rag = data.rag;
        const routing = data.routing;
        const coherence = data.coherence;
        
        content.innerHTML = `
            <div class="eval-grid">
                <!-- RAG Quality -->
                <div class="eval-card">
                    <h3>📚 Retrieval Quality (RAG)</h3>
                    <div style="color:var(--text-muted); font-size:12px; margin-bottom:16px;">Evaluated against held-out queries.</div>
                    <div class="eval-metric">
                        <span class="eval-metric-label">Queries Evaluated</span>
                        <span class="eval-metric-val">${rag.queries_evaluated}</span>
                    </div>
                    <div class="eval-metric">
                        <span class="eval-metric-label">Mean Reciprocal Rank (MRR)</span>
                        <span class="eval-metric-val">${rag.mrr.toFixed(3)}</span>
                    </div>
                    <div class="eval-metric">
                        <span class="eval-metric-label">Precision@3</span>
                        <span class="eval-metric-val">${rag.precision.toFixed(3)}</span>
                    </div>
                    <div class="eval-metric">
                        <span class="eval-metric-label">Avg Cosine Similarity</span>
                        <span class="eval-metric-val">${rag.avg_similarity.toFixed(4)}</span>
                    </div>
                </div>

                <!-- Routing Accuracy -->
                <div class="eval-card">
                    <h3>⚡ Routing Accuracy</h3>
                    <div style="color:var(--text-muted); font-size:12px; margin-bottom:16px;">Orchestrator intent classification performance.</div>
                    <div class="eval-metric">
                        <span class="eval-metric-label">Total Test Cases</span>
                        <span class="eval-metric-val">${routing.total}</span>
                    </div>
                    <div class="eval-metric">
                        <span class="eval-metric-label">Correct Classifications</span>
                        <span class="eval-metric-val" style="color:var(--accent-green)">${routing.correct}</span>
                    </div>
                    <div class="eval-metric">
                        <span class="eval-metric-label">Overall Accuracy</span>
                        <span class="eval-metric-val" style="color:var(--accent-green)">${(routing.accuracy * 100).toFixed(1)}%</span>
                    </div>
                </div>

                <!-- Coherence -->
                <div class="eval-card eval-card-full">
                    <h3>💬 Response Coherence (RAG vs Base)</h3>
                    <div style="color:var(--text-muted); font-size:12px; margin-bottom:16px;">
                        Qualitative evaluation of LLM output quality with and without retrieved context.
                        Average Judge Score: <span style="color:var(--accent); font-weight:bold;">${coherence.average_score.toFixed(1)}/10</span>
                    </div>
                    
                    ${coherence.results.map((r, i) => `
                        <div style="margin-top: 20px;">
                            <div style="font-size: 13px; font-weight: 600; margin-bottom: 12px; padding: 8px; background: rgba(0,0,0,0.2); border-radius: 4px;">
                                Query ${i+1}: "${r.query}" 
                                <span style="float:right; color:var(--accent-green);">Score: ${r.coherence_score}/10</span>
                            </div>
                            <div class="eval-coherence-grid">
                                <div class="eval-coherence-box">
                                    <h4>Base LLM (No Context)</h4>
                                    <div class="eval-coherence-text">${escHtml(r.base_response).replace(/\\n/g, '<br>')}</div>
                                </div>
                                <div class="eval-coherence-box" style="border-color: var(--accent-blue);">
                                    <h4 style="color:var(--accent-blue);">RAG-Augmented LLM</h4>
                                    <div class="eval-coherence-text">${escHtml(r.rag_response).replace(/\\n/g, '<br>')}</div>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <div style="text-align: right; margin-top: 10px;">
                <button class="btn" style="background:var(--accent);" onclick="runEvaluation()">🔄 Re-run Evaluation</button>
            </div>
        `;
        
    } catch (e) {
        content.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon" style="color:var(--accent-red);">❌</div>
                <div class="empty-state-text">Evaluation failed: ${e.message}</div>
                <button class="btn" style="margin-top:20px;" onclick="runEvaluation()">Retry</button>
            </div>
        `;
    }
}
