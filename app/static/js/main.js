const EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise'];
const EMOTION_COLORS = {
    angry: '#DC2626', disgust: '#16A34A', fear: '#9333EA',
    happy: '#CA8A04', neutral: '#6B7280', sad: '#2563EB', surprise: '#0891B2'
};
const EMOTION_EMOJI = {
    angry: '😠', disgust: '🤢', fear: '😨',
    happy: '😊', neutral: '😐', sad: '😢', surprise: '😲'
};

// ===== ТАБИ =====
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    });
});

// ===== ПІД-ТАБИ =====
document.querySelectorAll('.subtab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.subtab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.subtab-content').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('subtab-' + tab.dataset.subtab).classList.add('active');
    });
});

// ===== ФОТО =====
const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');
const previewImg = document.getElementById('previewImg');
const previewPlaceholder = document.getElementById('previewPlaceholder');
const analyzeBtn = document.getElementById('analyzeBtn');
const loader = document.getElementById('loader');
const errorMsg = document.getElementById('errorMsg');
const emptyState = document.getElementById('emptyState');
const resultsWrap = document.getElementById('resultsWrap');
const faceTabs = document.getElementById('faceTabs');
const resultContent = document.getElementById('resultContent');

let currentFile = null;

uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
});
fileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
        previewImg.style.display = 'block';
        previewPlaceholder.style.display = 'none';
    };
    reader.readAsDataURL(file);
    analyzeBtn.disabled = false;
    emptyState.style.display = 'none';
    resultsWrap.style.display = 'none';
    errorMsg.style.display = 'none';
}

analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    loader.style.display = 'block';
    errorMsg.style.display = 'none';
    resultsWrap.style.display = 'none';
    emptyState.style.display = 'none';
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('image', currentFile);

    try {
        const res = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Помилка сервера');
        previewImg.src = data.image;
        renderResults(data.results, 0);
    } catch (err) {
        errorMsg.textContent = '⚠️ ' + err.message;
        errorMsg.style.display = 'block';
    } finally {
        loader.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

function renderResults(results, activeIdx) {
    resultsWrap.style.display = 'block';
    faceTabs.innerHTML = '';
    if (results.length > 1) {
        results.forEach((r, i) => {
            const tab = document.createElement('button');
            tab.className = 'face-tab' + (i === activeIdx ? ' active' : '');
            tab.textContent = `Обличчя ${i+1} ${r.emoji}`;
            tab.onclick = () => renderResults(results, i);
            faceTabs.appendChild(tab);
        });
    }
    const r = results[activeIdx];
    const color = EMOTION_COLORS[r.emotion] || '#818cf8';
    const sorted = Object.entries(r.all_probs).sort((a,b) => b[1] - a[1]);
    resultContent.innerHTML = `
        <div class="result-card" style="--emotion-color:${color}">
            <div class="result-top">
                <div class="result-emoji">${r.emoji}</div>
                <div>
                    <div class="result-label">${r.emotion}</div>
                    <div class="result-conf">Впевненість: ${r.confidence}%</div>
                </div>
            </div>
            <div class="bars">
                ${sorted.map(([em, pct]) => `
                    <div class="bar-row">
                        <span class="bar-label">${em}</span>
                        <div class="bar-track">
                            <div class="bar-fill" style="width:${pct}%;background:${EMOTION_COLORS[em]}"></div>
                        </div>
                        <span class="bar-pct">${pct}%</span>
                    </div>
                `).join('')}
            </div>
        </div>`;
}

// ===== КАМЕРА =====
const startCamBtn = document.getElementById('startCamBtn');
const stopCamBtn = document.getElementById('stopCamBtn');
const cameraFeed = document.getElementById('cameraFeed');
const camPlaceholder = document.getElementById('camPlaceholder');
const emotionEmoji = document.getElementById('emotionEmoji');
const emotionName = document.getElementById('emotionName');
const emotionConf = document.getElementById('emotionConf');
const liveBars = document.getElementById('liveBars');
const statDominant = document.getElementById('statDominant');
const statFrames = document.getElementById('statFrames');

let pollInterval = null;
let sessionCounts = {};
let totalFrames = 0;
EMOTIONS.forEach(e => sessionCounts[e] = 0);

startCamBtn.addEventListener('click', async () => {
    await fetch('/camera/start', { method: 'POST' });
    cameraFeed.src = '/video_feed';
    cameraFeed.style.display = 'block';
    camPlaceholder.style.display = 'none';
    startCamBtn.style.display = 'none';
    stopCamBtn.style.display = 'block';
    pollInterval = setInterval(pollEmotion, 500);
});

stopCamBtn.addEventListener('click', async () => {
    await fetch('/camera/stop', { method: 'POST' });
    cameraFeed.src = '';
    cameraFeed.style.display = 'none';
    camPlaceholder.style.display = 'block';
    startCamBtn.style.display = 'block';
    stopCamBtn.style.display = 'none';
    clearInterval(pollInterval);
});

async function pollEmotion() {
    try {
        const res = await fetch('/current_emotion');
        if (!res.ok) return;
        const data = await res.json();
        if (!data.emotion) return;

        emotionEmoji.textContent = EMOTION_EMOJI[data.emotion] || '😐';
        emotionName.textContent = data.emotion;
        emotionName.style.color = EMOTION_COLORS[data.emotion];
        emotionConf.textContent = `Впевненість: ${data.confidence.toFixed(1)}%`;

        const rows = liveBars.querySelectorAll('.bar-row');
        EMOTIONS.forEach((em, i) => {
            const pct = (data.all_probs[em] || 0).toFixed(1);
            rows[i].querySelector('.bar-fill').style.width = pct + '%';
            rows[i].querySelector('.bar-pct').textContent = pct + '%';
        });

        sessionCounts[data.emotion]++;
        totalFrames++;
        statFrames.textContent = totalFrames;
        const dominant = Object.entries(sessionCounts).sort((a,b)=>b[1]-a[1])[0];
        const pct = (dominant[1]/totalFrames*100).toFixed(0);
        statDominant.textContent = `${dominant[0]} ${pct}%`;

    } catch(e) {}
}

// ===== ВІДЕО =====
const videoInput = document.getElementById('videoInput');
const videoUploadZone = document.getElementById('videoUploadZone');
const videoPlayer = document.getElementById('videoPlayer');
const videoPlayerWrap = document.getElementById('videoPlayerWrap');
const videoControls = document.getElementById('videoControls');
const analyzeVideoBtn = document.getElementById('analyzeVideoBtn');
const stopVideoBtn = document.getElementById('stopVideoBtn');
const progressWrap = document.getElementById('progressWrap');
const progressFill = document.getElementById('progressFill');
const progressTime = document.getElementById('progressTime');
const progressPct = document.getElementById('progressPct');
const videoEmoji = document.getElementById('videoEmoji');
const videoEmotionName = document.getElementById('videoEmotionName');
const videoConf = document.getElementById('videoConf');
const vstatDominant = document.getElementById('vstatDominant');
const vstatFrames = document.getElementById('vstatFrames');
const vstatFaces = document.getElementById('vstatFaces');
const emotionSummary = document.getElementById('emotionSummary');
const timelineTrack = document.getElementById('timelineTrack');
const timelineCursor = document.getElementById('timelineCursor');
const timelineLabels = document.getElementById('timelineLabels');
const snapshotsGrid = document.getElementById('snapshotsGrid');
const snapshotCount = document.getElementById('snapshotCount');

let videoAnalyzing = false;
let videoAnimFrame = null;
let vTotalFrames = 0;
let vTotalFaces = 0;
let vEmotionCounts = {};
EMOTIONS.forEach(e => vEmotionCounts[e] = 0);

// timeline дані
let timelineData = [];      // [{time, emotion, color, duration}]
let lastEmotion = null;
let lastEmotionTime = 0;
let lastEmotionStart = 0;

// знімки
let snapshots = [];
let lastSnapshotEmotion = null;
const SNAPSHOT_THRESHOLD = 0.7;  // confidence щоб зробити знімок
const MIN_SNAPSHOT_INTERVAL = 3; // мінімум 3 секунди між знімками
let lastSnapshotTime = -99;

const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

// canvas для знімків (видимий розмір)
const snapCanvas = document.createElement('canvas');
const snapCtx = snapCanvas.getContext('2d');

videoUploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    videoUploadZone.classList.add('drag-over');
});
videoUploadZone.addEventListener('dragleave', () => videoUploadZone.classList.remove('drag-over'));
videoUploadZone.addEventListener('drop', e => {
    e.preventDefault();
    videoUploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) loadVideo(file);
});
videoInput.addEventListener('change', e => {
    if (e.target.files[0]) loadVideo(e.target.files[0]);
});

function loadVideo(file) {
    const url = URL.createObjectURL(file);
    videoPlayer.src = url;
    videoPlayerWrap.style.display = 'block';
    videoControls.style.display = 'flex';
    resetVideoData();
}

function resetVideoData() {
    vTotalFrames = 0;
    vTotalFaces = 0;
    EMOTIONS.forEach(e => vEmotionCounts[e] = 0);
    timelineData = [];
    snapshots = [];
    lastEmotion = null;
    lastSnapshotEmotion = null;
    lastSnapshotTime = -99;
    timelineTrack.innerHTML = '';
    timelineLabels.innerHTML = '';
    snapshotsGrid.innerHTML = '<div class="snapshots-empty">Знімки з\'являться коли емоція різко зміниться</div>';
    snapshotCount.textContent = '';
    updateVideoStats();
}

analyzeVideoBtn.addEventListener('click', () => {
    videoAnalyzing = true;
    videoPlayer.play();
    analyzeVideoBtn.style.display = 'none';
    stopVideoBtn.style.display = 'block';
    progressWrap.style.display = 'block';
    resetVideoData();
    lastAnalyzedTime = -1;
    analyzeLoop();
});

stopVideoBtn.addEventListener('click', () => {
    videoAnalyzing = false;
    videoPlayer.pause();
    analyzeVideoBtn.style.display = 'block';
    stopVideoBtn.style.display = 'none';
    if (videoAnimFrame) cancelAnimationFrame(videoAnimFrame);
    finalizeTimeline();
    updateSummary();
});

videoPlayer.addEventListener('ended', () => {
    videoAnalyzing = false;
    analyzeVideoBtn.style.display = 'block';
    stopVideoBtn.style.display = 'none';
    finalizeTimeline();
    updateSummary();
});

// клік по timeline — перемотує відео
document.getElementById('timelineWrap').addEventListener('click', e => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    videoPlayer.currentTime = pct * videoPlayer.duration;
});

let lastAnalyzedTime = -1;
let isProcessing = false;
const ANALYZE_INTERVAL = 0.3;

async function analyzeLoop() {
    if (!videoAnalyzing) return;

    const currentTime = videoPlayer.currentTime;
    const duration = videoPlayer.duration || 1;
    const pct = (currentTime / duration * 100).toFixed(1);

    progressFill.style.width = pct + '%';
    progressPct.textContent = pct + '%';
    progressTime.textContent = `${formatTime(currentTime)} / ${formatTime(duration)}`;

    // курсор на timeline
    timelineCursor.style.left = pct + '%';

    if (!isProcessing && currentTime - lastAnalyzedTime >= ANALYZE_INTERVAL) {
        lastAnalyzedTime = currentTime;
        isProcessing = true;

        canvas.width = videoPlayer.videoWidth;
        canvas.height = videoPlayer.videoHeight;
        ctx.drawImage(videoPlayer, 0, 0);
        const frameData = canvas.toDataURL('image/jpeg', 0.6);

        try {
            const res = await fetch('/analyze_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame: frameData })
            });
            const data = await res.json();

            if (data.emotion) {
                vTotalFrames++;
                vTotalFaces += data.faces;
                vEmotionCounts[data.emotion]++;

                // UI
                videoEmoji.textContent = data.emoji;
                videoEmotionName.textContent = data.emotion;
                videoEmotionName.style.color = data.color;
                videoConf.textContent = `Впевненість: ${data.confidence}%`;

                EMOTIONS.forEach(em => {
                    const p = data.all_probs[em] || 0;
                    document.getElementById(`vbar-${em}`).style.width = p + '%';
                    document.getElementById(`vpct-${em}`).textContent = p + '%';
                });

                updateVideoStats();

                // timeline
                updateTimeline(data.emotion, data.color, currentTime, duration);

                // знімок якщо змінилась емоція і висока впевненість
                const emotionChanged = data.emotion !== lastSnapshotEmotion;
                const highConf = data.confidence >= SNAPSHOT_THRESHOLD * 100;
                const enoughTime = currentTime - lastSnapshotTime >= MIN_SNAPSHOT_INTERVAL;

                if (emotionChanged && highConf && enoughTime) {
                    takeSnapshot(data.emotion, data.color, data.emoji, data.confidence, currentTime);
                    lastSnapshotEmotion = data.emotion;
                    lastSnapshotTime = currentTime;
                }

                lastEmotion = data.emotion;
            }
        } catch(e) {}

        isProcessing = false;
    }

    videoAnimFrame = requestAnimationFrame(analyzeLoop);
}

function updateTimeline(emotion, color, currentTime, duration) {
    if (emotion !== lastEmotion) {
        // зберігаємо попередній сегмент
        if (lastEmotion !== null) {
            timelineData.push({
                emotion: lastEmotion,
                color: EMOTION_COLORS[lastEmotion],
                start: lastEmotionStart / duration,
                end: currentTime / duration
            });
        }
        lastEmotionStart = currentTime;
    }

    // перемальовуємо timeline
    renderTimeline(currentTime, duration);
}

function renderTimeline(currentTime, duration) {
    const segments = [...timelineData];
    if (lastEmotion) {
        segments.push({
            emotion: lastEmotion,
            color: EMOTION_COLORS[lastEmotion],
            start: lastEmotionStart / duration,
            end: currentTime / duration
        });
    }

    if (segments.length === 0) return;

    timelineTrack.innerHTML = segments.map(s => {
        const width = Math.max(0.1, (s.end - s.start) * 100);
        const startTime = formatTime(s.start * duration);
        const endTime = formatTime(s.end * duration);
        return `<div class="tl-segment"
                     style="width:${width}%;background:${s.color}"
                     title="${s.emotion}  ${startTime} → ${endTime}">
                </div>`;
    }).join('');

    // мітки часу — рівномірно
    const steps = 4;
    timelineLabels.innerHTML = Array.from({length: steps + 1}, (_, i) => {
        return `<span class="tl-label">${formatTime(i / steps * duration)}</span>`;
    }).join('');
}

function finalizeTimeline() {
    const duration = videoPlayer.duration || 1;
    if (lastEmotion) {
        timelineData.push({
            emotion: lastEmotion,
            color: EMOTION_COLORS[lastEmotion],
            start: lastEmotionStart / duration,
            end: videoPlayer.currentTime / duration
        });
        lastEmotion = null;
    }
    renderTimeline(videoPlayer.currentTime, duration);
}

function takeSnapshot(emotion, color, emoji, confidence, time) {
    // знімок з canvas
    snapCanvas.width = Math.min(canvas.width, 640);
    snapCanvas.height = Math.min(canvas.height, 480);
    snapCtx.drawImage(canvas, 0, 0, snapCanvas.width, snapCanvas.height);
    const imgData = snapCanvas.toDataURL('image/jpeg', 0.8);

    snapshots.push({ emotion, color, emoji, confidence, time, imgData });

    // оновлюємо grid
    const empty = snapshotsGrid.querySelector('.snapshots-empty');
    if (empty) empty.remove();

    const item = document.createElement('div');
    item.className = 'snapshot-item';
    item.style.setProperty('--snap-color', color);
    item.innerHTML = `
        <img src="${imgData}" alt="${emotion}">
        <div class="snapshot-info">
            <div>
                <div class="snapshot-emotion">${emoji} ${emotion}</div>
                <div class="snapshot-conf">${confidence.toFixed(1)}%</div>
            </div>
            <div class="snapshot-time">${formatTime(time)}</div>
        </div>
    `;

    // клік на знімок — перемотує відео
    item.addEventListener('click', () => {
        videoPlayer.currentTime = time;
    });

    snapshotsGrid.insertBefore(item, snapshotsGrid.firstChild);
    snapshotCount.textContent = `(${snapshots.length})`;
}

function updateVideoStats() {
    vstatFrames.textContent = vTotalFrames;
    vstatFaces.textContent = vTotalFaces;
    if (vTotalFrames > 0) {
        const dominant = Object.entries(vEmotionCounts).sort((a,b)=>b[1]-a[1])[0];
        const pct = (dominant[1]/vTotalFrames*100).toFixed(0);
        vstatDominant.textContent = `${dominant[0]} (${pct}%)`;
    }
}

function updateSummary() {
    if (vTotalFrames === 0) return;
    const sorted = Object.entries(vEmotionCounts).sort((a,b)=>b[1]-a[1]);
    emotionSummary.innerHTML = sorted.map(([em, count]) => {
        const pct = (count/vTotalFrames*100).toFixed(1);
        return `
            <div class="emotion-summary-item">
                <span class="esi-emoji">${EMOTION_EMOJI[em]}</span>
                <span class="esi-name">${em}</span>
                <div class="esi-bar-wrap">
                    <div class="esi-bar" style="width:${pct}%;background:${EMOTION_COLORS[em]}"></div>
                </div>
                <span class="esi-pct">${pct}%</span>
            </div>`;
    }).join('');
}

function formatTime(s) {
    const m = Math.floor(s/60);
    const sec = Math.floor(s%60).toString().padStart(2,'0');
    return `${m}:${sec}`;
}