// Global Web App State
const state = {
    connected: false,
    history: [],
    currentPredicting: false,
    activeViewers: new Map(), // maps output-card-id -> 3Dmol viewer instance
    activeViewerStyles: new Map() // maps output-card-id -> style ('cartoon', 'stick', 'sphere')
};

// Preset Configurations
const PRESETS = {
    'preset-kinase': {
        sequence: 'MTEYKLVVVGAGGVGKSALTI',
        ligand: 'C,C,O,N',
        seeds: 2
    },
    'preset-gpcr': {
        sequence: 'MTRLIPSLVGLAAVSL',
        ligand: 'C,C,C,O,N,S',
        seeds: 3
    },
    'preset-peptide': {
        sequence: 'LQPSETDFGEK',
        ligand: 'C,O,O',
        seeds: 2
    }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    checkBackendHealth();
    loadHistoryFromStorage();
    setupEventListeners();
});

// Check Backend Connection Status
async function checkBackendHealth() {
    const statusText = document.getElementById('backend-status-text');
    const statusDot = document.querySelector('.status-dot');
    
    statusDot.className = 'status-dot connecting';
    statusText.textContent = 'Connecting to IsoDDE...';
    
    try {
        const response = await fetch('/api/health');
        if (response.ok) {
            const data = await response.json();
            state.connected = true;
            statusDot.className = 'status-dot healthy';
            statusText.textContent = `Connected: ${data.model || 'IsoDDE Engine'}`;
        } else {
            throw new Error('Unhealthy status');
        }
    } catch (e) {
        state.connected = false;
        statusDot.className = 'status-dot';
        statusText.textContent = 'Disconnected from Engine';
    }
}

// Event Listeners Configuration
function setupEventListeners() {
    // Range slider value update
    const slider = document.getElementById('settings-seeds');
    const sliderVal = document.getElementById('seeds-val');
    slider.addEventListener('input', (e) => {
        sliderVal.textContent = e.target.value;
    });

    // Form submission
    const form = document.getElementById('prediction-form');
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        handleFormSubmit();
    });

    // Chat Message Text Input Enter Key
    const chatInput = document.getElementById('chat-message-input');
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleFormSubmit();
        }
    });
}

// Sidebar Toggle (Mobile Support)
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('hidden');
}

// Load Preset Data
function loadPreset(presetId) {
    const preset = PRESETS[presetId];
    if (!preset) return;
    
    document.getElementById('protein-seq').value = preset.sequence;
    document.getElementById('ligand-atoms').value = preset.ligand;
    document.getElementById('settings-seeds').value = preset.seeds;
    document.getElementById('seeds-val').textContent = preset.seeds;
    
    addAssistantMessage(`Loaded settings for: ${presetId.replace('preset-', '').toUpperCase()}. Ready to predict.`);
}

// Quick suggestions
function submitQuickQuery(queryText) {
    // Try to parse out sequence and elements
    let seq = 'MTEYKLVVVGAGGVGKSALTI';
    let lig = 'C,C,O,N';
    
    if (queryText.includes('sequence') || queryText.includes('protein')) {
        const seqMatch = queryText.match(/sequence\s+([A-Za-z]+)/i) || queryText.match(/protein\s+([A-Za-z]+)/i);
        if (seqMatch) seq = seqMatch[1];
    }
    
    if (queryText.includes('elements') || queryText.includes('ligand')) {
        const ligMatch = queryText.match(/elements\s+([A-Za-z,]+)/i) || queryText.match(/ligand\s+([A-Za-z,]+)/i);
        if (ligMatch) lig = ligMatch[1];
    }

    document.getElementById('protein-seq').value = seq.trim().toUpperCase();
    document.getElementById('ligand-atoms').value = lig.trim();
    
    appendMessage(queryText, 'user');
    runPredictionFlow(seq, lig);
}

// Main Submit Handlers
function handleFormSubmit() {
    if (state.currentPredicting) return;
    
    const seqInput = document.getElementById('protein-seq');
    const ligInput = document.getElementById('ligand-atoms');
    const chatMsgInput = document.getElementById('chat-message-input');
    
    const proteinSeq = seqInput.value.trim().toUpperCase();
    const ligandVal = ligInput.value.trim();
    const chatMsg = chatMsgInput.value.trim();
    
    if (!proteinSeq) {
        addAssistantMessage("Error: A protein sequence is required to run prediction.");
        return;
    }
    
    let userMsg = chatMsg;
    if (!userMsg) {
        userMsg = `Predict co-folding for sequence of length ${proteinSeq.length}`;
        if (ligandVal) userMsg += ` and ligand atoms [${ligandVal}]`;
    }
    
    appendMessage(userMsg, 'user');
    chatMsgInput.value = '';
    
    runPredictionFlow(proteinSeq, ligandVal);
}

// Run end-to-end prediction process
async function runPredictionFlow(proteinSeq, ligandVal) {
    state.currentPredicting = true;
    toggleFormInputs(false);
    
    // Add Assistant Typing/Progress Message
    const cardId = 'pred-' + Date.now();
    const messageId = appendLoadingMessage(cardId);
    const progressEl = document.getElementById(`${messageId}-log`);
    
    const seeds = parseInt(document.getElementById('settings-seeds').value);
    
    // Step-by-step progress animation logs
    const stages = [
        "Initializing tokenizers and embedding inputs...",
        "Running MSA Module (processing single/pair information flow)...",
        "Iterating through Pairformer Triangular Attention blocks...",
        "Sampling 3D structural coordinates via Diffusion Head...",
        "Executing Pocket Identification & Affinity Heads...",
        "Consolidating predicted complex structures..."
    ];
    
    let stageIdx = 0;
    const progressTimer = setInterval(() => {
        if (stageIdx < stages.length) {
            appendProgressLog(progressEl, stages[stageIdx]);
            stageIdx++;
        }
    }, 1500);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                protein_sequence: proteinSeq,
                ligand: ligandVal,
                num_seeds: seeds
            })
        });
        
        clearInterval(progressTimer);
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Server error running prediction");
        }
        
        const data = await response.json();
        
        // Remove loading bubble
        removeMessage(messageId);
        
        // Render Output Card
        appendPredictionResultCard(cardId, data, proteinSeq, ligandVal);
        
        // Save to Local History
        saveRunToHistory(cardId, proteinSeq, ligandVal, data);
        
    } catch (err) {
        clearInterval(progressTimer);
        removeMessage(messageId);
        addAssistantMessage(`❌ Prediction failed: ${err.message}`);
    } finally {
        state.currentPredicting = false;
        toggleFormInputs(true);
    }
}

// View history from LocalStorage
function loadHistoryFromStorage() {
    const stored = localStorage.getItem('isodde_history');
    if (stored) {
        try {
            state.history = JSON.parse(stored);
            updateHistorySidebar();
        } catch (e) {
            state.history = [];
        }
    }
}

// Save history
function saveRunToHistory(cardId, seq, ligand, data) {
    const record = {
        id: cardId,
        seq: seq,
        ligand: ligand,
        timestamp: new Date().toLocaleTimeString(),
        metrics: {
            pLDDT: data.pLDDT,
            ptm: data.ptm,
            affinity: data.binding_affinity_pkd,
            pockets: data.pockets.length
        },
        data: data
    };
    
    state.history.unshift(record);
    if (state.history.length > 20) state.history.pop();
    
    localStorage.setItem('isodde_history', JSON.stringify(state.history));
    updateHistorySidebar();
}

function updateHistorySidebar() {
    const list = document.getElementById('history-list');
    if (state.history.length === 0) {
        list.innerHTML = '<div class="empty-history">No recent runs</div>';
        return;
    }
    
    list.innerHTML = state.history.map(item => `
        <button class="history-item" onclick="reloadHistoryItem('${item.id}')">
            <span class="history-icon"><i class="fa-solid fa-file-waveform"></i></span>
            <div class="history-info">
                <div class="history-name">${item.seq.substring(0, 10)}... + ${item.ligand || 'None'}</div>
                <div class="history-meta">pLDDT: ${(item.metrics.pLDDT * 100).toFixed(1)}% | ${item.timestamp}</div>
            </div>
        </button>
    `).join('');
}

// Reload historical run card into conversation stream
function reloadHistoryItem(id) {
    const item = state.history.find(x => x.id === id);
    if (!item) return;
    
    appendMessage(`Reload prediction for ${item.seq.substring(0, 8)}...`, 'user');
    appendPredictionResultCard(item.id + '-reload', item.data, item.seq, item.ligand);
}

// Enable/Disable input fields
function toggleFormInputs(enabled) {
    document.getElementById('protein-seq').disabled = !enabled;
    document.getElementById('ligand-atoms').disabled = !enabled;
    document.getElementById('settings-seeds').disabled = !enabled;
    document.getElementById('btn-submit').disabled = !enabled;
    
    const btnText = document.getElementById('btn-text');
    if (enabled) {
        btnText.textContent = "Run Prediction";
    } else {
        btnText.textContent = "Predicting...";
    }
}

// Append Chat message
function appendMessage(text, sender) {
    const container = document.getElementById('chat-container');
    const msg = document.createElement('div');
    msg.className = `message ${sender}`;
    
    const avatarIcon = sender === 'user' ? 'fa-user' : 'fa-robot';
    
    msg.innerHTML = `
        <div class="message-avatar">
            <i class="fa-solid ${avatarIcon}"></i>
        </div>
        <div class="message-bubble">
            <p>${text}</p>
        </div>
    `;
    
    container.appendChild(msg);
    scrollToBottom();
    return msg;
}

// Add simple helper message
function addAssistantMessage(text) {
    appendMessage(text, 'assistant');
}

// Loading log messaging
function appendLoadingMessage(cardId) {
    const container = document.getElementById('chat-container');
    const msgId = 'loading-' + cardId;
    const msg = document.createElement('div');
    msg.className = 'message assistant loading-bubble';
    msg.id = msgId;
    
    msg.innerHTML = `
        <div class="message-avatar">
            <i class="fa-solid fa-robot"></i>
        </div>
        <div class="message-bubble">
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
            <p style="font-size: 12px; margin-top: 4px; color: var(--text-secondary);">IsoDDE Model is executing...</p>
            <div class="loading-log" id="${msgId}-log">Model Loading...</div>
        </div>
    `;
    
    container.appendChild(msg);
    scrollToBottom();
    return msgId;
}

function appendProgressLog(el, text) {
    if (el) {
        el.textContent += `\n→ ${text}`;
    }
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    const container = document.getElementById('chat-container');
    container.scrollTop = container.scrollHeight;
}

// Switch prediction output card tabs
function switchTab(btn, tabId) {
    const card = btn.closest('.prediction-card-inner');
    
    // Deactivate current tabs
    card.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    card.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    
    // Activate clicked tab
    btn.classList.add('active');
    
    const activePane = card.querySelector(`#${tabId}`);
    if (activePane) activePane.classList.add('active');
}

// Main Card Rendering logic
function appendPredictionResultCard(cardId, data, seq, ligand) {
    const container = document.getElementById('chat-container');
    const msg = document.createElement('div');
    msg.className = 'message assistant result-message';
    msg.style.maxWidth = '90%';
    
    msg.innerHTML = `
        <div class="message-avatar">
            <i class="fa-solid fa-robot"></i>
        </div>
        <div class="message-bubble" style="padding: 4px; border: none; background: transparent;">
            <div class="prediction-card-inner" id="${cardId}">
                <!-- Rendered dynamically from template -->
            </div>
        </div>
    `;
    
    container.appendChild(msg);
    
    // Copy template content
    const template = document.getElementById('prediction-card-template');
    const clone = template.content.cloneNode(true);
    
    const cardRoot = msg.querySelector('.prediction-card-inner');
    cardRoot.appendChild(clone);
    
    // Modify sub IDs in tabs to avoid overlapping viewers
    const viewerId = `mol-viewer-${cardId}`;
    cardRoot.querySelector('#mol-viewer-container').id = viewerId;
    
    const tab3d = cardRoot.querySelector('#tab-3d');
    const tabResidues = cardRoot.querySelector('#tab-residues');
    const tabPockets = cardRoot.querySelector('#tab-pockets');
    const tabContacts = cardRoot.querySelector('#tab-contacts');
    
    tab3d.id = `tab-3d-${cardId}`;
    tabResidues.id = `tab-residues-${cardId}`;
    tabPockets.id = `tab-pockets-${cardId}`;
    tabContacts.id = `tab-contacts-${cardId}`;
    
    // Update tab-buttons triggers
    const tabBtns = cardRoot.querySelectorAll('.tab-btn');
    tabBtns[0].setAttribute('onclick', `switchTab(this, 'tab-3d-${cardId}')`);
    tabBtns[1].setAttribute('onclick', `switchTab(this, 'tab-residues-${cardId}')`);
    tabBtns[2].setAttribute('onclick', `switchTab(this, 'tab-pockets-${cardId}')`);
    tabBtns[3].setAttribute('onclick', `switchTab(this, 'tab-contacts-${cardId}')`);
    
    // Populate simple metrics
    cardRoot.querySelector('.plddt-value').textContent = `${(data.pLDDT * 100).toFixed(1)}%`;
    cardRoot.querySelector('.plddt-fill').style.width = `${data.pLDDT * 100}%`;
    
    cardRoot.querySelector('.ptm-value').textContent = `${data.ptm.toFixed(2)}`;
    cardRoot.querySelector('.ptm-fill').style.width = `${data.ptm * 100}%`;
    
    cardRoot.querySelector('.affinity-value').textContent = `${data.binding_affinity_pkd.toFixed(2)} pKd`;
    cardRoot.querySelector('.pockets-count-value').textContent = `${data.pockets.length}`;
    
    const qr = data.quality_report || {};
    const totalViolations = (qr.bond_violations || 0) + (qr.clashes || 0);
    const qualityEl = cardRoot.querySelector('.quality-value');
    if (qualityEl) {
        qualityEl.textContent = totalViolations === 0 ? 'Passed' : `${totalViolations} Devs`;
        qualityEl.parentNode.title = `Bond Length Violations: ${qr.bond_violations || 0}\nSteric Clashes: ${qr.clashes || 0}`;
        if (totalViolations > 0) {
            qualityEl.style.color = '#ef4444'; // Red color for errors
        } else {
            qualityEl.style.color = '#10b981'; // Green color for Passed
        }
    }
    
    // Store PDB content as attributes
    cardRoot.setAttribute('data-pdb', data.pdb_content);
    cardRoot.setAttribute('data-seq-len', data.protein_length);
    cardRoot.setAttribute('data-lig-len', data.ligand_length);
    cardRoot.setAttribute('data-plddt', JSON.stringify(data.plddt_list || []));
    cardRoot.setAttribute('data-ss', JSON.stringify(data.secondary_structure || []));
    cardRoot.setAttribute('data-rsa', JSON.stringify(data.solvent_accessibility || []));
    
    if (data.sdf_content) {
        cardRoot.setAttribute('data-sdf', data.sdf_content);
        const sdfBtn = cardRoot.querySelector('.btn-download-sdf');
        if (sdfBtn) sdfBtn.style.display = 'inline-block';
    }
    
    // Trigger structural and contact renders
    init3DViewer(cardId, viewerId, data.pdb_content, data.protein_length, data.ligand_length);
    renderResidues(cardRoot, seq, data);
    renderPocketsTable(cardRoot, data.pockets, cardId);
    renderContactHeatmaps(cardRoot, data);
    
    scrollToBottom();
}

// Render residue sequence & heads
function renderResidues(cardRoot, seq, data) {
    const grid = cardRoot.querySelector('.residues-grid');
    if (!grid) return;
    grid.innerHTML = '';
    
    const ss = data.secondary_structure || [];
    const rsa = data.solvent_accessibility || [];
    
    const ssLetters = ['H', 'E', 'C'];
    const ssNames = ['Helix', 'Beta Sheet', 'Coil/Loop'];
    const ssColors = ['#ff7675', '#ffeaa7', '#a29bfe'];
    
    for (let i = 0; i < seq.length; i++) {
        const residueChar = seq[i];
        const resSS = ss[i] !== undefined ? ss[i] : 2;
        const resRSA = rsa[i] !== undefined ? rsa[i] : 0.0;
        
        const resEl = document.createElement('div');
        resEl.className = 'residue-item';
        resEl.style.display = 'flex';
        resEl.style.flexDirection = 'column';
        resEl.style.alignItems = 'center';
        resEl.style.justifyContent = 'center';
        resEl.style.width = '38px';
        resEl.style.height = '48px';
        resEl.style.border = '1px solid var(--border-color)';
        resEl.style.borderRadius = '4px';
        resEl.style.fontSize = '0.8rem';
        resEl.style.position = 'relative';
        resEl.style.cursor = 'default';
        resEl.style.backgroundColor = ssColors[resSS] + '15';
        resEl.style.borderColor = ssColors[resSS];
        
        resEl.title = `Residue ${i + 1}: ${residueChar}\nSS: ${ssNames[resSS]} (${ssLetters[resSS]})\nRSA: ${resRSA.toFixed(2)}`;
        
        const charSpan = document.createElement('span');
        charSpan.textContent = residueChar;
        charSpan.style.fontWeight = 'bold';
        charSpan.style.color = 'var(--text-primary)';
        resEl.appendChild(charSpan);
        
        const ssSpan = document.createElement('span');
        ssSpan.textContent = ssLetters[resSS];
        ssSpan.style.fontSize = '0.65rem';
        ssSpan.style.color = ssColors[resSS];
        ssSpan.style.marginTop = '2px';
        resEl.appendChild(ssSpan);
        
        const rsaSpan = document.createElement('span');
        rsaSpan.textContent = resRSA.toFixed(1);
        rsaSpan.style.fontSize = '0.6rem';
        rsaSpan.style.color = 'var(--text-muted)';
        rsaSpan.style.marginTop = '1px';
        resEl.appendChild(rsaSpan);
        
        grid.appendChild(resEl);
    }
}

// Download PDB helper
function downloadPDB(btn) {
    const card = btn.closest('.prediction-card-inner');
    const pdb = card.getAttribute('data-pdb');
    if (!pdb) return;
    
    const blob = new Blob([pdb], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `isodde_complex_${Date.now()}.pdb`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Download SDF helper
function downloadSDF(btn) {
    const card = btn.closest('.prediction-card-inner');
    const sdf = card.getAttribute('data-sdf');
    if (!sdf) return;
    
    const blob = new Blob([sdf], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `isodde_ligand_${Date.now()}.sdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// --- 3D Viewer, Pocket Focusing, and Heatmaps Implementation ---
function init3DViewer(cardId, containerId, pdbContent, protLen, ligLen) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = ''; // Clear stub

    try {
        // Initialize 3Dmol.js viewer
        let viewer = $3Dmol.createViewer($(container), {
            backgroundColor: '#05070a'
        });
        
        // Cache the viewer instance
        state.activeViewers.set(cardId, viewer);
        state.activeViewerStyles.set(cardId, 'cartoon');

        // Load PDB data
        viewer.addModel(pdbContent, "pdb");
        
        // Apply representation styles
        applyViewerStyle(cardId, 'cartoon');
        
        viewer.zoomTo();
        viewer.render();
    } catch (e) {
        console.error("Failed to initialize 3Dmol viewer:", e);
        container.innerHTML = `
            <div style="padding: 40px; text-align: center; color: var(--text-error);">
                <i class="fa-solid fa-triangle-exclamation fa-2x"></i>
                <p style="margin-top: 8px;">3D Viewer failed to load.</p>
            </div>
        `;
    }
}

function applyViewerStyle(cardId, styleName) {
    const viewer = state.activeViewers.get(cardId);
    if (!viewer) return;
    
    const card = document.getElementById(cardId);
    if (!card) return;
    const protLen = parseInt(card.getAttribute('data-seq-len') || 0);
    const totalLen = protLen + parseInt(card.getAttribute('data-lig-len') || 0);
    
    const plddtList = JSON.parse(card.getAttribute('data-plddt') || '[]');
    const ssList = JSON.parse(card.getAttribute('data-ss') || '[]');
    const rsaList = JSON.parse(card.getAttribute('data-rsa') || '[]');
    
    const colorMode = (state.activeViewerColors && state.activeViewerColors.get(cardId)) || 'chain';
    
    viewer.setStyle({}, {}); // Clear current style
    
    const ligSelection = { resi: Array.from({ length: totalLen - protLen }, (_, i) => protLen + i + 1) };

    // Apply residue-level styles for protein
    for (let i = 0; i < protLen; i++) {
        const resi = i + 1;
        let color = '#a29bfe'; // default coil color
        
        if (colorMode === 'plddt') {
            const plddt = plddtList[i] !== undefined ? plddtList[i] : 0.8;
            if (plddt >= 0.9) color = '#005f73';
            else if (plddt >= 0.7) color = '#0a9396';
            else if (plddt >= 0.5) color = '#ee9b00';
            else color = '#ae2012';
        } else if (colorMode === 'ss') {
            const ss = ssList[i] !== undefined ? ssList[i] : 2;
            const ssColors = ['#ff7675', '#ffeaa7', '#a29bfe'];
            color = ssColors[ss];
        } else if (colorMode === 'rsa') {
            const rsa = rsaList[i] !== undefined ? rsaList[i] : 0.0;
            color = rsa > 0.4 ? '#4ea8de' : '#f3a712';
        } else {
            // 'chain' / spectrum
            color = 'spectrum';
        }
        
        let styleObj = {};
        if (styleName === 'cartoon') {
            styleObj = { cartoon: { color: color } };
        } else if (styleName === 'stick') {
            styleObj = { stick: { color: color, radius: 0.15 } };
        } else if (styleName === 'sphere') {
            styleObj = { sphere: { color: color, scale: 0.9 } };
        }
        viewer.setStyle({ resi: resi }, styleObj);
    }
    
    // Style the ligand
    if (styleName === 'sphere') {
        viewer.setStyle(ligSelection, { sphere: { colorscheme: 'cyanCarbon', scale: 1.0 } });
    } else {
        viewer.setStyle(ligSelection, { stick: { colorscheme: 'cyanCarbon', radius: 0.35 }, sphere: { scale: 0.3 } });
    }
    
    viewer.render();
}

function setMolColorMode(selectEl) {
    const card = selectEl.closest('.prediction-card-inner');
    if (!card) return;
    const cardId = card.id;
    const colorMode = selectEl.value;
    
    if (!state.activeViewerColors) {
        state.activeViewerColors = new Map();
    }
    state.activeViewerColors.set(cardId, colorMode);
    
    const styleName = state.activeViewerStyles.get(cardId) || 'cartoon';
    applyViewerStyle(cardId, styleName);
}

function setMolStyle(btn, styleName) {
    const card = btn.closest('.prediction-card-inner');
    if (!card) return;
    const cardId = card.id;
    
    const controls = btn.parentNode;
    controls.querySelectorAll('button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    
    state.activeViewerStyles.set(cardId, styleName);
    applyViewerStyle(cardId, styleName);
}

function recenterViewer(btn) {
    const card = btn.closest('.prediction-card-inner');
    if (!card) return;
    const cardId = card.id;
    const viewer = state.activeViewers.get(cardId);
    if (viewer) {
        viewer.removeAllShapes(); // clear pocket highlights on recenter
        viewer.zoomTo();
        viewer.render();
    }
}

function renderPocketsTable(cardRoot, pockets, cardId) {
    const tbody = cardRoot.querySelector('.pockets-table-body');
    if (!pockets || pockets.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No pockets detected.</td></tr>';
        return;
    }
    
    tbody.innerHTML = pockets.map((pocket, idx) => `
        <tr>
            <td>Pocket ${idx+1}</td>
            <td>[${pocket.center.map(c => c.toFixed(1)).join(', ')}]</td>
            <td>${pocket.radius.toFixed(1)} Å</td>
            <td><span class="badge">${pocket.score.toFixed(3)}</span></td>
            <td>
                <button class="btn btn-xs btn-outline" onclick="focusPocket('${cardId}', ${idx})">
                    <i class="fa-solid fa-crosshairs"></i> View
                </button>
            </td>
        </tr>
    `).join('');
}

function focusPocket(cardId, pocketIdx) {
    const viewer = state.activeViewers.get(cardId);
    if (!viewer) return;
    
    // Clear any previous shapes
    viewer.removeAllShapes();
    
    // Find pocket coordinates in loaded history
    const baseId = cardId.replace('-reload', '');
    const historyItem = state.history.find(h => h.id === baseId || h.id + '-reload' === cardId);
    if (!historyItem || !historyItem.data.pockets[pocketIdx]) return;
    
    const pocket = historyItem.data.pockets[pocketIdx];
    const center = { x: pocket.center[0], y: pocket.center[1], z: pocket.center[2] };
    const radius = pocket.radius;
    
    // Add transparent pocket sphere highlight
    viewer.addSphere({
        center: center,
        radius: radius,
        color: 'orange',
        alpha: 0.45,
        wireframe: false
    });
    
    // Zoom in on the pocket zone
    viewer.zoomTo({ center: center }, 800);
    viewer.render();
    
    addAssistantMessage(`Focused viewer on Pocket ${pocketIdx+1} (Center: [${pocket.center.map(c => c.toFixed(1)).join(', ')}], Radius: ${radius.toFixed(1)}Å, Score: ${pocket.score.toFixed(3)}). Zone sphere is highlighted in orange.`);
}

function renderContactHeatmaps(cardRoot, data) {
    const canvasInt = cardRoot.querySelector('.interface-canvas');
    const canvasPl = cardRoot.querySelector('.pl-canvas');
    
    // 1. Render Interface Contacts Heatmap
    if (canvasInt && data.interface_contact_probs) {
        const ctx = canvasInt.getContext('2d');
        const N = data.interface_contact_probs.length;
        const w = canvasInt.width;
        const h = canvasInt.height;
        ctx.fillStyle = '#05070a';
        ctx.fillRect(0, 0, w, h);
        
        if (N > 0) {
            const cellSize = w / N;
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < N; j++) {
                    const prob = data.interface_contact_probs[i][j];
                    if (prob > 0.05) {
                        // Blend cyan based on contact probability
                        ctx.fillStyle = `rgba(0, 242, 254, ${prob.toFixed(3)})`;
                        ctx.fillRect(i * cellSize, j * cellSize, cellSize, cellSize);
                    }
                }
            }
        }
    }
    
    // 2. Render Protein-Ligand Contacts Heatmap
    if (canvasPl) {
        const ctx = canvasPl.getContext('2d');
        const w = canvasPl.width;
        const h = canvasPl.height;
        ctx.fillStyle = '#05070a';
        ctx.fillRect(0, 0, w, h);
        
        const protLen = data.protein_length;
        const ligLen = data.ligand_length;
        
        if (ligLen === 0) {
            ctx.fillStyle = '#6b7280';
            ctx.font = '11px Outfit';
            ctx.textAlign = 'center';
            ctx.fillText('No Ligand Configured', w / 2, h / 2);
            return;
        }
        
        if (data.protein_ligand_contact_probs && protLen > 0) {
            const cellW = w / ligLen;
            const cellH = h / protLen;
            
            for (let r = 0; r < protLen; r++) {
                for (let c = 0; c < ligLen; c++) {
                    const globalColIdx = protLen + c;
                    if (globalColIdx < data.protein_ligand_contact_probs[r].length) {
                        const prob = data.protein_ligand_contact_probs[r][globalColIdx];
                        if (prob > 0.05) {
                            // Blend neon blue/purple based on probability
                            ctx.fillStyle = `rgba(79, 172, 254, ${prob.toFixed(3)})`;
                            ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
                        }
                    }
                }
            }
        }
    }
}

