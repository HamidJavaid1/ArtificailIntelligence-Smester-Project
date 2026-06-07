// Orbit Wars - Web Client
// Handles game rendering and interaction in the browser

const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

// Game state
let gameState = null;
let gameId = null;
let selectedPlanet = null;
let aimAngle = 0;
let queuedOrders = [];
let autoPlay = false;
let autoPlayInterval = null;

// Configuration
const CONFIG = {
    WORLD_W: 100,
    WORLD_H: 100,
    SUN_X: 50,
    SUN_Y: 50,
    SUN_R: 10,
    MAX_TURNS: 500
};

// Player colors
const PLAYER_COLORS = [
    '#3b82f6',  // Blue - human
    '#f97316',  // Orange
    '#22c55e',  // Green
    '#ea9808'   // Yellow
];

const NEUTRAL_COLOR = '#4a5568';

// Initialize game
async function newGame() {
    const difficulty = document.getElementById('difficultySelect').value;
    const players = parseInt(document.getElementById('playersSelect').value);
    
    try {
        const response = await fetch('/api/new_game', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ players, difficulty, seed: null })
        });
        
        const data = await response.json();
        gameId = data.game_id;
        gameState = data.state;
        selectedPlanet = null;
        queuedOrders = [];
        
        document.getElementById('gameOverOverlay').style.display = 'none';
        render();
        updateUI();
    } catch (error) {
        console.error('Error creating game:', error);
    }
}

// Step the game forward
async function stepGame() {
    if (!gameId || !gameState) return;
    
    // Collect player 0 actions
    const playerActions = {};
    playerActions[0] = queuedOrders.map(o => ({
        planet_id: o.planetId,
        angle: o.angle,
        ships: o.ships
    }));
    queuedOrders = [];
    
    try {
        const response = await fetch('/api/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ game_id: gameId, actions: playerActions })
        });
        
        const data = await response.json();
        gameState = data.state;
        
        if (data.is_over) {
            stopAutoPlay();
            showGameOver(data.winner, data.scores);
        }
        
        render();
        updateUI();
    } catch (error) {
        console.error('Error stepping game:', error);
    }
}

// Render the game
function render() {
    if (!gameState) return;
    
    // Clear canvas
    ctx.fillStyle = '#0b0d12';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw stars
    drawStars();
    
    // Calculate scale
    const scale = Math.min(canvas.width, canvas.height) / 100;
    const offsetX = (canvas.width - 100 * scale) / 2;
    const offsetY = (canvas.height - 100 * scale) / 2;
    
    // Draw sun glow
    const sunX = offsetX + CONFIG.SUN_X * scale;
    const sunY = offsetY + CONFIG.SUN_Y * scale;
    drawSunGlow(sunX, sunY, scale);
    
    // Draw orbit rings
    gameState.planets.forEach(p => {
        if (p.orbit_r) {
            ctx.beginPath();
            ctx.arc(sunX, sunY, p.orbit_r * scale, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    });
    
    // Draw sun
    ctx.beginPath();
    ctx.arc(sunX, sunY, CONFIG.SUN_R * scale, 0, Math.PI * 2);
    ctx.fillStyle = '#f59e0b';
    ctx.fill();
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw fleets
    gameState.fleets.forEach(f => {
        const fx = offsetX + f.x * scale;
        const fy = offsetY + f.y * scale;
        const color = PLAYER_COLORS[f.owner % PLAYER_COLORS.length];
        
        drawArrow(fx, fy, f.angle, 7, color);
        
        if (f.ships >= 5) {
            ctx.fillStyle = color;
            ctx.font = '12px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(f.ships, fx, fy - 14);
        }
    });
    
    // Draw planets
    gameState.planets.forEach(p => {
        const px = offsetX + p.x * scale;
        const py = offsetY + p.y * scale;
        const pr = Math.max(4, p.radius * scale);
        const color = p.owner >= 0 ? PLAYER_COLORS[p.owner % PLAYER_COLORS.length] : NEUTRAL_COLOR;
        const isSelected = selectedPlanet && selectedPlanet.id === p.id;
        
        // Glow for owned planets
        if (p.owner >= 0) {
            ctx.beginPath();
            ctx.arc(px, py, pr * 2, 0, Math.PI * 2);
            ctx.fillStyle = color + '40';
            ctx.fill();
        }
        
        // Planet body
        ctx.beginPath();
        ctx.arc(px, py, pr, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        
        // Border
        ctx.strokeStyle = isSelected ? '#ffffff' : color;
        ctx.lineWidth = isSelected ? 3 : 1;
        ctx.stroke();
        
        // Ship count
        ctx.fillStyle = '#ffffff';
        const fontSize = Math.max(8, Math.min(14, pr));
        ctx.font = `${fontSize}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(Math.floor(p.ships), px, py);
    });
    
    // Draw aim line
    if (selectedPlanet && selectedPlanet.owner === 0) {
        const spx = offsetX + selectedPlanet.x * scale;
        const spy = offsetY + selectedPlanet.y * scale;
        const length = 80;
        const ex = spx + Math.cos(aimAngle) * length;
        const ey = spy + Math.sin(aimAngle) * length;
        
        ctx.beginPath();
        ctx.moveTo(spx, spy);
        ctx.lineTo(ex, ey);
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Dashes
        for (let t = 0; t < 80; t += 10) {
            const x1 = spx + Math.cos(aimAngle) * t;
            const y1 = spy + Math.sin(aimAngle) * t;
            const x2 = spx + Math.cos(aimAngle) * (t + 5);
            const y2 = spy + Math.sin(aimAngle) * (t + 5);
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
        
        drawArrow(ex, ey, aimAngle, 10, '#3b82f6');
    }
}

function drawStars() {
    // Simple static stars
    const seed = 42;
    for (let i = 0; i < 200; i++) {
        const x = (seed * i * 7919) % canvas.width;
        const y = (seed * i * 7907) % canvas.height;
        const r = (seed * i * 7919) % 1.5 + 0.3;
        const alpha = (seed * i * 7907) % 1;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${alpha * 200}, ${alpha * 200}, ${alpha * 200}, ${alpha})`;
        ctx.fill();
    }
}

function drawSunGlow(x, y, scale) {
    const glowR = CONFIG.SUN_R * scale * 3;
    for (let i = glowR; i > 0; i -= 4) {
        const alpha = Math.max(0, 60 * (1 - i / glowR));
        ctx.beginPath();
        ctx.arc(x, y, i, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(251, 191, 36, ${alpha / 255})`;
        ctx.fill();
    }
}

function drawArrow(x, y, angle, size, color) {
    ctx.beginPath();
    ctx.moveTo(x + Math.cos(angle) * size, y + Math.sin(angle) * size);
    ctx.lineTo(x + Math.cos(angle + 2.4) * size * 0.5, y + Math.sin(angle + 2.4) * size * 0.5);
    ctx.lineTo(x + Math.cos(angle - 2.4) * size * 0.5, y + Math.sin(angle - 2.4) * size * 0.5);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
}

function updateUI() {
    if (!gameState) return;
    
    document.getElementById('turnDisplay').textContent = `${gameState.turn} / ${CONFIG.MAX_TURNS}`;
    
    // Calculate total ships per player
    const playerShips = [0, 0, 0, 0];
    gameState.planets.forEach(p => {
        if (p.owner >= 0) {
            playerShips[p.owner] += p.ships;
        }
    });
    gameState.fleets.forEach(f => {
        playerShips[f.owner] += f.ships;
    });
    
    for (let i = 0; i < gameState.num_players; i++) {
        const el = document.getElementById(`player${i}Ships`);
        if (el) el.textContent = Math.floor(playerShips[i]);
    }
}

function showGameOver(winner, scores) {
    const overlay = document.getElementById('gameOverOverlay');
    const winnerText = document.getElementById('winnerText');
    
    let winnerName = winner === 0 ? 'You' : `AI ${winner}`;
    winnerText.textContent = `Winner: ${winnerName}`;
    
    overlay.style.display = 'flex';
}

function toggleAutoPlay() {
    autoPlay = !autoPlay;
    const btn = document.getElementById('autoPlayBtn');
    
    if (autoPlay) {
        btn.textContent = 'Stop Auto Play (A)';
        btn.style.background = '#ef4444';
        autoPlayInterval = setInterval(stepGame, 500);
    } else {
        stopAutoPlay();
    }
}

function stopAutoPlay() {
    autoPlay = false;
    const btn = document.getElementById('autoPlayBtn');
    btn.textContent = 'Auto Play (A)';
    btn.style.background = '#3b82f6';
    if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
        autoPlayInterval = null;
    }
}

// Event listeners
canvas.addEventListener('click', (e) => {
    if (!gameState) return;
    
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    
    const scale = Math.min(canvas.width, canvas.height) / 100;
    const offsetX = (canvas.width - 100 * scale) / 2;
    const offsetY = (canvas.height - 100 * scale) / 2;
    
    const wx = (mx - offsetX) / scale;
    const wy = (my - offsetY) / scale;
    
    // Check planet click
    let clicked = null;
    for (const p of gameState.planets) {
        const dist = Math.sqrt((wx - p.x) ** 2 + (wy - p.y) ** 2);
        if (dist < p.radius + 1.5) {
            clicked = p;
            break;
        }
    }
    
    if (clicked) {
        if (clicked.owner === 0) {
            selectedPlanet = clicked;
        } else if (selectedPlanet && selectedPlanet.owner === 0) {
            // Aim at clicked planet
            aimAngle = Math.atan2(clicked.y - selectedPlanet.y, clicked.x - selectedPlanet.x);
            // Queue launch order
            const ships = Math.max(1, Math.floor(selectedPlanet.ships / 2));
            queuedOrders.push({
                planetId: selectedPlanet.id,
                angle: aimAngle,
                ships: ships
            });
        }
    } else {
        // Aim in direction of click
        if (selectedPlanet && selectedPlanet.owner === 0) {
            aimAngle = Math.atan2(wy - selectedPlanet.y, wx - selectedPlanet.x);
        }
    }
    
    render();
});

canvas.addEventListener('mousemove', (e) => {
    if (!gameState || !selectedPlanet || selectedPlanet.owner !== 0) return;
    
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    
    const scale = Math.min(canvas.width, canvas.height) / 100;
    const offsetX = (canvas.width - 100 * scale) / 2;
    const offsetY = (canvas.height - 100 * scale) / 2;
    
    const wx = (mx - offsetX) / scale;
    const wy = (my - offsetY) / scale;
    
    aimAngle = Math.atan2(wy - selectedPlanet.y, wx - selectedPlanet.x);
    render();
});

document.getElementById('stepBtn').addEventListener('click', stepGame);
document.getElementById('autoPlayBtn').addEventListener('click', toggleAutoPlay);
document.getElementById('newGameBtn').addEventListener('click', newGame);

document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
        e.preventDefault();
        stepGame();
    } else if (e.code === 'KeyA') {
        toggleAutoPlay();
    } else if (e.code === 'KeyR') {
        newGame();
    }
});

// Start game on load
newGame();
