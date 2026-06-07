# Orbit Wars - Agent vs Agent Mode Guide

Guide for using trained RL models and agent vs agent gameplay in the browser.

## Trained RL Models for Difficulty Levels

The game now supports using trained reinforcement learning models for easy and hard difficulty levels.

### Training Models for Specific Difficulties

Train an RL model and save it with a specific filename to use it for a difficulty level:

```bash
# Train model for easy difficulty
python train_rl.py --episodes 100 --opponent easy --save easy_rl_model.pth

# Train model for hard difficulty
python train_rl.py --episodes 500 --opponent elite --save hard_rl_model.pth
```

### How It Works

- **Easy Difficulty**: If `easy_rl_model.pth` exists, the game will use the trained RL model instead of the rule-based easy agent
- **Hard Difficulty**: If `hard_rl_model.pth` exists, the game will use the trained RL model instead of the rule-based hard agent
- **Fallback**: If the model file doesn't exist or PyTorch is not available, the game falls back to the rule-based agent

### Example Workflow

```bash
# 1. Train a model for easy difficulty
python train_rl.py --episodes 200 --opponent easy --save easy_rl_model.pth

# 2. Use it in the game
python main.py --difficulty easy --seed 42

# 3. Or use it in the web app
python app.py
# Then select 'easy' difficulty in the browser
```

## Agent vs Agent Mode

Watch two AI agents play against each other in the browser.

### Creating an Agent vs Agent Game

Use the `/api/new_game` endpoint with `mode: 'agent_vs_agent'`:

```json
{
  "players": 2,
  "mode": "agent_vs_agent",
  "player_difficulties": {
    "0": "hard",
    "1": "medium"
  },
  "seed": 42
}
```

### API Endpoints

#### Create Agent vs Agent Game

```bash
curl -X POST http://localhost:5000/api/new_game \
  -H "Content-Type: application/json" \
  -d '{
    "players": 2,
    "mode": "agent_vs_agent",
    "player_difficulties": {
      "0": "hard",
      "1": "medium"
    },
    "seed": 42
  }'
```

Response:
```json
{
  "game_id": "0",
  "state": {...},
  "is_over": false,
  "mode": "agent_vs_agent"
}
```

#### Step Through Game

```bash
curl -X POST http://localhost:5000/api/step \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "0",
    "actions": {}
  }'
```

Note: In agent vs agent mode, you don't need to provide actions - the AI agents generate them automatically.

#### Auto-Play Multiple Steps

Watch multiple turns at once:

```bash
curl -X POST http://localhost:5000/api/auto_play \
  -H "Content-Type: application/json" \
  -d '{
    "game_id": "0",
    "max_steps": 100,
    "delay_ms": 100
  }'
```

Response:
```json
{
  "states": [...],  // Array of game states for each step
  "steps_played": 100,
  "is_over": false,
  "winner": null,
  "scores": [120, 85]
}
```

## Browser Integration

### JavaScript Example

```javascript
// Create agent vs agent game
async function createAgentVsAgentGame() {
  const response = await fetch('/api/new_game', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      players: 2,
      mode: 'agent_vs_agent',
      player_difficulties: {
        '0': 'hard',
        '1': 'medium'
      },
      seed: 42
    })
  });
  const data = await response.json();
  return data.game_id;
}

// Step through game
async function stepGame(gameId) {
  const response = await fetch('/api/step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      game_id: gameId,
      actions: {}
    })
  });
  return await response.json();
}

// Auto-play game
async function autoPlayGame(gameId, maxSteps = 100) {
  const response = await fetch('/api/auto_play', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      game_id: gameId,
      max_steps: maxSteps,
      delay_ms: 100
    })
  });
  return await response.json();
}

// Example usage
const gameId = await createAgentVsAgentGame();
console.log('Game created:', gameId);

// Watch the game play
const result = await autoPlayGame(gameId, 50);
console.log('Steps played:', result.steps_played);
console.log('Winner:', result.winner);
```

## Popular Agent vs Agent Matchups

### Hard vs Medium
```json
{
  "mode": "agent_vs_agent",
  "player_difficulties": {
    "0": "hard",
    "1": "medium"
  }
}
```

### Elite vs Hard
```json
{
  "mode": "agent_vs_agent",
  "player_difficulties": {
    "0": "elite",
    "1": "hard"
  }
}
```

### Medium vs Easy
```json
{
  "mode": "agent_vs_agent",
  "player_difficulties": {
    "0": "medium",
    "1": "easy"
  }
}
```

### RL vs Elite (if you have a trained model)
```json
{
  "mode": "agent_vs_agent",
  "player_difficulties": {
    "0": "rl",
    "1": "elite"
  }
}
```

## Training for Specific Matchups

You can train your RL agent against specific opponents to prepare for agent vs agent matches:

```bash
# Train against hard opponent to prepare for hard vs RL matches
python train_rl.py --episodes 500 --opponent hard --save hard_rl_model.pth

# Train against elite opponent for elite vs RL matches
python train_rl.py --episodes 500 --opponent elite --save elite_rl_model.pth
```

## Watching Games in Real-Time

To watch agent vs agent games in real-time in the browser:

1. Start the web server:
```bash
python app.py
```

2. Open the game in your browser at `http://localhost:5000`

3. Use the JavaScript console to create and watch games:

```javascript
// Create the game
const gameId = await (await fetch('/api/new_game', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    players: 2,
    mode: 'agent_vs_agent',
    player_difficulties: { '0': 'hard', '1': 'medium' },
    seed: 42
  })
})).json().then(d => d.game_id);

// Watch it play step by step
setInterval(async () => {
  const result = await (await fetch('/api/step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ game_id: gameId, actions: {} })
  })).json();
  
  console.log('Turn:', result.state.turn);
  console.log('Scores:', result.scores);
  
  if (result.is_over) {
    console.log('Game Over! Winner:', result.winner);
  }
}, 500);
```

## Tips for Agent vs Agent Testing

1. **Use Consistent Seeds**: Always use the same seed when comparing different agent versions
   ```json
   { "seed": 42 }
   ```

2. **Test Multiple Matchups**: Try different difficulty combinations to understand agent strengths
   - Hard vs Medium
   - Elite vs Hard
   - RL vs Elite

3. **Auto-Play for Speed**: Use the auto-play endpoint to run many games quickly for statistical analysis

4. **Save Interesting Games**: Note the seeds of interesting games for later analysis

5. **Compare with Trained Models**: Test your trained RL models against rule-based agents to measure improvement

## Troubleshooting

### RL Model Not Loading

If your trained model isn't being used:

1. Check the filename matches exactly:
   - Easy: `easy_rl_model.pth`
   - Hard: `hard_rl_model.pth`

2. Verify PyTorch is installed:
```bash
python -c "import torch; print(torch.__version__)"
```

3. Check the model file exists:
```bash
ls easy_rl_model.pth
```

### Agent vs Agent Not Working

If agent vs agent mode isn't working:

1. Ensure you're sending the correct mode:
```json
{ "mode": "agent_vs_agent" }
```

2. Verify player_difficulties are set:
```json
{ "player_difficulties": { "0": "hard", "1": "medium" } }
```

3. Check that both difficulty levels are valid: 'easy', 'medium', 'hard', 'elite', 'rl'

### Auto-Play Too Slow

If auto-play is too slow:

1. Reduce max_steps:
```json
{ "max_steps": 50 }
```

2. Increase delay between steps (if using client-side animation):
```json
{ "delay_ms": 50 }
```

## Advanced Usage

### Running Tournaments

Create a simple script to run multiple agent vs agent matches:

```python
import requests
import json

def run_matchup(diff1, diff2, seed):
    response = requests.post('http://localhost:5000/api/new_game', json={
        'players': 2,
        'mode': 'agent_vs_agent',
        'player_difficulties': {'0': diff1, '1': diff2},
        'seed': seed
    })
    game_id = response.json()['game_id']
    
    # Auto-play until game over
    while True:
        response = requests.post('http://localhost:5000/auto_play', json={
            'game_id': game_id,
            'max_steps': 100
        })
        result = response.json()
        if result['is_over']:
            return result['winner']

# Run tournament
matchups = [
    ('hard', 'medium'),
    ('elite', 'hard'),
    ('medium', 'easy')
]

for diff1, diff2 in matchups:
    for seed in range(10):
        winner = run_matchup(diff1, diff2, seed)
        print(f"{diff1} vs {diff2} (seed {seed}): Winner = {winner}")
```

### Analyzing Game Results

Save game states for later analysis:

```python
# Save states during auto-play
response = requests.post('http://localhost:5000/auto_play', json={
    'game_id': game_id,
    'max_steps': 100
})
result = response.json()

# Save states to file
with open('game_states.json', 'w') as f:
    json.dump(result['states'], f)
```
