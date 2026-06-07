# Orbit Wars - Reinforcement Learning Agent

A hybrid RL agent that combines neural network predictions with rule-based logic for precise and accurate gameplay.

## Features

- **Hybrid Architecture**: Combines deep Q-learning with rule-based fallback
- **Neural Network**: PyTorch-based policy network for decision making
- **Experience Replay**: Efficient training with replay buffer
- **State Encoding**: Converts game observations to neural network inputs
- **Action Decoding**: Translates network outputs to game actions
- **Rule-Based Fallback**: Ensures robustness when RL is uncertain

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

This installs:
- torch (PyTorch for neural networks)
- numpy (numerical operations)
- flask (web server)
- flask-cors (CORS support)

## Training the RL Agent

Train the agent on your laptop using the training script:

```bash
python train_rl.py --episodes 100 --opponent easy --save rl_model.pth
```

### Training Options

- `--episodes`: Number of training episodes (default: 100)
- `--player`: Player ID (0 or 1, default: 1)
- `--opponent`: Opponent difficulty - easy, medium, hard, elite (default: easy)
- `--save`: Model save path (default: rl_model.pth)
- `--lr`: Learning rate (default: 0.001)
- `--gamma`: Discount factor (default: 0.99)
- `--batch`: Batch size (default: 32)
- `--buffer`: Replay buffer size (default: 10000)

### Example Training Commands

```bash
# Quick training (50 episodes against easy opponent)
python train_rl.py --episodes 50 --opponent easy

# Full training (500 episodes against medium opponent)
python train_rl.py --episodes 500 --opponent medium --save my_model.pth

# Advanced training with custom parameters
python train_rl.py --episodes 200 --opponent hard --lr 0.0005 --batch 64

# Train against elite opponent (hardest difficulty)
python train_rl.py --episodes 500 --opponent elite --save elite_model.pth
```

## Using the Trained Agent

### Command Line

```bash
python main.py --difficulty rl --seed 42
```

### Web Application

1. Start the web server:
```bash
python app.py
```

2. Open the game in your browser
3. Select 'rl' as the difficulty level

### Python API

```python
from agent import make_agent
from rl_agent import make_rl_agent

# Use the RL agent with trained model
agent = make_rl_agent(player_id=1, model_path='rl_model.pth', use_rules=True)

# Get actions from observation
actions = agent.act(observation)
```

## Architecture Details

### Neural Network

- **Input Layer**: State vector (planets + fleets + game info)
- **Hidden Layers**: 2 hidden layers with 256 units each
- **Output Layer**: Action space (planet_id, angle, ships)
- **Activation**: ReLU with dropout (0.2)
- **Optimizer**: Adam

### State Encoding

The state encoder converts game observations into fixed-size vectors:
- Planets: ID, owner, x, y, radius, ships, production (normalized)
- Fleets: ID, owner, x, y, angle, ships (normalized)
- Game info: player_id, turn, my_planets, enemy_planets (normalized)

### Action Decoding

The action decoder translates network outputs into game actions:
- Selects planet from owned planets
- Calculates launch angle
- Determines ship count (percentage of available)

### Hybrid Approach

The agent uses a hybrid strategy:
- **RL Mode**: Uses neural network predictions (when available and confident)
- **Rule Mode**: Falls back to rule-based logic (when RL unavailable or uncertain)
- **Epsilon-Greedy**: Explores randomly during training

## Training Process

1. **Initialization**: Create agent and opponent
2. **Episode Loop**: Play game for specified turns
3. **Action Selection**: Agent selects action (epsilon-greedy during training)
4. **Environment Step**: Execute actions and get new state
5. **Reward Calculation**: Compute reward based on ship/planet changes
6. **Experience Storage**: Store (state, action, reward, next_state, done) in replay buffer
7. **Network Update**: Train network on batch of experiences
8. **Epsilon Decay**: Gradually reduce exploration rate

## Reward Function

The reward function encourages:
- **Ship Growth**: +0.1 per ship gained
- **Planet Capture**: +10 per planet captured
- **Ship Loss**: -0.5 per ship lost (penalty reduced)

## Performance Tips

### Training

- Start with easy opponent for faster learning
- Use GPU if available (PyTorch CUDA support)
- Monitor training progress every 10 episodes
- Save checkpoints regularly
- Adjust learning rate if training is unstable

### Inference

- Set `use_rules=True` for better performance
- Load trained model with `model_path` parameter
- Set `epsilon=0.0` to disable exploration during inference

## Troubleshooting

### PyTorch Not Available

If PyTorch is not installed, the agent will fall back to rule-based mode:
```bash
pip install torch
```

### CUDA Out of Memory

If training fails due to GPU memory:
- Reduce batch size: `--batch 16`
- Reduce replay buffer: `--buffer 5000`
- Use CPU: PyTorch will automatically fall back

### Slow Training

To speed up training:
- Reduce number of episodes
- Use easier opponent
- Reduce state size (modify StateEncoder parameters)
- Use GPU if available

## File Structure

```
orbitwar/
├── rl_agent.py          # RL agent implementation
├── train_rl.py          # Training script
├── agent.py             # Base agent with RL integration
├── engine.py            # Game engine
├── rl_model.pth         # Trained model (generated after training)
└── requirements.txt     # Dependencies
```

## Advanced Usage

### Custom Reward Function

Modify the `calculate_reward` method in `RLTrainer` to implement custom rewards:

```python
def calculate_reward(self, old_state, new_state, action_taken):
    # Your custom reward logic here
    return reward
```

### Custom Network Architecture

Modify the `OrbitWarsNet` class in `rl_agent.py`:

```python
class CustomNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # Your custom architecture
```

### Transfer Learning

Load a pre-trained model and continue training:

```python
agent = RLAgent(player_id=1, model_path='pretrained.pth')
trainer = RLTrainer(player_id=1)
trainer.agent = agent
trainer.train(num_episodes=50)
```

## License

This RL agent is part of the Orbit Wars project.
