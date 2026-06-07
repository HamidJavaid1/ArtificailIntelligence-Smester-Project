"""
Orbit Wars - Reinforcement Learning Agent
Hybrid agent combining neural network predictions with rule-based logic.
"""

from __future__ import annotations
import math
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import pickle
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from agent import Agent, _dist, _safe_angle


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network Architecture
# ─────────────────────────────────────────────────────────────────────────────

class OrbitWarsNet(nn.Module):
    """Neural network for Orbit Wars decision making."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(OrbitWarsNet, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Experience Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# State Encoder
# ─────────────────────────────────────────────────────────────────────────────

class StateEncoder:
    """Encodes game observation into neural network input."""
    
    def __init__(self, max_planets: int = 20, max_fleets: int = 50):
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.state_size = max_planets * 7 + max_fleets * 6 + 4  # planets + fleets + game info
        
    def encode(self, obs: dict, player_id: int) -> np.ndarray:
        """Encode observation into fixed-size state vector."""
        state = np.zeros(self.state_size)
        
        # Encode planets
        planets = obs.get('planets', [])
        for i, p in enumerate(planets[:self.max_planets]):
            idx = i * 7
            state[idx + 0] = p[0] / 100.0  # id normalized
            state[idx + 1] = (p[1] + 1) / 5.0  # owner normalized (-1 to 3)
            state[idx + 2] = p[2] / 100.0  # x normalized
            state[idx + 3] = p[3] / 100.0  # y normalized
            state[idx + 4] = p[4] / 10.0  # radius normalized
            state[idx + 5] = p[5] / 100.0  # ships normalized
            state[idx + 6] = p[6] / 10.0  # production normalized
        
        # Encode fleets
        fleets = obs.get('fleets', [])
        for i, f in enumerate(fleets[:self.max_fleets]):
            idx = self.max_planets * 7 + i * 6
            state[idx + 0] = f[0] / 1000.0  # id normalized
            state[idx + 1] = (f[1] + 1) / 5.0  # owner normalized
            state[idx + 2] = f[2] / 100.0  # x normalized
            state[idx + 3] = f[3] / 100.0  # y normalized
            state[idx + 4] = (f[4] + math.pi) / (2 * math.pi)  # angle normalized
            state[idx + 5] = f[6] / 100.0  # ships normalized
        
        # Encode game info
        idx = self.max_planets * 7 + self.max_fleets * 6
        state[idx + 0] = player_id / 4.0  # player id normalized
        state[idx + 1] = obs.get('turn', 0) / 500.0  # turn normalized
        state[idx + 2] = len([p for p in planets if p[1] == player_id]) / self.max_planets  # my planets
        state[idx + 3] = len([p for p in planets if p[1] >= 0 and p[1] != player_id]) / self.max_planets  # enemy planets
        
        return state


# ─────────────────────────────────────────────────────────────────────────────
# Action Decoder
# ─────────────────────────────────────────────────────────────────────────────

class ActionDecoder:
    """Decodes neural network output into game actions."""
    
    def __init__(self, max_actions: int = 5):
        self.max_actions = max_actions
        # Each action: [planet_id, angle, ships]
        self.action_size = max_actions * 3
        
    def decode(self, output: np.ndarray, obs: dict, player_id: int) -> List[List]:
        """Decode network output into list of actions."""
        actions = []
        planets = obs.get('planets', [])
        my_planets = [p for p in planets if p[1] == player_id]
        
        if not my_planets:
            return actions
        
        # Reshape output into actions
        output = output.reshape(self.max_actions, 3)
        
        for i in range(self.max_actions):
            planet_idx, angle_norm, ships_norm = output[i]
            
            # Select planet
            planet_idx = int(planet_idx * len(my_planets))
            planet_idx = max(0, min(planet_idx, len(my_planets) - 1))
            planet = my_planets[planet_idx]
            
            # Decode angle
            angle = angle_norm * 2 * math.pi - math.pi
            
            # Decode ships (percentage of available)
            ships_percent = max(0.1, min(0.8, ships_norm + 0.5))
            available = max(1, int(planet[5] * ships_percent))
            
            if available >= 1:
                actions.append([planet[0], angle, available])
        
        return actions


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid RL Agent
# ─────────────────────────────────────────────────────────────────────────────

class RLAgent(Agent):
    """Hybrid RL agent combining neural network with rule-based logic."""
    
    def __init__(
        self,
        player_id: int = 1,
        model_path: Optional[str] = None,
        use_rules: bool = True,
        epsilon: float = 0.1,
        seed: Optional[int] = None
    ):
        super().__init__(player_id=player_id, difficulty='rl', seed=seed)
        
        self.use_rules = use_rules
        self.epsilon = epsilon
        self.model_path = model_path
        
        # Initialize components
        self.state_encoder = StateEncoder()
        self.action_decoder = ActionDecoder()
        
        # Neural network
        self.state_size = self.state_encoder.state_size
        self.action_size = self.action_decoder.action_size
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.policy_net = OrbitWarsNet(self.state_size, self.action_size).to(self.device)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
            
            # Load model if path provided
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)
        else:
            print("Warning: PyTorch not available. Using rule-based fallback.")
            self.policy_net = None
        
        # Training mode flag
        self.training_mode = False
        
    def act(self, obs: dict) -> List[List]:
        """Generate actions using hybrid approach."""
        player = self.player_id
        planets = obs['planets']
        my_planets = [p for p in planets if p[1] == player]
        
        if not my_planets:
            return []
        
        # Rule-based fallback or hybrid
        if self.use_rules and (not TORCH_AVAILABLE or not self.policy_net or random.random() < self.epsilon):
            return self._rule_based_act(obs)
        
        # RL-based action
        if TORCH_AVAILABLE and self.policy_net:
            state = self.state_encoder.encode(obs, player)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.policy_net(state_tensor).cpu().numpy()[0]
            
            rl_actions = self.action_decoder.decode(output, obs, player)
            
            # Filter invalid actions
            valid_actions = []
            for action in rl_actions:
                planet_id, angle, ships = action
                planet = next((p for p in my_planets if p[0] == planet_id), None)
                if planet and ships <= planet[5]:
                    valid_actions.append(action)
            
            # Hybrid: combine with rule-based if enabled
            if self.use_rules and valid_actions:
                rule_actions = self._rule_based_act(obs)
                # Use rule actions for safety, RL for exploration
                if random.random() < 0.3:
                    return rule_actions
                return valid_actions[:1]  # Limit to 1 action for precision
            
            return valid_actions
        
        return self._rule_based_act(obs)
    
    def _rule_based_act(self, obs: dict) -> List[List]:
        """Fallback rule-based action (similar to easy agent)."""
        player = self.player_id
        planets = obs['planets']
        my_planets = [p for p in planets if p[1] == player]
        others = [p for p in planets if p[1] != player]
        
        if not others:
            return []
        
        actions = []
        
        for mp in my_planets:
            ships = mp[5]
            if ships < 15:
                continue
            
            # Find weakest target
            best_target = None
            best_score = float('inf')
            
            for target in others:
                target_ships = target[5]
                dist = _dist(mp[2], mp[3], target[2], target[3])
                
                if ships > target_ships * 2:
                    score = dist * 0.5 + target_ships * 0.3
                    if score < best_score:
                        best_score = score
                        best_target = target
            
            if best_target:
                send = int(best_target[5] * 1.5)
                send = min(send, int(ships) - 5)
                if send >= 3:
                    angle = _safe_angle(mp[2], mp[3], best_target[2], best_target[3])
                    actions.append([mp[0], angle, send])
        
        return actions
    
    def save_model(self, path: str):
        """Save the trained model."""
        if TORCH_AVAILABLE and self.policy_net:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)
            print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        if TORCH_AVAILABLE and self.policy_net and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {path}")
    
    def set_training_mode(self, training: bool):
        """Set training mode."""
        self.training_mode = training
        if TORCH_AVAILABLE and self.policy_net:
            self.policy_net.train(training)


# ─────────────────────────────────────────────────────────────────────────────
# Training Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

class RLTrainer:
    """Trainer for the RL agent."""
    
    def __init__(
        self,
        player_id: int = 1,
        opponent_difficulty: str = 'easy',
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        replay_buffer_size: int = 10000
    ):
        self.player_id = player_id
        self.opponent_difficulty = opponent_difficulty
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Initialize agent
        self.agent = RLAgent(player_id=player_id, use_rules=False, epsilon=1.0)
        self.agent.set_training_mode(True)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def calculate_reward(self, old_state: dict, new_state: dict, action_taken: List) -> float:
        """Calculate reward for an action."""
        # Simple reward: change in total ships
        old_ships = sum(p[5] for p in old_state['planets'] if p[1] == self.player_id)
        new_ships = sum(p[5] for p in new_state['planets'] if p[1] == self.player_id)
        
        reward = (new_ships - old_ships) * 0.1
        
        # Bonus for capturing planets
        old_planets = len([p for p in old_state['planets'] if p[1] == self.player_id])
        new_planets = len([p for p in new_state['planets'] if p[1] == self.player_id])
        reward += (new_planets - old_planets) * 10
        
        # Penalty for losing ships
        if reward < 0:
            reward *= 0.5
        
        return reward
    
    def train_episode(self, engine, max_turns: int = 500) -> Dict:
        """Train for one episode."""
        from agent import make_agent
        
        # Create opponent
        opponent = make_agent(1 - self.player_id, self.opponent_difficulty)
        
        episode_reward = 0
        episode_length = 0
        
        state = engine.observation(self.player_id)
        
        for turn in range(max_turns):
            if engine.is_over():
                break
            
            # Get action from agent
            actions = self.agent.act(state)
            
            # Get opponent action
            opp_state = engine.observation(1 - self.player_id)
            opp_actions = opponent.act(opp_state)
            
            # Execute step
            all_actions = {
                self.player_id: [(a[0], a[1], a[2]) for a in actions],
                1 - self.player_id: [(a[0], a[1], a[2]) for a in opp_actions]
            }
            
            engine.step(all_actions)
            
            # Get new state
            new_state = engine.observation(self.player_id)
            
            # Calculate reward
            reward = self.calculate_reward(state, new_state, actions)
            episode_reward += reward
            episode_length += 1
            
            # Store experience
            if TORCH_AVAILABLE:
                encoded_state = self.agent.state_encoder.encode(state, self.player_id)
                encoded_new_state = self.agent.state_encoder.encode(new_state, self.player_id)
                
                # For simplicity, use first action as the action index
                action_idx = 0 if actions else 0
                
                experience = Experience(
                    state=encoded_state,
                    action=action_idx,
                    reward=reward,
                    next_state=encoded_new_state,
                    done=engine.is_over()
                )
                self.replay_buffer.push(experience)
            
            state = new_state
        
        # Train network
        if len(self.replay_buffer) > self.batch_size:
            self._train_step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.agent.epsilon = self.epsilon
        
        # Record statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'epsilon': self.epsilon
        }
    
    def _train_step(self):
        """Perform one training step."""
        if not TORCH_AVAILABLE:
            return
        
        experiences = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.agent.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.agent.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.agent.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.agent.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.agent.device)
        
        # Current Q values
        current_q_values = self.agent.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        next_q_values = self.agent.policy_net(next_states).max(1)[0].detach()
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss
        loss = nn.functional.mse_loss(current_q_values.squeeze(), expected_q_values)
        
        # Optimize
        self.agent.optimizer.zero_grad()
        loss.backward()
        self.agent.optimizer.step()
    
    def train(self, num_episodes: int = 100, save_path: str = 'rl_model.pth'):
        """Train the agent for multiple episodes."""
        from engine import Engine
        
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            engine = Engine(num_players=2, seed=episode)
            
            stats = self.train_episode(engine)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Epsilon: {stats['epsilon']:.3f}")
                
                # Save checkpoint
                self.agent.save_model(save_path)
        
        print("Training complete!")
        self.agent.save_model(save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

def make_rl_agent(
    player_id: int = 1,
    model_path: Optional[str] = 'rl_model.pth',
    use_rules: bool = True,
    seed: Optional[int] = None
) -> RLAgent:
    """Create an RL agent."""
    return RLAgent(
        player_id=player_id,
        model_path=model_path,
        use_rules=use_rules,
        seed=seed
    )


def train_rl_agent(
    player_id: int = 1,
    num_episodes: int = 100,
    save_path: str = 'rl_model.pth',
    opponent_difficulty: str = 'easy'
):
    """Train an RL agent."""
    trainer = RLTrainer(
        player_id=player_id,
        opponent_difficulty=opponent_difficulty
    )
    trainer.train(num_episodes, save_path)
