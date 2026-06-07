"""
Orbit Wars - Web Server
Flask server with Python game engine and AI agents.
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from engine import Engine
from agent import make_agent

app = Flask(__name__)
CORS(app)

# Store game state in memory
games = {}

@app.route('/')
def index():
    """Serve the game HTML page."""
    return render_template('index.html')

@app.route('/api/test', methods=['GET'])
def test():
    """Test endpoint to verify server is working."""
    return jsonify({'status': 'ok', 'message': 'Server is working'})

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Create a new game instance."""
    try:
        data = request.json
        print(f"New game request: {data}")
        num_players = data.get('players', 2)
        difficulty = data.get('difficulty', 'elite')
        seed = data.get('seed', None)
        mode = data.get('mode', 'human_vs_ai')  # 'human_vs_ai' or 'agent_vs_agent'
        
        game_id = str(len(games))
        engine = Engine(num_players=num_players, seed=seed)
        
        # Handle different game modes
        if mode == 'agent_vs_agent':
            # Agent vs Agent mode: specify difficulties for each player
            player_difficulties = data.get('player_difficulties', {})
            agents = {}
            for i in range(num_players):
                diff = player_difficulties.get(str(i), difficulty)
                agents[i] = make_agent(i, diff)
        else:
            # Human vs AI mode (default)
            agents = {i: make_agent(i, difficulty) for i in range(1, num_players)}
        
        games[game_id] = {
            'engine': engine,
            'agents': agents,
            'difficulty': difficulty,
            'mode': mode
        }
        
        print(f"Game created with ID: {game_id}, mode: {mode}")
        print(f"Available game IDs: {list(games.keys())}")
        
        return jsonify({
            'game_id': game_id,
            'state': serialize_state(engine.state),
            'is_over': engine.is_over(),
            'mode': mode
        })
    except Exception as e:
        import traceback
        print(f"Error in new_game endpoint: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/state/<game_id>', methods=['GET'])
def get_state(game_id):
    """Get current game state."""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game = games[game_id]
    return jsonify({
        'state': serialize_state(game['engine'].state),
        'is_over': game['engine'].is_over(),
        'winner': game['engine'].winner() if game['engine'].is_over() else None,
        'scores': game['engine'].scores()
    })

@app.route('/api/step', methods=['POST'])
def step():
    """Execute one game step."""
    try:
        with open('debug.log', 'a') as f:
            f.write(f"=== STEP REQUEST ===\n")
            data = request.json
            f.write(f"Data: {data}\n")
            game_id = data.get('game_id')
            player_actions = data.get('actions', {})
            
            f.write(f"Games in memory: {list(games.keys())}\n")
            f.write(f"Looking for game_id: {game_id}\n")
            
            if game_id not in games:
                f.write(f"Game not found!\n")
                return jsonify({'error': 'Game not found'}), 404
            
            game = games[game_id]
            engine = game['engine']
            agents = game['agents']
            mode = game.get('mode', 'human_vs_ai')
            
            f.write(f"Game mode: {mode}\n")
            f.write(f"Player actions: {player_actions}\n")
            
            # Log planet ownership for debugging
            f.write(f"Planets: {[(p.id, p.owner, p.ships) for p in engine.state.planets]}\n")
            
            # Convert JavaScript dict format to Python tuples for player 0
            f.write(f"Keys in player_actions: {list(player_actions.keys())}\n")
            if '0' in player_actions:
                # Convert string key to integer and dict actions to tuples
                f.write(f"Converting player 0 actions from: {player_actions['0']}\n")
                player_actions[0] = [(a['planet_id'], a['angle'], a['ships']) for a in player_actions['0']]
                del player_actions['0']
                f.write(f"Converted player 0 actions: {player_actions[0]}\n")
            
            # Get AI actions for all AI players
            # In agent vs agent mode, all players are AI
            # In human vs ai mode, only players 1+ are AI
            if mode == 'agent_vs_agent':
                for i, agent in agents.items():
                    obs = engine.observation(i)
                    acts = agent.act(obs)
                    player_actions[i] = [(a[0], a[1], a[2]) for a in acts]
            else:
                for i, agent in agents.items():
                    obs = engine.observation(i)
                    acts = agent.act(obs)
                    player_actions[i] = [(a[0], a[1], a[2]) for a in acts]
            
            f.write(f"Final actions: {player_actions}\n")
            
            # Execute step
            still_running = engine.step(player_actions)
            
            f.write(f"Step completed: {still_running}\n")
            f.write(f"Fleets after step: {len(engine.state.fleets)}\n")
            for fleet in engine.state.fleets:
                f.write(f"  Fleet id={fleet.id}, owner={fleet.owner}, ships={fleet.ships}, from_planet={fleet.from_planet_id}\n")
            f.write(f"Planet 0 ships after step: {engine.state.planets[0].ships}\n")
            f.write(f"Planet 1 ships after step: {engine.state.planets[1].ships}\n")
            
            return jsonify({
                'state': serialize_state(engine.state),
                'is_over': engine.is_over(),
                'winner': engine.winner() if engine.is_over() else None,
                'scores': engine.scores()
            })
    except Exception as e:
        import traceback
        with open('debug.log', 'a') as f:
            f.write(f"=== ERROR ===\n")
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/auto_play', methods=['POST'])
def auto_play():
    """Auto-play multiple steps for agent vs agent games."""
    try:
        data = request.json
        game_id = data.get('game_id')
        max_steps = data.get('max_steps', 100)
        delay_ms = data.get('delay_ms', 100)
        
        if game_id not in games:
            return jsonify({'error': 'Game not found'}), 404
        
        game = games[game_id]
        engine = game['engine']
        mode = game.get('mode', 'human_vs_ai')
        
        if mode != 'agent_vs_agent':
            return jsonify({'error': 'Auto-play only available in agent_vs_agent mode'}), 400
        
        states = []
        steps_played = 0
        
        for _ in range(max_steps):
            if engine.is_over():
                break
            
            # Get actions from all AI agents
            player_actions = {}
            for i, agent in game['agents'].items():
                obs = engine.observation(i)
                acts = agent.act(obs)
                player_actions[i] = [(a[0], a[1], a[2]) for a in acts]
            
            # Execute step
            engine.step(player_actions)
            
            # Save state
            states.append(serialize_state(engine.state))
            steps_played += 1
        
        return jsonify({
            'states': states,
            'steps_played': steps_played,
            'is_over': engine.is_over(),
            'winner': engine.winner() if engine.is_over() else None,
            'scores': engine.scores()
        })
    except Exception as e:
        import traceback
        with open('debug.log', 'a') as f:
            f.write(f"=== AUTO_PLAY ERROR ===\n")
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def serialize_state(state):
    """Convert game state to JSON-serializable format."""
    return {
        'turn': state.turn,
        'num_players': state.num_players,
        'planets': [
            {
                'id': p.id,
                'x': p.x,
                'y': p.y,
                'radius': p.radius,
                'owner': p.owner,
                'ships': p.ships,
                'production': p.production,
                'orbit_r': p.orbit_r,
                'angle': p.angle,
                'is_comet': p.is_comet
            }
            for p in state.planets
        ],
        'fleets': [
            {
                'owner': f.owner,
                'ships': f.ships,
                'x': f.x,
                'y': f.y,
                'angle': f.angle,
                'speed': f.speed,
                'from_planet_id': f.from_planet_id
            }
            for f in state.fleets
        ]
    }

if __name__ == '__main__':
    app.run(debug=True, port=5000)
