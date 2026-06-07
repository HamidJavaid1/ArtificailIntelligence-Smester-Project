"""
Elite AI Agent for Orbit Wars - Strategic & Aggressive
Features: Rapid expansion, predictive targeting, aggressive early game
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

CENTER = (50, 50)
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0


def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def will_cross_sun(start, end):
    """Check if a path segment crosses or gets too close to the sun."""
    x1, y1 = start
    x2, y2 = end
    cx, cy = CENTER
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return distance((x1, y1), CENTER) < SUN_RADIUS + 2
    
    # Parametric line equation: p(t) = (x1, y1) + t*(dx, dy)
    # Find t that minimizes distance to sun
    t = ((cx - x1) * dx + (cy - y1) * dy) / (dx*dx + dy*dy)
    t = max(0, min(1, t))
    
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    dist_to_sun = distance((closest_x, closest_y), CENTER)
    
    return dist_to_sun < SUN_RADIUS + 1


def is_valid_move(from_pos, angle, max_dist):
    """Check if a move is valid (safe and in bounds)."""
    end_x = from_pos[0] + max_dist * math.cos(angle)
    end_y = from_pos[1] + max_dist * math.sin(angle)
    
    # Check bounds with buffer
    if end_x < 2 or end_x > BOARD_SIZE - 2 or end_y < 2 or end_y > BOARD_SIZE - 2:
        return False
    
    # Check sun
    if will_cross_sun(from_pos, (end_x, end_y)):
        return False
    
    return True


def elite_agent(obs):
    """
    Elite AI Agent - Focuses on rapid territory conquest and strategic expansion.
    """
    moves = []
    
    # Parse observation with robustness
    if isinstance(obs, dict):
        player = obs.get("player", 0)
        raw_planets = obs.get("planets", [])
        raw_fleets = obs.get("fleets", [])
    else:
        player = obs.player
        raw_planets = obs.planets
        raw_fleets = obs.fleets
    
    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    
    # Categorize planets
    my_planets = [p for p in planets if p.owner == player]
    neutral_planets = [p for p in planets if p.owner == -1]
    enemy_planets = [p for p in planets if p.owner != player and p.owner != -1]
    
    if not my_planets:
        return moves
    
    # Track committed ships per planet
    committed = {}
    
    def can_send(planet_id, ships):
        planet = next((p for p in my_planets if p.id == planet_id), None)
        if not planet:
            return False
        used = committed.get(planet_id, 0)
        return planet.ships - used >= ships
    
    def send_ships(planet_id, ships):
        committed[planet_id] = committed.get(planet_id, 0) + ships
    
    # PHASE 1: Aggressive Neutral Expansion (HIGH PRIORITY)
    # Sort neutrals by value and distance
    expansion_targets = []
    for neutral in neutral_planets:
        for my_planet in my_planets:
            d = distance((my_planet.x, my_planet.y), (neutral.x, neutral.y))
            # Score: production (higher = better) - distance (lower = better)
            score = neutral.production * 100 - d
            expansion_targets.append((neutral, my_planet, d, score))
    
    # Sort by score (best targets first)
    expansion_targets.sort(key=lambda x: -x[3])
    
    attacked_neutrals = set()
    for neutral, source, dist, _ in expansion_targets:
        if neutral.id in attacked_neutrals:
            continue
        
        # Calculate required force with aggressive margin
        required = max(neutral.ships + 1, 6)  # Aggressive - minimal surplus
        
        if can_send(source.id, required):
            angle = math.atan2(neutral.y - source.y, neutral.x - source.x)
            if is_valid_move((source.x, source.y), angle, dist):
                moves.append([source.id, angle, required])
                send_ships(source.id, required)
                attacked_neutrals.add(neutral.id)
    
    # PHASE 2: Strategic Enemy Attacks (Medium Priority)
    # Find and attack weak enemy planets
    for enemy in enemy_planets:
        if enemy.ships > 80:  # Skip strong enemies
            continue
        
        # Find best attacker
        best_attacker = None
        best_dist = float('inf')
        for my_planet in my_planets:
            available = my_planet.ships - committed.get(my_planet.id, 0)
            if available >= 15:  # Minimum attack force
                d = distance((my_planet.x, my_planet.y), (enemy.x, enemy.y))
                if d < best_dist:
                    best_dist = d
                    best_attacker = my_planet
        
        if best_attacker:
            required = max(enemy.ships + 2, 18)
            if can_send(best_attacker.id, required):
                angle = math.atan2(enemy.y - best_attacker.y, enemy.x - best_attacker.x)
                if is_valid_move((best_attacker.x, best_attacker.y), angle, best_dist):
                    moves.append([best_attacker.id, angle, required])
                    send_ships(best_attacker.id, required)
    
    # PHASE 3: Defensive Reinforcement (Lower Priority)
    # Reinforce planets that are too weak
    for my_planet in my_planets:
        if my_planet.ships < 7:
            # Find nearest strong planet to reinforce from
            reinforcer = None
            best_dist = float('inf')
            
            for other in my_planets:
                if other.id == my_planet.id:
                    continue
                available = other.ships - committed.get(other.id, 0)
                if available > 12:
                    d = distance((other.x, other.y), (my_planet.x, my_planet.y))
                    if d < best_dist:
                        best_dist = d
                        reinforcer = other
            
            if reinforcer:
                transfer = min(int((reinforcer.ships - committed.get(reinforcer.id, 0)) * 0.2), 10)
                if transfer > 2 and can_send(reinforcer.id, transfer):
                    angle = math.atan2(my_planet.y - reinforcer.y, my_planet.x - reinforcer.x)
                    if is_valid_move((reinforcer.x, reinforcer.y), angle, best_dist):
                        moves.append([reinforcer.id, angle, transfer])
                        send_ships(reinforcer.id, transfer)
    
    return moves


if __name__ == "__main__":
    from kaggle_environments import make
    
    env = make("orbit_wars", debug=True)
    print(f"Environment: {env.name}")
    print("="*60)
    print("Starting Elite AI Agent vs Random")
    print("="*60)
    
    # Run elite agent
    env.run([elite_agent, "random"])
    
    final = env.steps[-1]
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for i, s in enumerate(final):
        status_msg = "WIN" if s.reward > 0 else "LOSS" if s.reward < 0 else "DRAW"
        print(f"Player {i}: {status_msg:6s} (reward={s.reward:+d})")
    
    # Render
    html = env.render(mode="html", width=900, height=700)
    if html:
        with open("render.html", "w") as f:
            f.write(html)
        print("\n✓ Game visualization saved to render.html")
