"""
Orbit Wars - Game Engine
Orchestrates one full turn following the exact spec order.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional

import config
import physics
from game_state import GameState, Planet, Fleet
from combat import Arrival, resolve_arrivals


Action = Tuple[int, float, int]   # (planet_id, angle_rad, num_ships)


class Engine:
    """
    Wraps GameState and drives the simulation.

    Turn order (exact):
      1. Remove expired comets
      2. Spawn comet groups
      3. Process player fleet launches
      4. Planet production
      5. Fleet movement + collision checks
      6. Planet rotation + comet movement
      7. Sweeping collision resolution
      8. Combat resolution
      9. Win condition check
    """

    def __init__(self, num_players: int = 2, seed: Optional[int] = None):
        self.state = GameState(num_players=num_players, seed=seed)

    # ── Public API ───────────────────────────────────────────────────────────

    def step(self, actions_per_player: Dict[int, List[Action]]) -> bool:
        """
        Advance one turn.
        actions_per_player: {player_id: [(planet_id, angle, ships), ...]}
        Returns True if game is still running, False if over.
        """
        if self.state.over:
            return False

        G = self.state
        G.turn += 1

        # ── 1. Remove expired comets ─────────────────────────────────────────
        expired_ids = set()
        still_alive = []
        for c in G.comets:
            c.ttl -= 1
            if c.ttl <= 0 or physics.out_of_bounds(c.x, c.y):
                expired_ids.add(c.id)
            else:
                still_alive.append(c)
        G.comets  = still_alive
        G.planets = [p for p in G.planets if p.id not in expired_ids]

        # ── 2. Spawn comets ──────────────────────────────────────────────────
        if G.turn in config.COMET_SPAWN_TURNS:
            G.spawn_comets()

        # ── 3. Process fleet launches ────────────────────────────────────────
        # Group actions by planet to handle multiple launches correctly
        planet_actions: dict = {}  # {planet_id: [(player_id, angle, ships), ...]}
        for player_id, actions in actions_per_player.items():
            for (planet_id, angle, num_ships) in actions:
                if planet_id not in planet_actions:
                    planet_actions[planet_id] = []
                planet_actions[planet_id].append((player_id, angle, num_ships))
        
        # Process each planet's actions sequentially
        for planet_id, action_list in planet_actions.items():
            planet = self._get_planet(planet_id)
            if planet is None:
                continue
            
            # Launch fleets in order, deducting ships as we go
            for player_id, angle, num_ships in action_list:
                if num_ships >= 1 and int(planet.ships) >= 1:
                    G.launch_fleet(player_id, planet, angle, num_ships)

        # ── 4. Planet production ─────────────────────────────────────────────
        for p in G.planets:
            if p.owner >= 0:
                p.ships += p.production

        # ── 5. Fleet movement + collision detection ──────────────────────────
        to_remove: set = set()
        arrivals: Dict[int, List[Arrival]] = {}

        for f in G.fleets:
            prev_x, prev_y = f.x, f.y
            f.x += physics.fleet_speed(f.ships, config.FLEET_MAX_SPEED) * _dx(f.angle)
            f.y += physics.fleet_speed(f.ships, config.FLEET_MAX_SPEED) * _dy(f.angle)

            # Out of bounds
            if physics.out_of_bounds(f.x, f.y):
                to_remove.add(f.id)
                continue

            # Hits sun
            if physics.fleet_hits_sun(prev_x, prev_y, f.x, f.y,
                                      config.SUN_X, config.SUN_Y, config.SUN_R):
                to_remove.add(f.id)
                continue

            # Hits planet
            hit = False
            for p in G.planets:
                # Don't immediately collide with source planet
                if p.id == f.from_planet_id:
                    dist_now = physics.distance(f.x, f.y, p.x, p.y)
                    if dist_now < p.radius + 1.0:
                        continue
                if physics.fleet_hits_planet(prev_x, prev_y, f.x, f.y,
                                             p.x, p.y, p.radius):
                    if p.id not in arrivals:
                        arrivals[p.id] = []
                    arrivals[p.id].append(Arrival(owner=f.owner, ships=f.ships))
                    to_remove.add(f.id)
                    hit = True
                    break

        G.fleets = [f for f in G.fleets if f.id not in to_remove]

        # ── 6. Orbital movement ──────────────────────────────────────────────
        for p in G.planets:
            if p.orbits:
                p.angle += p.ang_vel
                p.x, p.y = physics.orbit_position(
                    config.SUN_X, config.SUN_Y, p.orbit_r, p.angle
                )

        # ── 7. & 8. Combat resolution ────────────────────────────────────────
        for planet_id, arr_list in arrivals.items():
            planet = self._get_planet(planet_id)
            if planet is not None:
                resolve_arrivals(planet, arr_list)

        # ── 9. Win condition ─────────────────────────────────────────────────
        active = G.active_players()
        if len(active) <= 1 or G.turn >= config.MAX_TURNS:
            G.over = True
            if active:
                # If one player left → winner; else whoever has most ships
                if len(active) == 1:
                    G.winner = active[0]
                else:
                    sc = G.scores()
                    G.winner = max(sc, key=sc.get)
            return False

        return True

    def observation(self, player: int) -> dict:
        return self.state.observation(player)

    def is_over(self) -> bool:
        return self.state.over

    def winner(self) -> int:
        return self.state.winner

    def scores(self) -> Dict[int, int]:
        return self.state.scores()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_planet(self, planet_id: int) -> Optional[Planet]:
        for p in self.state.planets:
            if p.id == planet_id:
                return p
        return None


def _dx(angle: float) -> float:
    import math
    return math.cos(angle)

def _dy(angle: float) -> float:
    import math
    return math.sin(angle)