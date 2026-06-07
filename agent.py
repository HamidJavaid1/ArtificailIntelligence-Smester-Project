"""
Orbit Wars - AI Agent
Submittable bot. Implements the required observation→action interface.

Usage (standalone submission):
    from agent import Agent
    bot = Agent(player_id=1, difficulty='hard')
    actions = bot.act(observation)   # returns [[planet_id, angle, ships], ...]

Difficulty levels:
  easy   – random target selection, conservative sends
  medium – nearest weak planet, basic sun avoidance
  hard   – scoring heuristic, threat response, comet capture, multi-launch
  elite  – advanced strategy, fleet coordination, predictive attacks, expansion optimization
"""

from __future__ import annotations
import math
import random
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (self-contained so agent.py can be submitted standalone)
# ─────────────────────────────────────────────────────────────────────────────

def _dist(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def _angle_to(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)

def _line_circle(x1, y1, x2, y2, cx, cy, cr):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy
    a = dx*dx + dy*dy
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - cr*cr
    d = b*b - 4*a*c
    if d < 0:
        return False
    d = math.sqrt(d)
    t1 = (-b - d) / (2*a)
    t2 = (-b + d) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)

def _safe_angle(px, py, tx, ty, sun_x=50, sun_y=50, sun_r=10):
    """Return angle to target, nudging if path crosses the sun."""
    base = _angle_to(px, py, tx, ty)
    if not _line_circle(px, py, tx, ty, sun_x, sun_y, sun_r + 1.5):
        return base
    # Try small offsets
    for delta in [0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2]:
        a = base + delta
        ex = px + math.cos(a) * 80
        ey = py + math.sin(a) * 80
        if not _line_circle(px, py, ex, ey, sun_x, sun_y, sun_r + 1.5):
            return a
    return base  # fallback

# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    """
    Orbit Wars AI Agent.

    Parameters
    ----------
    player_id : int
        Which player slot this agent controls (0-3).
    difficulty : str
        'easy', 'medium', or 'hard'.
    seed : int | None
        Random seed for reproducibility.
    """

    DIFFICULTIES = ('easy', 'medium', 'hard', 'elite')

    def __init__(
        self,
        player_id: int = 1,
        difficulty: str = 'medium',
        seed: Optional[int] = None,
    ):
        self.player_id  = player_id
        self.difficulty = difficulty.lower()
        assert self.difficulty in self.DIFFICULTIES, f"difficulty must be one of {self.DIFFICULTIES}"
        self._rng = random.Random(seed)

    # ── Main interface ────────────────────────────────────────────────────────

    def act(self, obs: dict) -> List[List]:
        """
        Given an observation dict, return a list of actions.
        Each action: [planet_id, angle_radians, num_ships]
        Returns [] to pass the turn.
        """
        player    = self.player_id
        planets   = obs['planets']   # [[id,owner,x,y,r,ships,prod], ...]
        fleets    = obs['fleets']    # [[id,owner,x,y,angle,from_id,ships], ...]
        comets    = obs.get('comet_planet_ids', [])
        turn      = obs.get('turn', 0)

        my_planets = [p for p in planets if p[1] == player]
        if not my_planets:
            return []

        if self.difficulty == 'easy':
            return self._act_easy(player, planets, my_planets, fleets)
        elif self.difficulty == 'medium':
            return self._act_medium(player, planets, my_planets, fleets, comets)
        elif self.difficulty == 'hard':
            return self._act_hard(player, planets, my_planets, fleets, comets, turn)
        else:
            return self._act_elite(player, planets, my_planets, fleets, comets, turn)

    # ── Easy ─────────────────────────────────────────────────────────────────

    def _act_easy(self, player, planets, my_planets, fleets) -> List[List]:
        actions = []
        others  = [p for p in planets if p[1] != player]
        if not others:
            return []

        for mp in my_planets:
            ships = mp[5]
            if ships < 10:
                continue
            send = int(ships * 0.4)
            if send < 2:
                continue
            target = self._rng.choice(others)
            angle  = _safe_angle(mp[2], mp[3], target[2], target[3])
            actions.append([mp[0], angle, send])

        return actions

    # ── Medium ───────────────────────────────────────────────────────────────

    def _act_medium(self, player, planets, my_planets, fleets, comets) -> List[List]:
        actions = []
        others  = [p for p in planets if p[1] != player]
        if not others:
            return []

        for mp in my_planets:
            ships = mp[5]
            if ships < 6:
                continue
            send = int(ships * 0.55)
            if send < 2:
                continue

            # Score targets: prefer close + weak
            best = min(
                others,
                key=lambda t: _dist(mp[2], mp[3], t[2], t[3]) * (1 + t[5] * 0.02)
            )
            angle = _safe_angle(mp[2], mp[3], best[2], best[3])
            actions.append([mp[0], angle, send])

        return actions

    # ── Hard ─────────────────────────────────────────────────────────────────

    def _act_hard(self, player, planets, my_planets, fleets, comets, turn) -> List[List]:
        actions = []
        others   = [p for p in planets if p[1] != player]
        neutrals = [p for p in planets if p[1] == -1]
        enemies  = [p for p in planets if p[1] >= 0 and p[1] != player]
        comet_set = set(comets)

        if not others:
            return []

        # Incoming threat map: which of my planets are being attacked?
        incoming: dict = {}
        for f in fleets:
            if f[1] == player:
                continue
            # Check if this fleet is heading toward one of my planets
            for mp in my_planets:
                if _line_circle(f[2], f[3],
                                f[2] + math.cos(f[4]) * 200,
                                f[3] + math.sin(f[4]) * 200,
                                mp[2], mp[3], mp[4]):
                    incoming[mp[0]] = incoming.get(mp[0], 0) + f[6]

        for mp in my_planets:
            ships = mp[5]
            pid, px, py = mp[0], mp[2], mp[3]

            # Defend if threatened
            threat = incoming.get(pid, 0)
            reserve = max(5, int(threat * 1.2)) if threat > 0 else 4

            available = ships - reserve
            if available < 3:
                continue

            # --- Prioritize comet capture ---
            comet_targets = [p for p in others if p[0] in comet_set]
            for ct in comet_targets:
                d = _dist(px, py, ct[2], ct[3])
                if d < 35 and ships > 5:
                    send  = min(int(available * 0.3), 5)
                    angle = _safe_angle(px, py, ct[2], ct[3])
                    actions.append([pid, angle, send])
                    available -= send
                    break

            if available < 3:
                continue

            # --- Score all non-owned targets ---
            def score(t):
                d     = _dist(px, py, t[2], t[3])
                is_n  = t[1] == -1
                prod  = t[6]
                tships = t[5]
                # Prefer high production, close, low garrison
                base  = d * 0.5 - prod * 8 + tships * 0.4
                if is_n:
                    base -= 5        # bonus for neutrals (easier)
                if tships > ships * 0.8:
                    base += 20       # penalty if outgunned
                return base

            valid = [t for t in others if t[5] < available * 1.5]
            if not valid:
                valid = others

            target = min(valid, key=score)
            tx, ty = target[2], target[3]
            garrison = target[5]

            # Send enough to win + buffer
            send = min(int(available * 0.65), int(garrison * 1.4) + 8)
            send = max(send, 3)
            send = min(send, int(available))

            if send < 2:
                continue

            angle = _safe_angle(px, py, tx, ty)
            actions.append([pid, angle, send])

            # Multi-launch: if very rich, also send explorers to neutrals
            available -= send
            if available > 20 and neutrals:
                closest_n = min(neutrals, key=lambda t: _dist(px, py, t[2], t[3]))
                if closest_n[0] != target[0]:
                    explore_send = min(int(available * 0.4), 10)
                    if explore_send >= 3:
                        a2 = _safe_angle(px, py, closest_n[2], closest_n[3])
                        actions.append([pid, a2, explore_send])

        return actions

    # ── Elite ─────────────────────────────────────────────────────────────────

    def _act_elite(self, player, planets, my_planets, fleets, comets, turn) -> List[List]:
        actions = []
        others   = [p for p in planets if p[1] != player]
        neutrals = [p for p in planets if p[1] == -1]
        enemies  = [p for p in planets if p[1] >= 0 and p[1] != player]
        comet_set = set(comets)

        if not others:
            return []

        # Advanced threat analysis with predictive modeling
        incoming: dict = {}
        for f in fleets:
            if f[1] == player:
                continue
            for mp in my_planets:
                if _line_circle(f[2], f[3],
                                f[2] + math.cos(f[4]) * 200,
                                f[3] + math.sin(f[4]) * 200,
                                mp[2], mp[3], mp[4]):
                    dist = _dist(f[2], f[3], mp[2], mp[3])
                    threat = f[6] * (1 + 50 / (dist + 1))
                    incoming[mp[0]] = incoming.get(mp[0], 0) + threat

        # Economic analysis
        my_total_ships = sum(mp[5] for mp in my_planets)
        enemy_total_ships = sum(p[5] for p in enemies) + sum(f[6] for f in fleets if f[1] in [e[1] for e in enemies])

        # Strategic phase determination
        early_game = turn < 50
        mid_game = 50 <= turn < 150
        late_game = turn >= 150

        for mp in my_planets:
            ships = mp[5]
            pid, px, py = mp[0], mp[2], mp[3]

            # Dynamic reserve calculation
            threat = incoming.get(pid, 0)
            if early_game:
                reserve = max(3, int(threat * 1.1))
            elif mid_game:
                reserve = max(4, int(threat * 1.3))
            else:
                reserve = max(5, int(threat * 1.5))

            available = ships - reserve
            if available < 4:
                continue

            # Elite comet capture with timing optimization
            comet_targets = [p for p in others if p[0] in comet_set]
            for ct in comet_targets:
                d = _dist(px, py, ct[2], ct[3])
                comet_threshold = 40 if early_game else 30
                if d < comet_threshold and ships > 8:
                    send = min(int(available * 0.4), 8)
                    angle = _safe_angle(px, py, ct[2], ct[3])
                    actions.append([pid, angle, send])
                    available -= send
                    break

            if available < 4:
                continue

            # Advanced target scoring
            def elite_score(t):
                d = _dist(px, py, t[2], t[3])
                is_n = t[1] == -1
                is_e = t[1] >= 0 and t[1] != player
                prod = t[6]
                tships = t[5]
                
                base = d * 0.4 - prod * 10 + tships * 0.3
                
                if is_n:
                    base -= 8
                    if prod >= 4:
                        base -= 5
                
                if is_e:
                    if tships < ships * 0.5:
                        base -= 15
                    elif tships > ships * 1.5:
                        base += 25
                
                if early_game and is_n:
                    base -= 10
                elif late_game and is_e and tships < ships:
                    base -= 12
                
                return base

            valid = [t for t in others if t[5] < available * 2.0]
            if not valid:
                valid = others

            target = min(valid, key=elite_score)
            tx, ty = target[2], target[3]
            garrison = target[5]

            if target[1] == -1:
                send = min(int(available * 0.7), int(garrison * 1.2) + 5)
            else:
                if garrison < available:
                    send = min(int(available * 0.8), int(garrison * 1.5) + 10)
                else:
                    send = min(int(available * 0.5), int(available * 0.6))
            
            send = max(send, 4)
            send = min(send, int(available))

            if send < 3:
                continue

            angle = _safe_angle(px, py, tx, ty)
            actions.append([pid, angle, send])

            available -= send
            
            # Elite multi-attack coordination
            if available > 25 and len(valid) > 1:
                valid_without_first = [t for t in valid if t[0] != target[0]]
                if valid_without_first:
                    target2 = min(valid_without_first, key=elite_score)
                    if target2[1] == -1 or target2[5] < available * 0.6:
                        send2 = min(int(available * 0.4), 15)
                        if send2 >= 4:
                            a2 = _safe_angle(px, py, target2[2], target2[3])
                            actions.append([pid, a2, send2])
                            available -= send2

            # Exploratory fleet to distant neutral
            if available > 15 and neutrals:
                distant_n = max(neutrals, key=lambda t: _dist(px, py, t[2], t[3]))
                if distant_n[0] != target[0]:
                    explore_send = min(int(available * 0.3), 8)
                    if explore_send >= 3:
                        a3 = _safe_angle(px, py, distant_n[2], distant_n[3])
                        actions.append([pid, a3, explore_send])

        # Elite defensive coordination
        threatened_planets = [mp for mp in my_planets if incoming.get(mp[0], 0) > 0]
        if len(threatened_planets) > 1:
            safe_planets = [mp for mp in my_planets if incoming.get(mp[0], 0) == 0 and mp[5] > 20]
            if safe_planets:
                defender = max(safe_planets, key=lambda p: p[5])
                most_threatened = max(threatened_planets, key=lambda p: incoming.get(p[0], 0))
                
                if defender[5] > 30:
                    reinforce = int(defender[5] * 0.3)
                    if reinforce >= 5:
                        angle = _safe_angle(defender[2], defender[3], most_threatened[2], most_threatened[3])
                        existing = [a for a in actions if a[0] == defender[0]]
                        if not existing:
                            actions.append([defender[0], angle, reinforce])

        return actions


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory (used by engine wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def make_agent(player_id: int, difficulty: str = 'medium', seed: int = None) -> Agent:
    return Agent(player_id=player_id, difficulty=difficulty, seed=seed)