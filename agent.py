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

    DIFFICULTIES = ('easy', 'medium', 'hard', 'elite', 'rl')

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
            # Check if RL model is available for easy
            try:
                from rl_agent import make_rl_agent
                if os.path.exists('easy_rl_model.pth'):
                    rl_agent = make_rl_agent(player_id=player, model_path='easy_rl_model.pth', use_rules=True)
                    return rl_agent.act(obs)
            except (ImportError, Exception):
                pass
            return self._act_easy(player, planets, my_planets, fleets)
        elif self.difficulty == 'medium':
            return self._act_medium(player, planets, my_planets, fleets, comets)
        elif self.difficulty == 'hard':
            # Check if RL model is available for hard
            try:
                from rl_agent import make_rl_agent
                if os.path.exists('hard_rl_model.pth'):
                    rl_agent = make_rl_agent(player_id=player, model_path='hard_rl_model.pth', use_rules=True)
                    return rl_agent.act(obs)
            except (ImportError, Exception):
                pass
            return self._act_hard(player, planets, my_planets, fleets, comets, turn)
        elif self.difficulty == 'rl':
            # Import RL agent dynamically to avoid circular imports
            try:
                from rl_agent import make_rl_agent
                rl_agent = make_rl_agent(player_id=player, use_rules=True)
                return rl_agent.act(obs)
            except ImportError:
                # Fallback to easy if RL not available
                return self._act_easy(player, planets, my_planets, fleets)
        else:
            return self._act_elite(player, planets, my_planets, fleets, comets, turn)

    # ── Easy ─────────────────────────────────────────────────────────────────

    def _act_easy(self, player, planets, my_planets, fleets) -> List[List]:
        """Easy agent: Conservative, launches only when it has clear advantage."""
        actions = []
        others  = [p for p in planets if p[1] != player]
        if not others:
            return []

        # Only launch from planets with significant ship count
        for mp in my_planets:
            ships = mp[5]
            if ships < 15:
                continue

            # Find weakest target that we can definitely capture
            best_target = None
            best_score = float('inf')

            for target in others:
                target_ships = target[5]
                dist = _dist(mp[2], mp[3], target[2], target[3])

                # Only attack if we have at least 2x their ships
                if ships > target_ships * 2:
                    # Prefer closer targets with fewer ships
                    score = dist * 0.5 + target_ships * 0.3
                    if score < best_score:
                        best_score = score
                        best_target = target

            if best_target:
                # Send just enough to win (1.5x target ships to be safe)
                send = int(best_target[5] * 1.5)
                send = min(send, int(ships) - 5)  # Keep some reserve
                if send >= 3:
                    angle = _safe_angle(mp[2], mp[3], best_target[2], best_target[3])
                    actions.append([mp[0], angle, send])

        return actions

    # ── Medium ───────────────────────────────────────────────────────────────

    def _act_medium(self, player, planets, my_planets, fleets, comets) -> List[List]:
        """Medium agent: Balanced precision, considers production and distance."""
        actions = []
        others  = [p for p in planets if p[1] != player]
        neutrals = [p for p in planets if p[1] == -1]
        if not others:
            return []

        for mp in my_planets:
            ships = mp[5]
            if ships < 10:
                continue

            # Calculate available ships after keeping reserve
            reserve = 3
            available = ships - reserve
            if available < 4:
                continue

            # Score targets based on multiple factors
            best_target = None
            best_score = float('inf')

            for target in others:
                target_ships = target[5]
                target_prod = target[6]
                dist = _dist(mp[2], mp[3], target[2], target[3])
                is_neutral = target[1] == -1

                # Calculate ships needed to capture
                ships_needed = int(target_ships * 1.25)  # 25% buffer
                if is_neutral:
                    ships_needed = int(target_ships * 1.1)  # Less buffer for neutrals

                # Only attack if we have enough ships
                if available >= ships_needed:
                    # Score: distance + target strength - production value
                    # Lower score is better
                    score = dist * 0.35 + target_ships * 0.18 - target_prod * 2.5
                    if is_neutral:
                        score -= 6  # Bonus for neutrals

                    if score < best_score:
                        best_score = score
                        best_target = target
                        best_send = ships_needed

            if best_target:
                angle = _safe_angle(mp[2], mp[3], best_target[2], best_target[3])
                actions.append([mp[0], angle, best_send])

        return actions

    # ── Hard ─────────────────────────────────────────────────────────────────

    def _act_hard(self, player, planets, my_planets, fleets, comets, turn) -> List[List]:
        """Hard agent: Highly precise, calculates exact ship counts and considers threats."""
        actions = []
        others   = [p for p in planets if p[1] != player]
        neutrals = [p for p in planets if p[1] == -1]
        enemies  = [p for p in planets if p[1] >= 0 and p[1] != player]
        comet_set = set(comets)

        if not others:
            return []

        # Calculate incoming threats to each planet
        incoming: dict = {}
        for f in fleets:
            if f[1] == player:
                continue
            for mp in my_planets:
                if _line_circle(f[2], f[3],
                                f[2] + math.cos(f[4]) * 200,
                                f[3] + math.sin(f[4]) * 200,
                                mp[2], mp[3], mp[4]):
                    incoming[mp[0]] = incoming.get(mp[0], 0) + f[6]

        for mp in my_planets:
            ships = mp[5]
            pid, px, py = mp[0], mp[2], mp[3]

            # Calculate defensive reserve based on threats
            threat = incoming.get(pid, 0)
            if threat > 0:
                reserve = int(threat * 1.5) + 3  # 50% buffer + minimum
            else:
                reserve = 3  # Minimum reserve

            available = ships - reserve
            if available < 5:
                continue

            # Prioritize comet capture if close and valuable
            comet_targets = [p for p in others if p[0] in comet_set]
            for ct in comet_targets:
                d = _dist(px, py, ct[2], ct[3])
                if d < 30 and available > 8:
                    # Send precise amount to capture comet
                    send = min(int(ct[5] * 1.2) + 2, int(available * 0.3))
                    if send >= 3:
                        angle = _safe_angle(px, py, ct[2], ct[3])
                        actions.append([pid, angle, send])
                        available -= send
                        break

            if available < 5:
                continue

            # Find the best target with precise calculations
            best_target = None
            best_score = float('inf')
            best_send = 0

            for target in others:
                target_ships = target[5]
                target_prod = target[6]
                dist = _dist(px, py, target[2], target[3])
                is_neutral = target[1] == -1
                is_enemy = target[1] >= 0 and target[1] != player

                # Calculate exact ships needed
                if is_neutral:
                    ships_needed = int(target_ships * 1.15)  # 15% buffer for neutrals
                elif is_enemy:
                    ships_needed = int(target_ships * 1.4)  # 40% buffer for enemies
                else:
                    ships_needed = int(target_ships * 1.2)

                # Only consider if we have enough ships
                if available >= ships_needed:
                    # Comprehensive scoring
                    score = dist * 0.3 + target_ships * 0.15 - target_prod * 3

                    # Bonuses and penalties
                    if is_neutral:
                        score -= 8  # Strong bonus for neutrals
                    if is_enemy and target_ships < ships * 0.6:
                        score -= 12  # Bonus for weak enemies
                    if is_enemy and target_ships > ships:
                        score += 50  # Heavy penalty for strong enemies

                    if score < best_score:
                        best_score = score
                        best_target = target
                        best_send = ships_needed

            if best_target:
                angle = _safe_angle(px, py, best_target[2], best_target[3])
                actions.append([pid, angle, best_send])

        return actions

    # ── Elite ─────────────────────────────────────────────────────────────────

    def _act_elite(self, player, planets, my_planets, fleets, comets, turn) -> List[List]:
        """Elite agent: Ultra-precise, strategic timing, minimal but highly effective launches."""
        actions = []
        others   = [p for p in planets if p[1] != player]
        neutrals = [p for p in planets if p[1] == -1]
        enemies  = [p for p in planets if p[1] >= 0 and p[1] != player]
        comet_set = set(comets)

        if not others:
            return []

        # Advanced threat analysis with distance-weighted threat calculation
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
                    # Closer fleets are more threatening
                    threat = f[6] * (1 + 30 / (dist + 1))
                    incoming[mp[0]] = incoming.get(mp[0], 0) + threat

        # Game phase analysis
        early_game = turn < 50
        mid_game = 50 <= turn < 150
        late_game = turn >= 150

        for mp in my_planets:
            ships = mp[5]
            pid, px, py = mp[0], mp[2], mp[3]

            # Dynamic reserve based on threat and game phase
            threat = incoming.get(pid, 0)
            if early_game:
                reserve = max(4, int(threat * 1.2))
            elif mid_game:
                reserve = max(5, int(threat * 1.4))
            else:
                reserve = max(6, int(threat * 1.6))

            available = ships - reserve
            if available < 6:
                continue

            # Strategic comet capture - only if very close and valuable
            comet_targets = [p for p in others if p[0] in comet_set]
            for ct in comet_targets:
                d = _dist(px, py, ct[2], ct[3])
                comet_threshold = 25 if early_game else 20
                if d < comet_threshold and available > 10:
                    # Precise calculation for comet capture
                    send = min(int(ct[5] * 1.1) + 3, int(available * 0.25))
                    if send >= 4:
                        angle = _safe_angle(px, py, ct[2], ct[3])
                        actions.append([pid, angle, send])
                        available -= send
                        break

            if available < 6:
                continue

            # Ultra-precise target selection
            best_target = None
            best_score = float('inf')
            best_send = 0

            for target in others:
                target_ships = target[5]
                target_prod = target[6]
                dist = _dist(px, py, target[2], target[3])
                is_neutral = target[1] == -1
                is_enemy = target[1] >= 0 and target[1] != player

                # Precise ship calculation based on target type
                if is_neutral:
                    ships_needed = int(target_ships * 1.1)  # 10% buffer for neutrals
                elif is_enemy:
                    ships_needed = int(target_ships * 1.35)  # 35% buffer for enemies
                else:
                    ships_needed = int(target_ships * 1.15)

                # Only attack if we have sufficient advantage
                if available >= ships_needed:
                    # Multi-factor scoring
                    score = dist * 0.25 + target_ships * 0.1 - target_prod * 4

                    # Strategic bonuses
                    if is_neutral:
                        score -= 10  # High priority on neutrals
                        if target_prod >= 4:
                            score -= 8  # Extra bonus for high-production neutrals

                    if is_enemy:
                        if target_ships < ships * 0.4:
                            score -= 15  # Easy target bonus
                        elif target_ships > ships:
                            score += 100  # Avoid suicide attacks

                    # Game phase adjustments
                    if early_game and is_neutral:
                        score -= 12  # Expand early
                    if late_game and is_enemy and target_ships < ships * 0.7:
                        score -= 10  # Aggressive late game

                    if score < best_score:
                        best_score = score
                        best_target = target
                        best_send = ships_needed

            # Only launch one fleet per planet per turn (precision over quantity)
            if best_target:
                angle = _safe_angle(px, py, best_target[2], best_target[3])
                actions.append([pid, angle, best_send])

        # Strategic defensive reinforcement - only if critical
        threatened_planets = [mp for mp in my_planets if incoming.get(mp[0], 0) > 0]
        if len(threatened_planets) == 1:
            # If only one planet is threatened, reinforce from safe planets
            threatened = threatened_planets[0]
            threat_level = incoming.get(threatened[0], 0)
            safe_planets = [mp for mp in my_planets if incoming.get(mp[0], 0) == 0 and mp[5] > 25]

            if safe_planets and threat_level > threatened[5] * 0.8:
                defender = max(safe_planets, key=lambda p: p[5])
                # Calculate precise reinforcement needed
                reinforce_needed = int(threat_level - threatened[5] * 0.5)
                reinforce = min(reinforce_needed, int(defender[5] * 0.4))
                if reinforce >= 8:
                    angle = _safe_angle(defender[2], defender[3], threatened[2], threatened[3])
                    existing = [a for a in actions if a[0] == defender[0]]
                    if not existing:
                        actions.append([defender[0], angle, reinforce])

        return actions


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory (used by engine wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def make_agent(player_id: int, difficulty: str = 'medium', seed: int = None) -> Agent:
    return Agent(player_id=player_id, difficulty=difficulty, seed=seed)