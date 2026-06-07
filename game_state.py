"""
Orbit Wars - Game State
Planet, Fleet, Comet data structures and the central GameState manager.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import config
import physics


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Planet:
    id: int
    owner: int          # -1 = neutral, 0..N-1 = player index
    x: float
    y: float
    radius: float
    ships: float        # float for fractional accumulation; display as int
    production: int     # ships produced per turn when owned
    orbits: bool
    angle: float        # current angle in radians (only used when orbiting)
    ang_vel: float      # radians per turn (positive = CCW)
    orbit_r: float      # orbital radius from sun center
    is_comet: bool = False
    ttl: int = 0        # turns until expiry (comets only)

    def to_obs(self) -> list:
        """Serialize for agent observation."""
        return [self.id, self.owner, self.x, self.y,
                self.radius, int(self.ships), self.production]


@dataclass
class Fleet:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int
    speed: float

    def to_obs(self) -> list:
        return [self.id, self.owner, self.x, self.y,
                self.angle, self.from_planet_id, self.ships]


# ─────────────────────────────────────────────────────────────────────────────
# GameState
# ─────────────────────────────────────────────────────────────────────────────

class GameState:
    """Central game state. Manages all entities and exposes observation dict."""

    def __init__(self, num_players: int = 2, seed: Optional[int] = None):
        self.num_players   = num_players
        self.turn          = 0
        self.over          = False
        self.winner        = -1

        self._rng          = random.Random(seed)
        self._planet_id    = 0
        self._fleet_id     = 0
        self._comet_id     = 5000

        self.planets:  List[Planet] = []
        self.fleets:   List[Fleet]  = []
        self.comets:   List[Planet] = []   # subset of planets
        self.initial_planets: List[Planet] = []

        self._generate_planets()
        self.initial_planets = list(self.planets)

    # ── Planet generation ────────────────────────────────────────────────────

    def _rng_float(self, lo: float, hi: float) -> float:
        return lo + self._rng.random() * (hi - lo)

    def _rng_int(self, lo: int, hi: int) -> int:
        return self._rng.randint(lo, hi)

    def _make_planet(
        self, owner: int, x: float, y: float,
        production: int, ships: float,
        is_comet: bool = False, ttl: int = 0
    ) -> Planet:
        r = config.PLANET_RADIUS_BASE + math.log(production) if config.PLANET_RADIUS_LOG else config.PLANET_RADIUS_BASE
        od = physics.distance(x, y, config.SUN_X, config.SUN_Y)
        orbits = physics.should_orbit(x, y, r, config.SUN_X, config.SUN_Y, config.SUN_R)
        angle  = math.atan2(y - config.SUN_Y, x - config.SUN_X)
        direction = 1 if self._rng.random() < 0.5 else -1
        ang_vel = direction * self._rng_float(config.ORBIT_ANG_VEL_MIN, config.ORBIT_ANG_VEL_MAX)

        p = Planet(
            id=self._planet_id, owner=owner,
            x=x, y=y, radius=r, ships=ships,
            production=production,
            orbits=orbits, angle=angle,
            ang_vel=ang_vel, orbit_r=od,
            is_comet=is_comet, ttl=ttl,
        )
        self._planet_id += 1
        return p

    def _no_collide(self, x: float, y: float, r: float, min_gap: float = 4.0) -> bool:
        if physics.distance(x, y, config.SUN_X, config.SUN_Y) < config.SUN_R + r + 3:
            return False
        if x < 3 or x > 97 or y < 3 or y > 97:
            return False
        for p in self.planets:
            if physics.distance(x, y, p.x, p.y) < p.radius + r + min_gap:
                return False
        return True

    def _generate_planets(self):
        n = self.num_players

        # Home planet angles — symmetric
        if n == 2:
            home_angles = [math.pi * 0.25, math.pi * 1.25]
        elif n == 3:
            home_angles = [0, math.pi * 2 / 3, math.pi * 4 / 3]
        else:
            home_angles = [math.pi * 0.25, math.pi * 0.75, math.pi * 1.25, math.pi * 1.75]

        for i, a in enumerate(home_angles):
            r2 = config.HOME_ORBIT_RADIUS + self._rng_float(-2, 2)
            x  = config.SUN_X + math.cos(a) * r2
            y  = config.SUN_Y + math.sin(a) * r2
            prod = self._rng_int(2, 4)
            p = self._make_planet(i, x, y, prod, config.HOME_SHIPS)
            self.planets.append(p)

        # Neutral planets
        attempts = 0
        while len(self.planets) < config.NUM_PLANETS and attempts < 5000:
            attempts += 1
            x    = self._rng_float(3, 97)
            y    = self._rng_float(3, 97)
            prod = self._rng_int(config.PRODUCTION_MIN, config.PRODUCTION_MAX)
            r    = config.PLANET_RADIUS_BASE + math.log(prod)
            if not self._no_collide(x, y, r):
                continue
            ships = self._rng_int(config.NEUTRAL_INIT_MIN, config.NEUTRAL_INIT_MAX)
            p = self._make_planet(-1, x, y, prod, ships)
            self.planets.append(p)

    # ── Fleet launching ──────────────────────────────────────────────────────

    def launch_fleet(
        self, owner: int, planet: Planet,
        angle: float, ships: int
    ) -> Optional[Fleet]:
        """Launch a fleet. Returns Fleet or None if invalid."""
        if planet.owner != owner:
            return None
        ships = int(min(ships, int(planet.ships) - 1))
        if ships < 1:
            return None

        planet.ships -= ships
        sx, sy = physics.fleet_spawn_pos(planet.x, planet.y, planet.radius, angle)
        f = Fleet(
            id=self._fleet_id, owner=owner,
            x=sx, y=sy, angle=angle,
            from_planet_id=planet.id,
            ships=ships,
            speed=physics.fleet_speed(ships, config.FLEET_MAX_SPEED, config.FLEET_MIN_SPEED),
        )
        self._fleet_id += 1
        self.fleets.append(f)
        return f

    # ── Comet spawning ───────────────────────────────────────────────────────

    def spawn_comets(self):
        base_angles = [0, math.pi / 2, math.pi, math.pi * 1.5]
        for ba in base_angles:
            a   = ba + (self._rng.random() - 0.5) * 0.3
            orr = self._rng_float(config.COMET_ORBIT_MIN, config.COMET_ORBIT_MAX)
            x   = config.SUN_X + math.cos(a) * orr
            y   = config.SUN_Y + math.sin(a) * orr
            r   = config.COMET_RADIUS
            direction = 1 if self._rng.random() < 0.5 else -1
            ang_vel = direction * self._rng_float(config.COMET_ANG_VEL_MIN, config.COMET_ANG_VEL_MAX)

            c = Planet(
                id=self._comet_id, owner=-1,
                x=x, y=y, radius=r, ships=0,
                production=config.COMET_PRODUCTION,
                orbits=True, angle=a,
                ang_vel=ang_vel, orbit_r=orr,
                is_comet=True, ttl=config.COMET_TTL,
            )
            self._comet_id += 1
            self.planets.append(c)
            self.comets.append(c)

    # ── Observation dict ─────────────────────────────────────────────────────

    def observation(self, player: int) -> dict:
        """Return full observation for a given player (matches spec)."""
        return {
            "planets":              [p.to_obs() for p in self.planets],
            "fleets":               [f.to_obs() for f in self.fleets],
            "player":               player,
            "angular_velocity":     config.ORBIT_ANG_VEL_MIN,
            "initial_planets":      [p.to_obs() for p in self.initial_planets],
            "comets":               [p.to_obs() for p in self.comets],
            "comet_planet_ids":     [p.id for p in self.comets],
            "remainingOverageTime": max(0.0, float(config.MAX_TURNS - self.turn)),
            "turn":                 self.turn,
        }

    # ── Scoring ──────────────────────────────────────────────────────────────

    def total_ships(self, player: int) -> int:
        s = sum(int(p.ships) for p in self.planets if p.owner == player)
        s += sum(f.ships for f in self.fleets if f.owner == player)
        return s

    def active_players(self) -> List[int]:
        alive = set()
        for p in self.planets:
            if p.owner >= 0:
                alive.add(p.owner)
        for f in self.fleets:
            alive.add(f.owner)
        return [i for i in range(self.num_players) if i in alive]

    def scores(self) -> Dict[int, int]:
        return {i: self.total_ships(i) for i in range(self.num_players)}