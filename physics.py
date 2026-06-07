"""
Orbit Wars - Physics Engine
Handles continuous movement, orbital mechanics, and collision detection.
All collision uses line-segment intersection (not point checks).
"""

import math
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Vector helpers
# ─────────────────────────────────────────────────────────────────────────────

def distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def angle_to(x1: float, y1: float, x2: float, y2: float) -> float:
    """Angle in radians from point 1 → point 2."""
    return math.atan2(y2 - y1, x2 - x1)


def normalize_angle(a: float) -> float:
    """Wrap angle to [0, 2π)."""
    return a % (2 * math.pi)


def point_on_segment_closest(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float
) -> Tuple[float, float]:
    """Closest point on segment AB to point P."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return ax, ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return ax + t * dx, ay + t * dy


# ─────────────────────────────────────────────────────────────────────────────
# Collision detection
# ─────────────────────────────────────────────────────────────────────────────

def line_circle_intersect(
    x1: float, y1: float,
    x2: float, y2: float,
    cx: float, cy: float,
    cr: float
) -> bool:
    """
    True if line segment (x1,y1)→(x2,y2) intersects circle (cx,cy,cr).
    Uses quadratic formula on parametric line equation.
    """
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy

    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - cr * cr

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        return False

    disc_sqrt = math.sqrt(discriminant)
    t1 = (-b - disc_sqrt) / (2.0 * a)
    t2 = (-b + disc_sqrt) / (2.0 * a)

    # Segment hits circle if either root is in [0,1], or circle straddles segment
    if 0.0 <= t1 <= 1.0 or 0.0 <= t2 <= 1.0:
        return True
    if t1 < 0.0 < t2:          # segment starts inside circle
        return True
    return False


def fleet_hits_sun(
    x1: float, y1: float,
    x2: float, y2: float,
    sun_x: float, sun_y: float,
    sun_r: float
) -> bool:
    return line_circle_intersect(x1, y1, x2, y2, sun_x, sun_y, sun_r)


def fleet_hits_planet(
    x1: float, y1: float,
    x2: float, y2: float,
    planet_x: float, planet_y: float,
    planet_r: float
) -> bool:
    return line_circle_intersect(x1, y1, x2, y2, planet_x, planet_y, planet_r)


def out_of_bounds(x: float, y: float, w: float = 100.0, h: float = 100.0) -> bool:
    return x < 0 or x > w or y < 0 or y > h


# ─────────────────────────────────────────────────────────────────────────────
# Fleet speed formula
# ─────────────────────────────────────────────────────────────────────────────

def fleet_speed(ships: int, max_speed: float = 6.0, min_speed: float = 1.0) -> float:
    """
    Speed scales with fleet size via log curve.
    1 ship → min_speed, 1000 ships → max_speed (approximately).
    """
    if ships <= 1:
        return min_speed
    ships = min(ships, 1000)
    t = (math.log(ships) / math.log(1000)) ** 1.5
    return min_speed + (max_speed - min_speed) * t


# ─────────────────────────────────────────────────────────────────────────────
# Orbital mechanics
# ─────────────────────────────────────────────────────────────────────────────

def orbit_position(
    center_x: float, center_y: float,
    radius: float, angle: float
) -> Tuple[float, float]:
    """Compute x,y of an orbiting body given its current angle."""
    return (
        center_x + math.cos(angle) * radius,
        center_y + math.sin(angle) * radius,
    )


def should_orbit(
    planet_x: float, planet_y: float, planet_r: float,
    sun_x: float, sun_y: float, sun_r: float
) -> bool:
    """
    A planet orbits if it's inside the orbital zone:
    orbital_radius + planet_radius < 50 AND far enough from sun surface.
    """
    od = distance(planet_x, planet_y, sun_x, sun_y)
    return (od + planet_r < 50.0) and (od > sun_r + planet_r + 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Fleet spawn position
# ─────────────────────────────────────────────────────────────────────────────

def fleet_spawn_pos(
    planet_x: float, planet_y: float,
    planet_r: float, angle: float,
    offset: float = 0.6
) -> Tuple[float, float]:
    """Spawn fleet just outside planet radius."""
    r = planet_r + offset
    return (
        planet_x + math.cos(angle) * r,
        planet_y + math.sin(angle) * r,
    )