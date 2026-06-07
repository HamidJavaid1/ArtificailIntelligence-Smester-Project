"""
Orbit Wars - Renderer (Pygame)
Draws sun, planets, fleets, comets, UI panel.
Falls back gracefully if pygame is not installed.
"""

from __future__ import annotations
import math
from typing import Optional, TYPE_CHECKING

try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

if TYPE_CHECKING:
    from game_state import GameState

import config

# ── Color palette ──────────────────────────────────────────────────────────
BG        = (11, 13, 18)
SUN_C     = (245, 158,  11)
SUN_GLOW  = (251, 191,  36)
PANEL_BG  = (19,  21,  30)
NEUTRAL   = (74,  85, 104)
WHITE     = (255, 255, 255)
GRAY      = (100, 116, 139)
TEXT_MUT  = (138, 146, 164)
TEXT_HI   = (205, 214, 244)

PLAYER_COLORS = [
    (59,  130, 246),   # Blue  – human
    (249, 115,  22),   # Orange
    (34,  197,  94),   # Green
    (234, 179,   8),   # Yellow
]


def _pc(player: int):
    return PLAYER_COLORS[player % len(PLAYER_COLORS)]


class Renderer:
    """Pygame-based renderer. Instantiate once, call draw() each frame."""

    PANEL_W  = 220
    FONT_SM  = 12
    FONT_MED = 14

    def __init__(self, width: int = 1100, height: int = 700):
        if not PYGAME_OK:
            raise RuntimeError("pygame is not installed. Run: pip install pygame")
        pygame.init()
        self.W = width
        self.H = height
        self.canvas_w = width - self.PANEL_W
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Orbit Wars")
        self.font_sm  = pygame.font.SysFont("monospace", self.FONT_SM)
        self.font_med = pygame.font.SysFont("monospace", self.FONT_MED)
        self.font_lg  = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_xl  = pygame.font.SysFont("monospace", 26, bold=True)
        self._star_surf: Optional[pygame.Surface] = None
        self._stars = []

    def generate_stars(self, seed=42):
        import random
        r = random.Random(seed)
        self._stars = [(r.random()*self.canvas_w, r.random()*self.H,
                        r.random()*1.2+.3, r.random()) for _ in range(200)]

    # ── Coordinate transform ───────────────────────────────────────────────

    def world2screen(self, wx: float, wy: float):
        scale  = min(self.canvas_w, self.H) / 100.0
        ox     = (self.canvas_w - 100 * scale) / 2
        oy     = (self.H        - 100 * scale) / 2
        return int(ox + wx * scale), int(oy + wy * scale), scale

    def screen2world(self, sx: int, sy: int):
        scale  = min(self.canvas_w, self.H) / 100.0
        ox     = (self.canvas_w - 100 * scale) / 2
        oy     = (self.H        - 100 * scale) / 2
        return (sx - ox) / scale, (sy - oy) / scale

    # ── Main draw ──────────────────────────────────────────────────────────

    def draw(
        self,
        state: "GameState",
        selected_planet=None,
        aim_angle: float = 0.0,
        queued: int = 0,
        message: str = "",
    ):
        surf = self.screen
        surf.fill(BG)

        self._draw_stars(surf)
        self._draw_game(surf, state, selected_planet, aim_angle)
        self._draw_panel(surf, state, queued, message)

        if message:
            self._draw_overlay(surf, message)

        pygame.display.flip()

    # ── Stars ──────────────────────────────────────────────────────────────

    def _draw_stars(self, surf):
        for (sx, sy, r, a) in self._stars:
            alpha = int(a * 200)
            c = (alpha, alpha, alpha)
            pygame.draw.circle(surf, c, (int(sx), int(sy)), max(1, int(r)))

    # ── Game world ─────────────────────────────────────────────────────────

    def _draw_game(self, surf, state, selected_planet, aim_angle):
        # Sun glow
        cx, cy, sc = self.world2screen(config.SUN_X, config.SUN_Y)
        glow_r = int(config.SUN_R * sc * 3)
        glow_surf = pygame.Surface((glow_r*2, glow_r*2), pygame.SRCALPHA)
        for i in range(glow_r, 0, -4):
            alpha = max(0, int(60 * (1 - i/glow_r)))
            pygame.draw.circle(glow_surf, (*SUN_GLOW, alpha), (glow_r, glow_r), i)
        surf.blit(glow_surf, (cx - glow_r, cy - glow_r))

        # Orbit rings
        for p in state.planets:
            if p.orbits:
                r = int(p.orbit_r * sc)
                pygame.draw.circle(surf, (255,255,255,15), (cx, cy), r, 1)

        # Sun
        sun_r = int(config.SUN_R * sc)
        pygame.draw.circle(surf, SUN_C, (cx, cy), sun_r)
        pygame.draw.circle(surf, SUN_GLOW, (cx, cy), sun_r, 2)

        # Fleets
        for f in state.fleets:
            fx, fy, _ = self.world2screen(f.x, f.y)
            col = _pc(f.owner)
            self._draw_arrow(surf, fx, fy, f.angle, 7, col)
            if f.ships >= 5:
                lbl = self.font_sm.render(str(f.ships), True, col)
                surf.blit(lbl, (fx - lbl.get_width()//2, fy - 14))

        # Planets
        for p in state.planets:
            px, py, _ = self.world2screen(p.x, p.y)
            pr = max(4, int(p.radius * sc))
            col = _pc(p.owner) if p.owner >= 0 else NEUTRAL
            is_sel = selected_planet and p.id == selected_planet.id

            if p.is_comet:
                self._draw_comet_tail(surf, p, px, py, pr, sc)

            # Glow for owned
            if p.owner >= 0:
                gs = max(1, pr * 2)
                gl = pygame.Surface((gs*2, gs*2), pygame.SRCALPHA)
                pygame.draw.circle(gl, (*col, 40), (gs, gs), gs)
                surf.blit(gl, (px - gs, py - gs))

            pygame.draw.circle(surf, col, (px, py), pr)
            border_col = WHITE if is_sel else col
            border_w   = 3 if is_sel else 1
            pygame.draw.circle(surf, border_col, (px, py), pr, border_w)

            # Ship count
            ship_str = str(int(p.ships))
            fs = max(8, min(14, pr))
            fn = pygame.font.SysFont("monospace", fs)
            lbl = fn.render(ship_str, True, WHITE)
            surf.blit(lbl, (px - lbl.get_width()//2, py - lbl.get_height()//2))

        # Aim line
        if selected_planet and selected_planet.owner == 0:
            spx, spy, _ = self.world2screen(selected_planet.x, selected_planet.y)
            length = 80
            ex = int(spx + math.cos(aim_angle) * length)
            ey = int(spy + math.sin(aim_angle) * length)
            pygame.draw.line(surf, (59, 130, 246), (spx, spy), (ex, ey), 2)
            # Dashes
            for t in range(0, 80, 10):
                x1 = int(spx + math.cos(aim_angle) * t)
                y1 = int(spy + math.sin(aim_angle) * t)
                x2 = int(spx + math.cos(aim_angle) * (t+5))
                y2 = int(spy + math.sin(aim_angle) * (t+5))
                pygame.draw.line(surf, (59, 130, 246), (x1, y1), (x2, y2), 1)
            self._draw_arrow(surf, ex, ey, aim_angle, 10, (59,130,246))

    def _draw_arrow(self, surf, x, y, angle, size, color):
        pts = [
            (x + math.cos(angle)*size,         y + math.sin(angle)*size),
            (x + math.cos(angle+2.4)*size*0.5, y + math.sin(angle+2.4)*size*0.5),
            (x + math.cos(angle-2.4)*size*0.5, y + math.sin(angle-2.4)*size*0.5),
        ]
        pygame.draw.polygon(surf, color, [(int(a), int(b)) for a,b in pts])

    def _draw_comet_tail(self, surf, p, px, py, pr, sc):
        tail_len = int(pr * 6)
        ta = p.angle + math.pi
        for i in range(tail_len, 0, -4):
            alpha = max(0, int(80 * (1 - i/tail_len)))
            tx = int(px + math.cos(ta)*i)
            ty = int(py + math.sin(ta)*i)
            ts = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(ts, (147, 197, 253, alpha), (3,3), 3)
            surf.blit(ts, (tx-3, ty-3))

    # ── Panel ──────────────────────────────────────────────────────────────

    def _draw_panel(self, surf, state, queued, message):
        panel_x = self.canvas_w
        pygame.draw.rect(surf, PANEL_BG, (panel_x, 0, self.PANEL_W, self.H))
        pygame.draw.line(surf, (30, 35, 52), (panel_x, 0), (panel_x, self.H), 1)

        y = 20
        def text(s, color=TEXT_HI, font=None):
            nonlocal y
            f = font or self.font_med
            lbl = f.render(s, True, color)
            surf.blit(lbl, (panel_x + 14, y))
            y += lbl.get_height() + 4

        text(f"Step {state.turn} / {config.MAX_TURNS}", TEXT_HI, self.font_lg)
        y += 6

        # Scores
        for i in range(state.num_players):
            name = f"Player {i}" if i > 0 else "You"
            ships = state.total_ships(i)
            col  = _pc(i)
            pygame.draw.rect(surf, col, (panel_x+14, y, 10, 10), border_radius=2)
            lbl = self.font_med.render(f"  {name}", True, TEXT_MUT)
            surf.blit(lbl, (panel_x+14, y))
            vbl = self.font_med.render(str(ships), True, col)
            surf.blit(vbl, (panel_x + self.PANEL_W - 14 - vbl.get_width(), y))
            y += 18

        y += 8
        text(f"Queued orders: {queued}", TEXT_MUT, self.font_sm)
        y += 8

        # Key hints
        hints = [
            "── Controls ──────────────",
            "SPACE     step one turn",
            "A         auto-play toggle",
            "R         reset game",
            "",
            "Click your planet → menu",
            "Drag canvas → set angle",
            "Click enemy → aim at it",
        ]
        for h in hints:
            lbl = self.font_sm.render(h, True, TEXT_MUT if not h.startswith("──") else GRAY)
            surf.blit(lbl, (panel_x + 14, y))
            y += 16

    def _draw_overlay(self, surf, message):
        ov = pygame.Surface((self.canvas_w, self.H), pygame.SRCALPHA)
        ov.fill((11, 13, 18, 180))
        surf.blit(ov, (0, 0))

        lines = message.split("\n")
        cy = self.H // 2 - len(lines) * 20
        for line in lines:
            if not line.strip():
                cy += 20; continue
            font = self.font_xl if cy == self.H // 2 - len(lines) * 20 else self.font_lg
            lbl  = font.render(line, True, WHITE)
            surf.blit(lbl, (self.canvas_w//2 - lbl.get_width()//2, cy))
            cy += 36

        hint = self.font_med.render("Press R to restart or Q to quit", True, TEXT_MUT)
        surf.blit(hint, (self.canvas_w//2 - hint.get_width()//2, cy + 20))