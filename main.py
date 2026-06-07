"""
Orbit Wars - Main Entry Point
Run this file to play the game:
    python main.py
    python main.py --players 3 --difficulty hard --seed 42
"""

from __future__ import annotations
import sys
import math
import argparse
import time
from typing import List, Optional

import config
from engine import Engine
from agent import make_agent

try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Orbit Wars")
    p.add_argument("--players",    type=int, default=2,        choices=[2,3,4], help="Number of players (1 human + AI)")
    p.add_argument("--difficulty", type=str, default="medium", choices=["easy","medium","hard"])
    p.add_argument("--seed",       type=int, default=None,     help="Random seed for reproducible maps")
    p.add_argument("--headless",   action="store_true",        help="Run headless simulation (no GUI)")
    p.add_argument("--auto-speed", type=float, default=0.0,    help="Seconds per turn in headless mode")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Headless runner (for testing / submission evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def run_headless(num_players: int, difficulty: str, seed: Optional[int], delay: float = 0.0):
    """Run a full game simulation without graphics, print final scores."""
    print(f"\n{'='*50}")
    print(f"  ORBIT WARS — Headless Simulation")
    print(f"  Players: {num_players}  |  Difficulty: {difficulty}  |  Seed: {seed}")
    print(f"{'='*50}\n")

    engine = Engine(num_players=num_players, seed=seed)
    agents = {i: make_agent(i, difficulty) for i in range(num_players)}

    while not engine.is_over():
        actions = {}
        for i in range(num_players):
            obs = engine.observation(i)
            acts = agents[i].act(obs)
            actions[i] = [(a[0], a[1], a[2]) for a in acts]

        engine.step(actions)

        if delay > 0:
            time.sleep(delay)

        t = engine.state.turn
        if t % 50 == 0:
            sc = engine.scores()
            score_str = "  ".join(f"P{i}:{v}" for i,v in sc.items())
            print(f"  Turn {t:3d}  |  {score_str}")

    sc = engine.scores()
    w  = engine.winner()
    print(f"\n{'─'*50}")
    print(f"  GAME OVER at turn {engine.state.turn}")
    print(f"  Winner: Player {w}")
    print(f"  Final scores:")
    for i, s in sc.items():
        marker = " ◀ WINNER" if i == w else ""
        print(f"    Player {i}: {s:4d} ships{marker}")
    print(f"{'─'*50}\n")
    return w, sc


# ─────────────────────────────────────────────────────────────────────────────
# Pygame game loop
# ─────────────────────────────────────────────────────────────────────────────

def run_game(num_players: int, difficulty: str, seed: Optional[int]):
    if not PYGAME_OK:
        print("pygame not found. Install with:  pip install pygame")
        print("Running in headless mode instead...\n")
        run_headless(num_players, difficulty, seed)
        return

    from render import Renderer

    renderer = Renderer(width=1100, height=700)
    renderer.generate_stars(seed or 0)

    def new_game():
        nonlocal engine, agents, selected_planet, aim_angle, queued_orders, auto_play, game_msg
        engine          = Engine(num_players=num_players, seed=seed)
        agents          = {i: make_agent(i, difficulty) for i in range(1, num_players)}
        selected_planet = None
        aim_angle       = 0.0
        queued_orders   = []
        auto_play       = False
        game_msg        = ""

    engine          = None
    agents          = {}
    selected_planet = None
    aim_angle       = 0.0
    queued_orders:  List = []
    auto_play       = False
    game_msg        = ""
    new_game()

    clock   = pygame.time.Clock()
    running = True

    def do_step():
        nonlocal game_msg
        if engine.is_over():
            return

        # Collect player 0 actions
        player_actions = {}
        player_actions[0] = [(o['planet_id'], o['angle'], o['ships'])
                              for o in queued_orders]
        queued_orders.clear()

        # AI actions
        for i, agent in agents.items():
            obs  = engine.observation(i)
            acts = agent.act(obs)
            player_actions[i] = [(a[0], a[1], a[2]) for a in acts]

        still_running = engine.step(player_actions)

        if not still_running:
            sc = engine.scores()
            w  = engine.winner()
            lines = ["GAME OVER", ""]
            for i in range(num_players):
                marker = "◀ WIN" if i == w else "     "
                name   = "You" if i == 0 else f"AI {i}"
                lines.append(f"{marker}  {name}: {sc[i]} ships")
            game_msg = "\n".join(lines)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    do_step()
                elif event.key == pygame.K_a:
                    auto_play = not auto_play
                elif event.key == pygame.K_r:
                    new_game()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if engine.is_over():
                    continue
                mx, my = pygame.mouse.get_pos()
                if mx >= renderer.canvas_w:
                    continue  # clicked panel

                wx, wy = renderer.screen2world(mx, my)

                # Check planet click
                clicked = None
                for p in engine.state.planets:
                    from physics import distance
                    if distance(wx, wy, p.x, p.y) < p.radius + 1.5:
                        clicked = p
                        break

                if clicked:
                    if clicked.owner == 0:
                        selected_planet = clicked
                    elif selected_planet and selected_planet.owner == 0:
                        # Aim at clicked planet
                        aim_angle = math.atan2(
                            clicked.y - selected_planet.y,
                            clicked.x - selected_planet.x,
                        )
                else:
                    # Aim in direction of click
                    if selected_planet and selected_planet.owner == 0:
                        aim_angle = math.atan2(
                            wy - selected_planet.y,
                            wx - selected_planet.x,
                        )

            elif event.type == pygame.MOUSEMOTION:
                if engine.is_over():
                    continue
                if pygame.mouse.get_pressed()[0]:  # left drag
                    mx, my = pygame.mouse.get_pos()
                    if mx < renderer.canvas_w and selected_planet and selected_planet.owner == 0:
                        wx, wy = renderer.screen2world(mx, my)
                        aim_angle = math.atan2(
                            wy - selected_planet.y,
                            wx - selected_planet.x,
                        )

            elif event.type == pygame.MOUSEWHEEL:
                # Scroll to change ship count (future enhancement)
                pass

        # Auto-play
        if auto_play and not engine.is_over():
            do_step()

        # Handle right-click = queue launch
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RETURN] and selected_planet and selected_planet.owner == 0:
            ships = max(1, int(selected_planet.ships) // 2)
            queued_orders.append({
                'planet_id': selected_planet.id,
                'angle':     aim_angle,
                'ships':     ships,
            })

        renderer.draw(
            state=engine.state,
            selected_planet=selected_planet,
            aim_angle=aim_angle,
            queued=len(queued_orders),
            message=game_msg,
        )
        clock.tick(60)

    pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    if args.headless:
        run_headless(args.players, args.difficulty, args.seed, args.auto_speed)
    else:
        run_game(args.players, args.difficulty, args.seed)