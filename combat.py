"""
Orbit Wars - Combat Resolver
Handles multi-player fleet arrival, attacker vs attacker, then attacker vs garrison.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Arrival:
    owner: int
    ships: int


def resolve_arrivals(planet, arrivals: List[Arrival]) -> None:
    """
    Full combat resolution per spec:
    Step 1 – group attackers by owner, sum ships.
    Step 2 – attacker vs attacker: largest vs second-largest → difference survives; tie → all destroyed.
    Step 3 – survivor vs planet garrison.
    """
    if not arrivals:
        return

    # Step 1: aggregate by owner
    by_owner: Dict[int, int] = {}
    for a in arrivals:
        by_owner[a.owner] = by_owner.get(a.owner, 0) + a.ships

    # Add garrison as defender if planet is owned
    entries: List[Tuple[int, int]] = sorted(by_owner.items(), key=lambda x: -x[1])

    # Step 2: resolve inter-attacker combat
    if len(entries) == 1:
        survivor_owner, survivor_ships = entries[0]
    else:
        top_owner,  top_ships  = entries[0]
        sec_owner,  sec_ships  = entries[1]
        diff = top_ships - sec_ships
        if diff > 0:
            survivor_owner  = top_owner
            survivor_ships  = diff
        else:
            # Tie — all attackers wiped out
            return

    # Step 3: survivor vs garrison
    if survivor_owner == planet.owner:
        # Reinforcement
        planet.ships += survivor_ships
    else:
        if survivor_ships > planet.ships:
            # Capture
            planet.owner  = survivor_owner
            planet.ships  = survivor_ships - planet.ships
        else:
            # Repelled
            planet.ships -= survivor_ships