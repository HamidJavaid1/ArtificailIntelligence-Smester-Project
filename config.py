"""
Orbit Wars - Configuration & Constants
All game parameters in one place. Edit here to tune the game.
"""

# ── World ──────────────────────────────────────────────────────────────────────
WORLD_W = 100
WORLD_H = 100
SUN_X   = 50.0
SUN_Y   = 50.0
SUN_R   = 10.0

# ── Game length ────────────────────────────────────────────────────────────────
MAX_TURNS = 500

# ── Planets ───────────────────────────────────────────────────────────────────
NUM_PLANETS        = 20          # total planets (including home planets)
NEUTRAL_INIT_MIN   = 5
NEUTRAL_INIT_MAX   = 50
HOME_SHIPS         = 10
HOME_ORBIT_RADIUS  = 34.0        # distance from sun for starting planets
PRODUCTION_MIN     = 1
PRODUCTION_MAX     = 5
PLANET_RADIUS_BASE = 1.0
PLANET_RADIUS_LOG  = True        # radius = 1 + ln(production)

# ── Orbital mechanics ─────────────────────────────────────────────────────────
ORBIT_ANG_VEL_MIN  = 0.025      # rad/turn
ORBIT_ANG_VEL_MAX  = 0.050

# ── Fleets ───────────────────────────────────────────────────────────────────
FLEET_MAX_SPEED    = 6.0
FLEET_MIN_SPEED    = 1.0

# ── Comets ───────────────────────────────────────────────────────────────────
COMET_SPAWN_TURNS  = [50, 150, 250, 350, 450]
COMET_GROUP_SIZE   = 4           # spawned per wave (mirror symmetry)
COMET_RADIUS       = 1.0
COMET_PRODUCTION   = 1
COMET_SPEED        = 4.0
COMET_TTL          = 180         # turns before expiry
COMET_ORBIT_MIN    = 12.0
COMET_ORBIT_MAX    = 30.0
COMET_ANG_VEL_MIN  = 0.040
COMET_ANG_VEL_MAX  = 0.070

# ── Players ───────────────────────────────────────────────────────────────────
MAX_PLAYERS        = 4
PLAYER_COLORS      = ["Blue", "Orange", "Green", "Yellow"]

# ── Simulation ────────────────────────────────────────────────────────────────
DEFAULT_SEED       = None        # None = random