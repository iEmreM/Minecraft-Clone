"""World generation: climate noise -> biome -> column, then caves, ores, trees
and villages.

The shape is lifted from the real game's `data/minecraft/worldgen/` (see
`referans.md` §3), reduced to what a heightmap generator can carry:

    continentalness -> how far inland          -> base height
    erosion         -> how mountainous         -> how much relief
    weirdness       -> folded into peaks/valleys -> where the relief goes
    temperature, humidity                      -> which biome sits on it

That triple is the whole reason "plains" and "mountains" are not two separate
noises here either: they are the same noise read at a different erosion band.
`weirdness` does double duty — folded (`1 - |3|w| - 2|`) it gives peaks and
valleys, and its *zero crossings* are winding curves, which is where the rivers
go.

**Everything is sampled on a coarse grid and interpolated**, exactly as
`noise_settings/overworld.json` does with `size_horizontal: 1` (4-block cells):
the climate fields have wavelengths of hundreds of blocks, so evaluating them
per column was paying 256 noise calls for information that changes 25 times over
a chunk. Caves are the same trick in 3D and it matters far more there — a 3D
threshold per block was the single most expensive thing the old generator did.

Heights are deliberately compressed against the real game's (sea level 32, not
63). Mesh cost scales with a chunk's highest block, so a taller world is a
slower one for nothing: 60 blocks of usable relief above the water reads the
same as 190 when a block is a block.

The numbers in the SHAPE section are calibration, not derivation — they set how
big a continent is, how far apart biomes sit and how often a river cuts through.
Editing them and seeing nothing change means numba served a cached kernel:
`@njit(cache=True)` freezes module globals into the artifact and does not
invalidate it when they change. Delete `world/__pycache__`.
"""

import math

import numpy as np
from numba import njit

from world.blocks import BLOCK_DTYPE
from world.fast_noise import fast_noise2, fast_noise3, fast_rand

# ---------------------------------------------------------------------------
# Block ids
# ---------------------------------------------------------------------------
# ponytail: still duplicated from world/blocks.py (todo D-7). Numba reads these
# as module-level constants, so folding them in means blocks.py exporting plain
# ints; the assert at the bottom of blocks.py is what keeps them honest.
AIR = 0
GRASS = 1
DIRT = 2
STONE = 3
SAND = 4
SNOW = 5
LEAVES = 6
WOOD = 7
WATER = 8
BRICKS = 10
COBBLESTONE = 11
GRAVEL = 12
DEEPSLATE = 16
CLAY = 20
RED_SAND = 21
SANDSTONE = 22
RED_SANDSTONE = 23
PODZOL = 24
MUD = 27
BEDROCK = 31
BIRCH_LOG = 38
SPRUCE_LOG = 39
JUNGLE_LOG = 40
ACACIA_LOG = 41
DARK_OAK_LOG = 42
OAK_PLANKS = 44
SPRUCE_PLANKS = 46
ACACIA_PLANKS = 48
BIRCH_LEAVES = 54
SPRUCE_LEAVES = 55
SNOWY_SPRUCE_LEAVES = 359
JUNGLE_LEAVES = 56
ACACIA_LEAVES = 57
DARK_OAK_LEAVES = 58
COAL_ORE = 59
IRON_ORE = 60
COPPER_ORE = 61
GOLD_ORE = 62
REDSTONE_ORE = 63
LAPIS_ORE = 64
DIAMOND_ORE = 65
EMERALD_ORE = 66
TERRACOTTA = 91
CRAFTING_TABLE = 93
HAY_BALE = 101
COARSE_DIRT = 125
SNOWY_GRASS = 128
ANDESITE = 13
DIORITE = 14
GRANITE = 15
TUFF = 18
CALCITE = 19
PACKED_ICE = 28
MOSSY_COBBLESTONE = 76
SMOOTH_SANDSTONE = 171
CHERRY_LOG = 43
CHERRY_LEAVES = 147
AZALEA_LEAVES = 150
STRIPPED_OAK_LOG = 138
STRIPPED_SPRUCE_LOG = 140
BOOKSHELF = 95
PUMPKIN = 97
MELON = 100
GLOWSTONE = 102
BARREL = 183
CUT_SANDSTONE = 172
BROWN_MUSHROOM_BLOCK = 197
RED_MUSHROOM_BLOCK = 198
BONE_BLOCK = 127
TUBE_CORAL = 285
BRAIN_CORAL = 286
BUBBLE_CORAL = 287
FIRE_CORAL = 288
HORN_CORAL = 289
WHITE_WOOL = 109
RED_WOOL = 114
ORANGE_WOOL = 115
LIGHT_BLUE_WOOL = 120
DEEP_COAL_ORE = 151
DEEP_IRON_ORE = 152
DEEP_COPPER_ORE = 153
DEEP_GOLD_ORE = 154
DEEP_REDSTONE_ORE = 155
DEEP_LAPIS_ORE = 156
DEEP_DIAMOND_ORE = 157
DEEP_EMERALD_ORE = 158
WHITE_TERRACOTTA = 215
BROWN_TERRACOTTA = 219
RED_TERRACOTTA = 220
ORANGE_TERRACOTTA = 221
YELLOW_TERRACOTTA = 222
GLASS = 256
# Ground cover — the non-cube blocks (world/shapes.py). Everything from here
# down is drawn by the mesher's shape path and walked straight through.
SHORT_GRASS = 400
FERN = 401
DEAD_BUSH = 402
TALL_GRASS = 403
TALL_GRASS_TOP = 404
LARGE_FERN = 405
LARGE_FERN_TOP = 406
DANDELION = 407
POPPY = 408
BLUE_ORCHID = 409
ALLIUM = 410
AZURE_BLUET = 411
RED_TULIP = 412
ORANGE_TULIP = 413
WHITE_TULIP = 414
PINK_TULIP = 415
OXEYE_DAISY = 416
CORNFLOWER = 417
LILY_OF_THE_VALLEY = 418
SUNFLOWER = 421
SUNFLOWER_TOP = 422
LILAC = 423
LILAC_TOP = 424
ROSE_BUSH = 425
ROSE_BUSH_TOP = 426
PEONY = 427
PEONY_TOP = 428
BROWN_MUSHROOM = 429
RED_MUSHROOM = 430
WHEAT = 445
CARROTS = 446
POTATOES = 447
BEETROOTS = 448
SUGAR_CANE = 449
CACTUS = 450
SEAGRASS = 453
SWEET_BERRY_BUSH = 455

CHUNK_SIZE = 16
CHUNK_HEIGHT = 256

# ---------------------------------------------------------------------------
# Shape — the calibration knobs
# ---------------------------------------------------------------------------
WORLD_SEED = 42

SEA_LEVEL = 32
# The water plane engine/water_surface.py draws, sitting on top of the highest
# water block (y = SEA_LEVEL - 1). Everything below it that is not solid is a
# real WATER block, so the sea has a floor you can dive to.
WATER_LINE = SEA_LEVEL - 0.05

DEEPSLATE_LEVEL = 12    # stone below this is deepslate, and so are its ores
MIN_TERRAIN = 6         # nothing generates below this, bedrock aside

# Wavelength of each climate field, in blocks, and its amplitude per octave.
# **These are the real game's own numbers.** Each field is one file under
# `worldgen/noise/`, giving a `firstOctave` and a list of amplitudes, and the
# router (`noise_settings/overworld.json`) samples all five through
# `shifted_noise` at `xz_scale: 0.25`. So the wavelength is
# `2**-firstOctave * 4` blocks:
#
#     continentalness  firstOctave -9   ->  512 * 4 = 2048
#     erosion          firstOctave -9   ->  512 * 4 = 2048
#     ridge (weird)    firstOctave -7   ->  128 * 4 =  512
#     temperature      firstOctave -10  -> 1024 * 4 = 4096
#     vegetation       firstOctave -8   ->  256 * 4 = 1024
#
# These used to be 620 and 500 for temperature and humidity, on the theory that
# the real game's would put one biome across the whole visible world. What they
# actually did was the opposite: a band edge only has to wander by a fraction of
# a wavelength to double back on itself, so at 500 blocks the map was strewn
# with biomes 20 blocks across. Measured, the median biome ran **10 blocks**
# before it changed. At the reference's scale it is an order of magnitude more.
#
# The amplitudes are not geometric — the reference switches erosion's third
# octave off entirely — so they are copied as a table rather than approximated
# by a persistence. Tails are cut where the octave's wavelength approaches the
# 4-block lattice these are sampled on, which would alias; the missing detail is
# what the domain warp below and ROUGH_* put back.
CONT_WL = 2048.0
ERO_WL = 2048.0
WEIRD_WL = 512.0
TEMP_WL = 4096.0
HUMID_WL = 1024.0

# Continentalness is the one field whose amplitudes are *not* the reference's.
# Its list there is [1,1,2,2,2,1,1,1,1] — double weight on the octaves at 512,
# 256 and 128 blocks — so the coastline's detail lives at a scale of roughly a
# hundred blocks. Measured over a 6400-block window that gives **1668 separate
# landmasses and 3185 separate bodies of water**: no continent and no ocean
# anywhere, just land and sea mottled together at exactly the scale that was
# complained about. Moved onto the long octaves it is 126 landmasses, and the
# body of water a random sea block belongs to goes from 714 blocks across to
# 1863.
#
# The reference gets away with its own weighting because its land/sea decision
# is not this spline: continentalness feeds `depth`, `factor` and `jaggedness`
# together and the surface is where a 3D density crosses zero, so its fine
# octaves texture a coast that something else has already placed. Ours is a
# heightmap and the fine octaves *are* the placement.
CONT_AMP = np.array([4.0, 3.0, 1.5, 0.7, 0.4], dtype=np.float64)
ERO_AMP = np.array([1.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)
WEIRD_AMP = np.array([1.0, 2.0, 1.0], dtype=np.float64)
TEMP_AMP = np.array([1.5, 0.0, 1.0], dtype=np.float64)
HUMID_AMP = np.array([1.0, 1.0], dtype=np.float64)

# Domain warp — the reference's `shift_x` / `shift_z`, which every one of the
# five fields is sampled through. `shift_a` reads its `offset` noise (firstOctave
# -3) at 4x the block coordinate and adds the result to the *scaled* coordinate,
# so in world units it drags the sample point by about ±4 blocks over a handful
# of blocks. It is a border ruffle and nothing more; its own wavelength is below
# what a 4-block lattice can carry, so it is stretched to 24 and 12 here.
#
# This replaced a hand-rolled jitter that added a 430-block fractal *and* a
# per-block hash to temperature and humidity directly. Warping the coordinate
# and warping the value look alike at one column and are not the same thing: a
# value jitter a third of a band wide moves a column whole bands at a time,
# which is how a plains grew 20-block deserts inside it and why every border was
# a 10-block salt-and-pepper of two top blocks. A coordinate warp cannot do
# that — it moves *where* the map is read, so a column still lands somewhere
# the map really says.
#
# Amplitude is load-bearing: at ±16 over 32 blocks the displacement's own
# gradient exceeds 1 and the coordinate map folds over itself, which scrambles
# height and biome alike into 6-block confetti. Measured, that is worse than
# the jitter it replaced.
WARP_WL = 24.0
WARP_AMP = np.array([1.0, 1.0], dtype=np.float64)
WARP_DIST = 5.0

# The ridged multifractal that carries all the relief, per column, never
# interpolated — this is the field that has to resolve to the block. GAIN is how
# strongly one octave's crest feeds the next; below ~1.2 the feedback stops
# mattering and it degenerates into ordinary fBm. MID and WEIGHT map its
# roughly-[0,1] output onto "how far above the base height", and VALLEY_BIAS
# then cuts the downside so ranges rise out of a plain instead of trenching it.
#
# The wavelength went 230 -> 420 when the relief amplitude was doubled. The two
# are one knob: a ridge system twice as tall on the same footprint is not a
# bigger mountain range, it is the same range with the slopes turned into
# cliffs. Four octaves off 420 still resolve detail down to ~50 blocks, which is
# the right scale of spur on a 90-block massif.
RIDGE_WL = 420.0
# How many of the four octaves the biome picker sees. See `alt` in column_height.
RIDGE_SMOOTH = 2
RIDGE_GAIN = 1.9
RIDGE_MID = 0.34
RIDGE_WEIGHT = 1.35
PV_WEIGHT = 0.40
VALLEY_BIAS = 0.35

# A short-wavelength term at a *fixed* amplitude, added on top of everything
# else so that flat ground still has some. Without it the world came out as a
# contour map: a heightmap this smooth rounds to the same integer across ten or
# more blocks, so every slope wore concentric one-block terraces and a beach
# looked like a topographic survey.
#
# It used to be twice this amplitude at 15 blocks with a second octave at 7, and
# that is its own artefact — a 7-block wave about a block tall rounds to a field
# of isolated one-block pips, which is what a desert and a village green were
# covered in. The ridged relief now carries the fine detail that the second
# octave was standing in for, so what is left here only has to be broad enough
# to tilt the ground off the contour: one wave over 24 blocks, with a weak
# harmonic that frays step edges without ever reaching a whole block itself.
ROUGH_WL = 24.0
ROUGH_AMP = 1.5

# Surface patches — see B_PATCH_A. These used to cover about a fifth of the
# ground on a 26-block field, and a fifth is not a scattering: with two alternate
# blocks always on, every biome came out as a three-colour quilt rather than as
# ground with something on it. The reference's disks are rare and far apart, so
# the thresholds moved out to about a fifteenth of the ground and the field
# lengthened, which also makes each patch a patch rather than a speckle.
PATCH_WL = 40.0
PATCH_A_T = 0.60
PATCH_B_T = -0.66
PATCH_DITHER = 0.14

# Multi-octave simplex lands well inside [-1, 1]; the band edges below are the
# real game's, so the fields have to be stretched to reach them.
#
# Continentalness needs more of it than the rest, and that is an artefact of how
# the octaves are combined rather than a departure from the reference. Its
# amplitude list is the longest and puts its weight in the middle (1,1,2,2,2,1,1),
# so averaging seven octaves cancels far more than averaging two does: at the
# shared gain its standard deviation came out 0.24 against temperature's 0.44,
# and the reference's band edges — which are absolute numbers, -0.455 for deep
# ocean and -0.19 for the coast — then sat 2 standard deviations out. Deep ocean
# was 3% of the world instead of the reference's ~10%. Scaled to the same spread
# as the other fields, the bands get the shares they were written for.
CLIMATE_GAIN = 1.35
CONT_GAIN = 2.4

# Rivers are the zero crossing of weirdness, so they are curves rather than
# noise blobs. Widths are in weirdness units, not blocks: the same threshold is a
# wide slow river on flat ground and a narrow one where the field is steep.
#
# **A river is a valley with water in the bottom of it, and the two are separate
# numbers.** They used to be one: the terrain was dragged straight down to the
# bed across the width of the channel, so on any raised ground the result was a
# sheer slot with the water at the bottom — you could stand on the rim and look
# down a wall. RIVER_VALLEY is four times the channel, and the ground is eased
# into it with a smoothstep so the banks meet the land flat instead of at a
# corner. Only the inner RIVER_WIDTH is the river as far as the biome is
# concerned.
RIVER_WIDTH = 0.035
RIVER_VALLEY = 0.15
RIVER_BED = SEA_LEVEL - 3.0
RIVER_MAX_H = SEA_LEVEL + 34.0   # above this, rivers fade out instead of
                                 # slicing a canyon through a mountain
RIVER_FADE = 26.0                # blocks of height it takes to fade out — long,
                                 # or the fade is itself a step in the ground

# Caves: two noise fields, each near zero on a surface; where two surfaces cross
# you get a curve, and a thickened curve is a tunnel. A single threshold on one
# field gives blobs instead, which is what the old generator had.
CAVE_WL = 46.0
CAVE_WIDTH = 0.062
# Caves stop this far under the surface. fast_builder.SEAL_COVER (8) assumes
# every cave is roofed by at least that much rock — the far LOD fills in
# anything with less cover, so a cave mouth would close up at a distance.
CAVE_ROOF = 9
CAVE_FLOOR = 3

# Coarse sampling. PAD is how far outside the chunk the heightmap reaches, which
# is what lets a tree rooted in the next chunk drop its canopy across the seam
# instead of being skipped (the old generator's treeless strips).
PAD = 4
SPAN = CHUNK_SIZE + 2 * PAD          # 24
GRID_STEP = 4
GRID_N = SPAN // GRID_STEP + 2       # 8 — covers 0..28, one past the span
CAVE_STEP = 4

# A chunk samples its grid at world x = 16*chunk_x - PAD + 4i. CHUNK_SIZE is a
# multiple of GRID_STEP, so every chunk's grid points land on the *same* global
# lattice — which is the only reason neighbouring chunks agree about the column
# they share. Anything outside the kernel that wants the world's real height has
# to read that lattice too, not the underlying noise; see climate_interp.
GRID_ANCHOR = -PAD % GRID_STEP       # 1

# ---------------------------------------------------------------------------
# Splines — continentalness/erosion to height, straight off the shape of
# density_function/overworld/{offset,factor}.json
# ---------------------------------------------------------------------------
# Base height. The x breakpoints are the real game's continentalness bands
# (deep ocean / ocean / coast / near-inland / mid-inland / far-inland) — at
# their own values now, not shifted 0.08 toward land as they were. That shift
# bought a world that was 36% water instead of 47%, and it cost exactly what was
# asked for back: an ocean you cross in twenty seconds is a lake.
CONT_X = np.array([-1.0, -0.455, -0.19, -0.11, 0.03, 0.30, 1.0], dtype=np.float64)
CONT_Y = np.array([2.0, 10.0, 22.0, 33.0, 36.0, 40.0, 50.0], dtype=np.float64)
# How much of the erosion relief a column gets. Oceans keep a little so the
# floor is not a plate, but not enough to breach the surface out at sea.
CONT_W = np.array([0.12, 0.18, 0.28, 0.45, 0.75, 1.0, 1.0], dtype=np.float64)

# Relief amplitude by erosion, on the real game's erosion band edges. The bump
# at 0.55 is theirs too — that band is plateaus and shattered terrain, not the
# flattest ground.
#
# Roughly doubled at the rugged end. The reference's peaks stand ~190 blocks
# over its sea level out of a 384-block world; ours were reaching 40 over a
# 256-block one, so "mountain" meant a hill you could walk up without noticing.
# The flat end is untouched — a plain has to stay a plain, and only the low
# erosion bands get to be alpine, which is what makes erosion a *character*
# rather than a height multiplier.
ERO_X = np.array([-1.0, -0.78, -0.375, -0.2225, 0.05, 0.45, 0.55, 1.0], dtype=np.float64)
ERO_Y = np.array([104.0, 80.0, 42.0, 22.0, 10.0, 4.0, 16.0, 2.0], dtype=np.float64)

# How far the *floor* is lifted in a rugged region, on the same erosion bands.
# The reference's `offset` spline is nested — continents, then erosion, then
# folded ridges — so a mountainous band does not merely get more amplitude, it
# gets higher ground to be rugged on. Leaving that out is what made doubling the
# relief backfire: the valleys between the new peaks were dug below sea level,
# every continent filled with lakes, and at a 5600-block view the world had no
# oceans and no continents left — just land and water mottled together at a
# 150-block scale, which is *shorter* than the terrain features that were
# supposed to be big. Continentalness only decides where the sea is if the
# relief stays inside the height it hands out.
ERO_LIFT = np.array([46.0, 34.0, 15.0, 6.0, 2.0, 0.0, 4.0, 0.0], dtype=np.float64)

# ---------------------------------------------------------------------------
# Biomes
# ---------------------------------------------------------------------------
(OCEAN, DEEP_OCEAN, FROZEN_OCEAN, WARM_OCEAN, BEACH, SNOWY_BEACH, STONY_SHORE,
 RIVER, PLAINS, FOREST, BIRCH_FOREST, DARK_FOREST, TAIGA, SNOWY_TAIGA,
 SNOWY_PLAINS, SAVANNA, DESERT, JUNGLE, SWAMP, BADLANDS, WINDSWEPT_HILLS,
 MEADOW, GROVE, SNOWY_SLOPES, JAGGED_PEAKS, STONY_PEAKS) = range(26)

BIOME_COUNT = 26

BIOME_NAMES = [
    'Ocean', 'Deep Ocean', 'Frozen Ocean', 'Warm Ocean', 'Beach', 'Snowy Beach',
    'Stony Shore', 'River', 'Plains', 'Forest', 'Birch Forest', 'Dark Forest',
    'Taiga', 'Snowy Taiga', 'Snowy Plains', 'Savanna', 'Desert', 'Jungle',
    'Swamp', 'Badlands', 'Windswept Hills', 'Meadow', 'Grove', 'Snowy Slopes',
    'Jagged Peaks', 'Stony Peaks',
]

# The real game's temperature and humidity band edges (biome/*.json parameters).
TEMP_BANDS = np.array([-0.45, -0.15, 0.2, 0.55], dtype=np.float64)
HUMID_BANDS = np.array([-0.35, -0.1, 0.1, 0.3], dtype=np.float64)

# temperature (rows, icy -> hot) x humidity (columns, dry -> wet). This is the
# real game's middle-biome table with the variants we have no blocks to
# distinguish collapsed into their parent.
LAND_GRID = np.array([
    [SNOWY_PLAINS,  SNOWY_PLAINS, SNOWY_PLAINS, SNOWY_TAIGA,  SNOWY_TAIGA],
    [PLAINS,        PLAINS,       FOREST,       TAIGA,        TAIGA],
    [PLAINS,        PLAINS,       FOREST,       BIRCH_FOREST, DARK_FOREST],
    [SAVANNA,       SAVANNA,      PLAINS,       JUNGLE,       JUNGLE],
    [DESERT,        DESERT,       DESERT,       DESERT,       JUNGLE],
], dtype=np.int32)


def _biome_table(default, **rows):
    table = np.full(BIOME_COUNT, default, dtype=np.int32)
    for name, value in rows.items():
        table[globals()[name]] = value
    return table


# Top block of a column that stands above water.
B_TOP = _biome_table(
    GRASS,
    OCEAN=GRAVEL, DEEP_OCEAN=GRAVEL, FROZEN_OCEAN=GRAVEL, WARM_OCEAN=SAND,
    BEACH=SAND, SNOWY_BEACH=SNOW, STONY_SHORE=STONE, RIVER=SAND,
    SNOWY_TAIGA=SNOWY_GRASS, SNOWY_PLAINS=SNOWY_GRASS, DESERT=SAND,
    BADLANDS=RED_SAND, TAIGA=PODZOL, GROVE=SNOWY_GRASS, SNOWY_SLOPES=SNOW,
    JAGGED_PEAKS=SNOW, STONY_PEAKS=STONE, DARK_FOREST=GRASS,
)

# The 2-4 blocks under it.
B_FILL = _biome_table(
    DIRT,
    OCEAN=GRAVEL, DEEP_OCEAN=GRAVEL, FROZEN_OCEAN=GRAVEL, WARM_OCEAN=SAND,
    BEACH=SAND, SNOWY_BEACH=SAND, STONY_SHORE=STONE, RIVER=SAND,
    DESERT=SANDSTONE, BADLANDS=RED_SANDSTONE, SWAMP=MUD,
    JAGGED_PEAKS=STONE, STONY_PEAKS=STONE, SNOWY_SLOPES=STONE,
)

# What a column that ends up under the sea gets instead — the sea bed.
B_UNDER = _biome_table(
    GRAVEL,
    BEACH=SAND, SNOWY_BEACH=SAND, WARM_OCEAN=SAND, RIVER=SAND, DESERT=SAND,
    BADLANDS=RED_SAND, SWAMP=MUD, JUNGLE=CLAY, FOREST=DIRT, DARK_FOREST=DIRT,
    PLAINS=SAND, SAVANNA=SAND,
)

# How often a column grows a tree, per 10 000. A canopy is 5 wide, so anything
# over ~600 is a closed roof of leaves rather than a forest.
#
# The reference states these as attempts per chunk (`vegetation/trees_*.json`'s
# `count_extra`), which is 256 columns, so its number times 39 is the figure in
# this table: forest 10 -> 391, taiga 10 -> 391, savanna 1 -> 43. Two of ours
# were a long way off and both showed: plains asks for 0.05 attempts a chunk and
# had 22 (a plains with a wood in it), windswept hills asks for 0.1 and had 40.
# Jungle and dark forest go the other way — 50 and 16 attempts a chunk are 1950
# and 625 — and there we stop short of the reference on purpose, because those
# are attempts against a canopy that mostly rejects them and ours is a straight
# per-column roll that does not.
B_TREES = _biome_table(
    0,
    PLAINS=8, FOREST=380, BIRCH_FOREST=360, DARK_FOREST=450, TAIGA=340,
    SNOWY_TAIGA=260, JUNGLE=520, SAVANNA=43, SWAMP=90, MEADOW=14,
    WINDSWEPT_HILLS=15, GROVE=300, SNOWY_PLAINS=4, BADLANDS=3,
)

# ---------------------------------------------------------------------------
# Tree shapes
# ---------------------------------------------------------------------------
# Every biome draws from a *mix* of these, not from one. That is the single
# reason the old forests read as one tree stamped over and over: the reference
# does not give a biome a tree, it gives it a `random_selector` over three to
# five of them (`configured_feature/trees_*.json` — plains is 2/3 oak, 1/3
# fancy oak and a rare fallen trunk; taiga is 2/3 spruce, 1/3 pine; jungle mixes
# four). The shapes below are the reference's trunk and foliage placers:
# straight/forking/fancy/giant trunks against blob/spruce/pine/acacia/dark-oak
# crowns, each with its own per-tree ranges.
(TS_NONE, TS_OAK, TS_BIRCH, TS_SPRUCE, TS_PINE, TS_JUNGLE, TS_ACACIA,
 TS_DARK_OAK, TS_FANCY, TS_MEGA_SPRUCE, TS_MEGA_JUNGLE, TS_BUSH, TS_FALLEN,
 TS_CHERRY, TS_MUSHROOM, TS_SUPER_BIRCH, TS_SWAMP_OAK, TS_MEGA_PINE,
 TS_FLAT_MUSHROOM) = range(19)
TS_COUNT = 19

# The last four are the reference's own separate configured features, not
# variations invented here, and each is a different tree rather than a taller
# one: `super_birch_bees` (old growth birch, meadow) runs to 13 logs where the
# plain birch stops at 7; `swamp_oak` carries a radius-3 crown, which is why the
# wiki counts 137 leaves on it against a plain oak's 56; `mega_pine` is a mega
# spruce with a 3-7 crown instead of a 13-17 one, so it is a bare column with a
# tuft where the other is a cone to the ground; and `huge_brown_mushroom` is a
# flat plate where the red one is a dome.

# No shape may reach further than PAD sideways, or a canopy rooted in the next
# chunk gets clipped at the seam — the bare-strip bug test_worldgen guards.
# Branch shapes do not try to be lucky about it: `_tip_blob` sizes the ball on
# the end of a branch from what is left of the budget.
TREE_SLOTS = 6

# Weights are the reference's `random_selector` chances *after* flattening.
# Its list is walked in order and each entry is an independent coin — a 0.2 on
# the second entry only fires if the first one missed — so a table of the raw
# numbers would be wrong. `trees_birch_and_oak` reads birch 0.2, fancy oak 0.1,
# else oak; that is 20 / 8 / 72, which is what is written here.
_TREE_MIX = {
    # (weight, shape, log, leaf)
    # trees_plains: fancy oak 1/3, else oak. Plus the fallen oak the reference
    # scatters separately (`fallen_oak_tree`).
    'PLAINS':          [(65, TS_OAK, WOOD, LEAVES), (32, TS_FANCY, WOOD, LEAVES),
                        (3, TS_FALLEN, WOOD, LEAVES)],
    # trees_birch_and_oak
    'FOREST':          [(69, TS_OAK, WOOD, LEAVES), (20, TS_BIRCH, BIRCH_LOG, BIRCH_LEAVES),
                        (8, TS_FANCY, WOOD, LEAVES), (3, TS_FALLEN, WOOD, LEAVES)],
    # trees_birch is birch and nothing else — the oaks that used to be in here
    # are what stopped a birch forest reading as one.
    'BIRCH_FOREST':    [(96, TS_BIRCH, BIRCH_LOG, BIRCH_LEAVES),
                        (4, TS_FALLEN, BIRCH_LOG, BIRCH_LEAVES)],
    # dark_forest_vegetation: brown mushroom .025, red .05, dark oak 2/3,
    # birch .2, fancy oak .1, else oak -> 2.5 / 4.9 / 61.8 / 6.2 / 2.5 / 22.1.
    'DARK_FOREST':     [(62, TS_DARK_OAK, DARK_OAK_LOG, DARK_OAK_LEAVES),
                        (22, TS_OAK, WOOD, LEAVES),
                        (6, TS_BIRCH, BIRCH_LOG, BIRCH_LEAVES),
                        (5, TS_MUSHROOM, BONE_BLOCK, RED_MUSHROOM_BLOCK),
                        (3, TS_FLAT_MUSHROOM, BONE_BLOCK, BROWN_MUSHROOM_BLOCK),
                        (2, TS_FANCY, WOOD, LEAVES)],
    # trees_taiga: pine 1/3, else spruce. The megas belong to old growth taiga,
    # which we have no biome for, so they are a rarity here rather than a third
    # of the forest — one is ~550 leaves and a stand of them is a wall.
    'TAIGA':           [(60, TS_SPRUCE, SPRUCE_LOG, SPRUCE_LEAVES),
                        (32, TS_PINE, SPRUCE_LOG, SPRUCE_LEAVES),
                        (3, TS_MEGA_SPRUCE, SPRUCE_LOG, SPRUCE_LEAVES),
                        (2, TS_MEGA_PINE, SPRUCE_LOG, SPRUCE_LEAVES),
                        (3, TS_FALLEN, SPRUCE_LOG, SPRUCE_LEAVES)],
    # Everything that grows where it snows carries the snow-topped leaf. It is
    # the same tree; only the block's up-face differs, so a canopy shows white
    # exactly where it is exposed to the sky and green underneath.
    'SNOWY_TAIGA':     [(63, TS_SPRUCE, SPRUCE_LOG, SNOWY_SPRUCE_LEAVES),
                        (33, TS_PINE, SPRUCE_LOG, SNOWY_SPRUCE_LEAVES),
                        (4, TS_FALLEN, SPRUCE_LOG, SNOWY_SPRUCE_LEAVES)],
    'GROVE':           [(67, TS_SPRUCE, SPRUCE_LOG, SNOWY_SPRUCE_LEAVES),
                        (33, TS_PINE, SPRUCE_LOG, SNOWY_SPRUCE_LEAVES)],
    'SNOWY_PLAINS':    [(100, TS_SPRUCE, SPRUCE_LOG, SNOWY_SPRUCE_LEAVES)],
    # trees_jungle: fancy oak .1, bush .5, mega jungle 1/3, else jungle
    # -> 10 / 45 / 15 / 30. The bushes are what fill the floor under a canopy
    # that is otherwise all at one height.
    'JUNGLE':          [(45, TS_BUSH, JUNGLE_LOG, JUNGLE_LEAVES),
                        (30, TS_JUNGLE, JUNGLE_LOG, JUNGLE_LEAVES),
                        (15, TS_MEGA_JUNGLE, JUNGLE_LOG, JUNGLE_LEAVES),
                        (10, TS_FANCY, WOOD, LEAVES)],
    # trees_savanna: acacia .8, else oak.
    'SAVANNA':         [(80, TS_ACACIA, ACACIA_LOG, ACACIA_LEAVES),
                        (20, TS_OAK, WOOD, LEAVES)],
    # trees_swamp is the wide-crowned swamp oak and only that.
    'SWAMP':           [(87, TS_SWAMP_OAK, WOOD, LEAVES),
                        (8, TS_FALLEN, WOOD, LEAVES),
                        (5, TS_FLAT_MUSHROOM, BONE_BLOCK, BROWN_MUSHROOM_BLOCK)],
    # meadow_trees: super birch .5, else oak. Cherry groves border meadows
    # there, and cherry is the one wood set we had no other use for.
    'MEADOW':          [(45, TS_SUPER_BIRCH, BIRCH_LOG, BIRCH_LEAVES),
                        (35, TS_OAK, WOOD, LEAVES),
                        (20, TS_CHERRY, CHERRY_LOG, CHERRY_LEAVES)],
    # trees_windswept_hills: spruce .666, fancy oak .1, else oak.
    'WINDSWEPT_HILLS': [(64, TS_SPRUCE, SPRUCE_LOG, SPRUCE_LEAVES),
                        (30, TS_OAK, WOOD, LEAVES),
                        (3, TS_FANCY, WOOD, LEAVES),
                        (3, TS_FALLEN, SPRUCE_LOG, SPRUCE_LEAVES)],
    # Wooded badlands is oak on the plateaus and nothing else. The azalea bush
    # that used to be here is a lush-cave feature and grows nowhere near one.
    'BADLANDS':        [(100, TS_OAK, WOOD, LEAVES)],
}


def _tree_mix(rows):
    count = np.zeros(BIOME_COUNT, dtype=np.int32)
    shape = np.zeros((BIOME_COUNT, TREE_SLOTS), dtype=np.int32)
    log = np.zeros((BIOME_COUNT, TREE_SLOTS), dtype=np.int32)
    leaf = np.zeros((BIOME_COUNT, TREE_SLOTS), dtype=np.int32)
    cumulative = np.ones((BIOME_COUNT, TREE_SLOTS), dtype=np.float64)
    for name, variants in rows.items():
        assert len(variants) <= TREE_SLOTS, name
        biome = globals()[name]
        total = float(sum(v[0] for v in variants))
        acc = 0.0
        for i, (weight, kind, log_id, leaf_id) in enumerate(variants):
            acc += weight / total
            cumulative[biome, i] = acc
            shape[biome, i] = kind
            log[biome, i] = log_id
            leaf[biome, i] = leaf_id
        cumulative[biome, len(variants) - 1] = 1.0
        count[biome] = len(variants)
    return count, shape, log, leaf, cumulative


B_TREE_N, B_TREE_SHAPE, B_TREE_LOG, B_TREE_LEAF, B_TREE_CUM = _tree_mix(_TREE_MIX)

# Cheapest possible rejection for a column that cannot grow a tree in *any*
# biome. The padded ring outside a chunk exists only so canopies can cross the
# seam, and this hash lets 96% of it skip the climate work entirely — see the
# tree pass in generate_chunk_fast.
TREE_RATE_MAX = int(B_TREES.max())

# Boulders, per 10 000 columns — the reference's `forest_rock`, a mossy lump
# dropped on the ground in forests and mountains. Cheap, and out of proportion
# to its cost: a slope with nothing on it reads as a mesh, and a slope with two
# rocks on it reads as ground.
B_ROCKS = _biome_table(
    0,
    PLAINS=2, MEADOW=9, FOREST=6, BIRCH_FOREST=5, DARK_FOREST=7, JUNGLE=4,
    TAIGA=8, SNOWY_TAIGA=8, GROVE=10, SWAMP=4, SAVANNA=3,
    WINDSWEPT_HILLS=26, STONY_SHORE=22, SNOWY_SLOPES=18, JAGGED_PEAKS=16,
    STONY_PEAKS=24,
)
ROCK_RATE_MAX = int(B_ROCKS.max())
# What a boulder is made of, by biome family. Mossy in the wet, bare in the
# cold and high.
B_ROCK_BLOCK = _biome_table(
    MOSSY_COBBLESTONE,
    GROVE=STONE, SNOWY_SLOPES=STONE, JAGGED_PEAKS=STONE, STONY_PEAKS=ANDESITE,
    WINDSWEPT_HILLS=ANDESITE, SNOWY_TAIGA=STONE, MEADOW=GRANITE,
    SAVANNA=GRANITE, JUNGLE=MOSSY_COBBLESTONE, STONY_SHORE=DIORITE,
)

# The reference's warm oceans grow coral. We have the blocks and nothing else
# was using them.
CORALS = np.array([TUBE_CORAL, BRAIN_CORAL, BUBBLE_CORAL, FIRE_CORAL,
                   HORN_CORAL], dtype=np.int32)
assert all(B_TREES[globals()[name]] > 0 for name in _TREE_MIX), \
    "a biome has a tree mix but no tree rate, so it will never grow one"
assert all(B_TREES[b] == 0 or B_TREE_N[b] > 0 for b in range(BIOME_COUNT)), \
    "a biome has a tree rate but no mix to draw from"

# ---------------------------------------------------------------------------
# Ground cover
# ---------------------------------------------------------------------------
# The grass, ferns and flowers that go on top of the ground, and the same
# structure the trees use: a rate per 10 000 columns and a mix to draw from,
# because a biome with one plant in it reads as a texture rather than as ground.
# The mixes are the reference's `patch_*` and `flower_*` configured features —
# `flower_plains` really is the nine-flower list below, and a meadow really does
# get more flowers than anything else.
#
# One block or two: the second column is what stands on top of the first, which
# is how the two-block plants work in the real game as well (an upper and a
# lower block state). 0 means the plant is one block tall.
_PLANT_MIX = {
    # patch_grass_plain + flower_plains
    'PLAINS':          [(760, SHORT_GRASS, 0), (40, TALL_GRASS, TALL_GRASS_TOP),
                        (30, DANDELION, 0), (30, POPPY, 0),
                        (20, AZURE_BLUET, 0), (20, CORNFLOWER, 0),
                        (20, OXEYE_DAISY, 0), (40, RED_TULIP, 0),
                        (20, ORANGE_TULIP, 0), (10, WHITE_TULIP, 0),
                        (10, PINK_TULIP, 0)],
    # flower_meadow is the densest flower list in the game.
    'MEADOW':          [(560, SHORT_GRASS, 0), (60, TALL_GRASS, TALL_GRASS_TOP),
                        (70, DANDELION, 0), (70, ALLIUM, 0),
                        (70, AZURE_BLUET, 0), (60, CORNFLOWER, 0),
                        (60, OXEYE_DAISY, 0), (50, POPPY, 0)],
    'FOREST':          [(700, SHORT_GRASS, 0), (90, FERN, 0),
                        (60, TALL_GRASS, TALL_GRASS_TOP),
                        (40, LILY_OF_THE_VALLEY, 0), (40, POPPY, 0),
                        (30, DANDELION, 0), (20, RED_MUSHROOM, 0),
                        (20, BROWN_MUSHROOM, 0)],
    # The birch woods are the reference's flower forest neighbours: this is
    # where the big two-block flowers grow.
    'BIRCH_FOREST':    [(620, SHORT_GRASS, 0), (80, TALL_GRASS, TALL_GRASS_TOP),
                        (60, LILY_OF_THE_VALLEY, 0), (60, LILAC, LILAC_TOP),
                        (60, ROSE_BUSH, ROSE_BUSH_TOP), (60, PEONY, PEONY_TOP),
                        (60, SUNFLOWER, SUNFLOWER_TOP)],
    # A closed canopy is dark, so the floor is mushrooms rather than flowers.
    'DARK_FOREST':     [(600, SHORT_GRASS, 0), (150, BROWN_MUSHROOM, 0),
                        (150, RED_MUSHROOM, 0), (100, ROSE_BUSH, ROSE_BUSH_TOP)],
    'TAIGA':           [(480, FERN, 0), (240, SHORT_GRASS, 0),
                        (160, LARGE_FERN, LARGE_FERN_TOP),
                        (120, SWEET_BERRY_BUSH, 0)],
    'SNOWY_TAIGA':     [(600, FERN, 0), (300, SHORT_GRASS, 0),
                        (100, LARGE_FERN, LARGE_FERN_TOP)],
    # Grass poking up through the snow, which is what the reference's snowy
    # taiga and grove look like. Nothing above them: a snowy slope is bare snow
    # and a peak is bare rock.
    'SNOWY_PLAINS':    [(1000, SHORT_GRASS, 0)],
    'GROVE':           [(600, SHORT_GRASS, 0), (400, FERN, 0)],
    'WINDSWEPT_HILLS': [(880, SHORT_GRASS, 0), (120, TALL_GRASS, TALL_GRASS_TOP)],
    'SAVANNA':         [(820, SHORT_GRASS, 0), (180, TALL_GRASS, TALL_GRASS_TOP)],
    'JUNGLE':          [(620, SHORT_GRASS, 0), (280, FERN, 0),
                        (60, LARGE_FERN, LARGE_FERN_TOP), (40, SUGAR_CANE, SUGAR_CANE)],
    'SWAMP':           [(620, SHORT_GRASS, 0), (140, BLUE_ORCHID, 0),
                        (120, BROWN_MUSHROOM, 0), (120, SUGAR_CANE, SUGAR_CANE)],
    'DESERT':          [(700, DEAD_BUSH, 0), (300, CACTUS, CACTUS)],
    'BADLANDS':        [(1000, DEAD_BUSH, 0)],
    'RIVER':           [(700, SHORT_GRASS, 0), (300, SUGAR_CANE, SUGAR_CANE)],
    'BEACH':           [(1000, SUGAR_CANE, SUGAR_CANE)],
    # The sea floor. Seagrass is the one plant that goes in water, so it is
    # gated on the column being *under* the waterline rather than above it.
    'OCEAN':           [(1000, SEAGRASS, 0)],
    'WARM_OCEAN':      [(1000, SEAGRASS, 0)],
    'DEEP_OCEAN':      [(1000, SEAGRASS, 0)],
}

PLANT_SLOTS = 11


def _plant_mix(rows):
    count = np.zeros(BIOME_COUNT, dtype=np.int32)
    block = np.zeros((BIOME_COUNT, PLANT_SLOTS), dtype=np.int32)
    above = np.zeros((BIOME_COUNT, PLANT_SLOTS), dtype=np.int32)
    cumulative = np.ones((BIOME_COUNT, PLANT_SLOTS), dtype=np.float64)
    for name, variants in rows.items():
        assert len(variants) <= PLANT_SLOTS, name
        biome = globals()[name]
        total = float(sum(v[0] for v in variants))
        acc = 0.0
        for i, (weight, plant, top) in enumerate(variants):
            acc += weight / total
            cumulative[biome, i] = acc
            block[biome, i] = plant
            above[biome, i] = top
        cumulative[biome, len(variants) - 1] = 1.0
        count[biome] = len(variants)
    return count, block, above, cumulative


B_PLANT_N, B_PLANT_ID, B_PLANT_TOP, B_PLANT_CUM = _plant_mix(_PLANT_MIX)

# How often a column grows something, per 10 000 — the same units as B_TREES.
# A fifth of a plains is the reference's own density roughly halved: its
# `patch_grass_plain` runs 32 tries a chunk against a placement filter, ours is
# one roll per column with nothing to reject it, and every plant is 4 quads in
# the blended pass.
B_PLANTS = _biome_table(
    0,
    PLAINS=2000, MEADOW=2800, FOREST=1700, BIRCH_FOREST=1600, DARK_FOREST=900,
    TAIGA=1300, SNOWY_TAIGA=650, SNOWY_PLAINS=380,
    GROVE=450, WINDSWEPT_HILLS=650, SAVANNA=2200, JUNGLE=2600, SWAMP=1100,
    DESERT=110, BADLANDS=80, RIVER=500, BEACH=40,
    OCEAN=900, WARM_OCEAN=1400, DEEP_OCEAN=500,
)
PLANT_RATE_MAX = int(B_PLANTS.max())

assert all(B_PLANTS[globals()[name]] > 0 for name in _PLANT_MIX), \
    'a biome has a plant mix but no rate, so nothing will ever grow'
assert all(B_PLANTS[b] == 0 or B_PLANT_N[b] > 0 for b in range(BIOME_COUNT)), \
    'a biome has a plant rate but no mix to draw from'

# Where a village may stand. Flat, walkable, above water.
B_VILLAGE = _biome_table(
    0, PLAINS=1, MEADOW=1, SAVANNA=1, TAIGA=1, SNOWY_PLAINS=1, DESERT=1,
    SNOWY_TAIGA=1, FOREST=1,
)

# Village materials per biome family: plains, desert, savanna, snowy.
# Indexed by the style B_VILLAGE_STYLE picks, not by biome, so a plains and a
# forest village are built the same way — as they are in the reference, which
# has exactly these four house sets plus taiga.
V_WALL = np.array([OAK_PLANKS, SANDSTONE, ACACIA_PLANKS, SPRUCE_PLANKS], dtype=np.int32)
V_POST = np.array([WOOD, CUT_SANDSTONE, ACACIA_LOG, SPRUCE_LOG], dtype=np.int32)
V_FLOOR = np.array([COBBLESTONE, SANDSTONE, COARSE_DIRT, COBBLESTONE], dtype=np.int32)
V_ROOF = np.array([BRICKS, SMOOTH_SANDSTONE, HAY_BALE, COBBLESTONE], dtype=np.int32)
V_PATH = np.array([GRAVEL, SANDSTONE, COARSE_DIRT, GRAVEL], dtype=np.int32)
# Trim: window frames, beams and upper floors. One block that is not the wall
# is what stops a facade being a flat plane of planks.
V_ACCENT = np.array([STRIPPED_OAK_LOG, CUT_SANDSTONE, ACACIA_LOG,
                     STRIPPED_SPRUCE_LOG], dtype=np.int32)
V_BED = np.array([RED_WOOL, WHITE_WOOL, ORANGE_WOOL, LIGHT_BLUE_WOOL], dtype=np.int32)
# Real crops now that there is a shape that can draw one — a field of four
# parallel planes rather than the full cubes (hay, melon, pumpkin) that stood in
# for them while every block had to be a cube.
V_CROP = np.array([WHEAT, BEETROOTS, CARROTS, POTATOES], dtype=np.int32)
B_VILLAGE_STYLE = _biome_table(
    0, DESERT=1, BADLANDS=1, SAVANNA=2, TAIGA=3, SNOWY_TAIGA=3, SNOWY_PLAINS=3,
)

# The kinds of building a plot can hold. The reference has thirty-odd templates
# per style; these are the shapes they reduce to once you drop the trades, which
# need block entities we do not have.
(VB_HOUSE, VB_BIG_HOUSE, VB_FARM, VB_PEN, VB_TOWER, VB_LIBRARY) = range(6)

# Two alternate surfaces per biome, picked by a patch noise rather than by the
# biome. This is the reference's disk_gravel / disk_sand / disk_clay / coarse
# dirt scattering (`configured_feature/disk_*.json`, `forest_rock`), and it is
# most of why its ground does not read as flat paint: a plain is not one colour,
# it is grass with bald patches. Two thresholds on one noise field give two patch
# types for one sample, and the patch is *only* the top block — a disk is a
# surface feature, and going deeper would cost a second column pass for
# something you cannot see.
B_PATCH_A = _biome_table(
    COARSE_DIRT,
    TAIGA=GRASS, SNOWY_TAIGA=SNOW, SNOWY_PLAINS=SNOW, DARK_FOREST=PODZOL,
    JUNGLE=PODZOL, SWAMP=MUD, DESERT=SANDSTONE, BADLANDS=ORANGE_TERRACOTTA,
    MEADOW=COARSE_DIRT, WINDSWEPT_HILLS=STONE, GROVE=SNOW, SNOWY_SLOPES=STONE,
    JAGGED_PEAKS=STONE, STONY_PEAKS=GRAVEL, STONY_SHORE=GRAVEL, BEACH=GRAVEL,
    SNOWY_BEACH=PACKED_ICE, RIVER=CLAY,
)
B_PATCH_B = _biome_table(
    GRAVEL,
    FOREST=PODZOL, BIRCH_FOREST=PODZOL, DARK_FOREST=COARSE_DIRT,
    TAIGA=COARSE_DIRT, SNOWY_TAIGA=PODZOL, SNOWY_PLAINS=PACKED_ICE,
    JUNGLE=COARSE_DIRT, SWAMP=CLAY, DESERT=SMOOTH_SANDSTONE,
    BADLANDS=WHITE_TERRACOTTA, MEADOW=STONE, GROVE=PODZOL,
    SNOWY_SLOPES=PACKED_ICE, JAGGED_PEAKS=PACKED_ICE, STONY_PEAKS=CALCITE,
    STONY_SHORE=MOSSY_COBBLESTONE, BEACH=CLAY, RIVER=GRAVEL,
)

# Badlands banding. The real game drives this from a noise (clay_bands_offset);
# a repeating table shifted by a per-column hash reads the same from outside and
# costs nothing.
BADLANDS_BANDS = np.array([
    TERRACOTTA, ORANGE_TERRACOTTA, TERRACOTTA, YELLOW_TERRACOTTA,
    BROWN_TERRACOTTA, TERRACOTTA, ORANGE_TERRACOTTA, RED_TERRACOTTA,
    TERRACOTTA, WHITE_TERRACOTTA, ORANGE_TERRACOTTA, TERRACOTTA,
    BROWN_TERRACOTTA, TERRACOTTA, ORANGE_TERRACOTTA, TERRACOTTA,
], dtype=np.int32)

# ---------------------------------------------------------------------------
# Ores. Attempts per chunk, vertical band, blob radius. Ranges are scaled to
# our compressed world: diamond hugs the bedrock, coal reaches the surface.
# ---------------------------------------------------------------------------
ORE_ID = np.array([COAL_ORE, IRON_ORE, COPPER_ORE, GOLD_ORE, REDSTONE_ORE,
                   LAPIS_ORE, DIAMOND_ORE, EMERALD_ORE], dtype=np.int32)
ORE_DEEP_ID = np.array([DEEP_COAL_ORE, DEEP_IRON_ORE, DEEP_COPPER_ORE,
                        DEEP_GOLD_ORE, DEEP_REDSTONE_ORE, DEEP_LAPIS_ORE,
                        DEEP_DIAMOND_ORE, DEEP_EMERALD_ORE], dtype=np.int32)
ORE_TRIES = np.array([9, 6, 5, 2, 3, 2, 2, 3], dtype=np.int32)
ORE_MIN_Y = np.array([8, 4, 6, 3, 3, 3, 3, 40], dtype=np.int32)
ORE_MAX_Y = np.array([60, 34, 40, 14, 10, 14, 8, 96], dtype=np.int32)
# The reference scatters granite, diorite, andesite and tuff through the stone
# in veins of 33 blocks (`ore_granite.json` and friends) — far bigger and far
# more numerous than any actual ore, because they are what stops a cave wall
# from being a single grey. Same blob machinery, no depth gate worth the name;
# only tuff is a deep rock.
ROCK_ID = np.array([GRANITE, DIORITE, ANDESITE, TUFF], dtype=np.int32)
ROCK_TRIES = np.array([4, 4, 4, 3], dtype=np.int32)
ROCK_MIN_Y = np.array([2, 2, 2, 1], dtype=np.int32)
ROCK_MAX_Y = np.array([70, 70, 70, 18], dtype=np.int32)
ROCK_RADIUS = np.array([2, 2, 2, 2], dtype=np.int32)
# A blob is the ball of radius r, so 0 is a single block and 1 is seven. Sized
# against how much stone a column actually has here: our bedrock-to-surface span
# is a quarter of the real game's, so its vein counts would bury the world in
# coal.
ORE_RADIUS = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int32)
# Emerald only shows up in the mountains, so it is gated on the biome rather
# than on depth like the rest.
ORE_MOUNTAIN_ONLY = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int32)

# ---------------------------------------------------------------------------
# Villages
# ---------------------------------------------------------------------------
VILLAGE_SPACING = 16     # chunks between region origins
VILLAGE_JITTER = 11      # how far into its region a village may wander
VILLAGE_CELL = 15        # spacing of the street grid; plots sit in its squares
VILLAGE_RADIUS = 44      # blocks; the streets reach this far
VILLAGE_PROBE = 18       # how far out the site is checked for being level
# Only the plaza has to be level — buildings stand at their own plot's height
# and the streets follow the ground — so this is much looser than it was, which
# is also what stopped villages being confined to the flattest ground there is.
VILLAGE_FLAT = 9


@njit(nogil=True, fastmath=True, cache=True)
def _spline(xs, ys, t):
    """Piecewise-linear lookup. The real game's splines carry derivatives too;
    at one sample per 4 blocks the extra smoothness is below a block."""
    n = xs.shape[0]
    if t <= xs[0]:
        return ys[0]
    if t >= xs[n - 1]:
        return ys[n - 1]
    for i in range(n - 1):
        if t < xs[i + 1]:
            f = (t - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + (ys[i + 1] - ys[i]) * f
    return ys[n - 1]


@njit(nogil=True, fastmath=True, cache=True)
def _octave2(x, z, freq, octaves, persistence):
    total = 0.0
    amp = 1.0
    norm = 0.0
    f = freq
    for _ in range(octaves):
        total += fast_noise2(x * f, z * f) * amp
        norm += amp
        amp *= persistence
        f *= 2.0
    return total / norm


@njit(nogil=True, fastmath=True, cache=True)
def _ridged(x, z, freq, octaves, persistence, gain):
    """Ridged multifractal, roughly [0, 1], sharp along its ridges.

    Returns it twice: the whole thing, and the first RIDGE_SMOOTH octaves on
    their own. The second is the massif without its own texture, and it is what
    the biome picker uses — see the note on `alt` in column_height.

    `1 - |noise|` peaks along the noise's zero set, which is a winding curve —
    a ridgeline rather than a blob. The *weight feedback* is what turns a stack
    of them into a mountain range instead of crumpled paper: each octave is
    multiplied by how high the octave below it came out, so fine detail only
    grows where there is already a crest, and dies away in the valleys. That is
    where branching spurs and foothills come from, and no amount of plain fBm
    produces them — fBm puts the same amount of detail everywhere, which is
    exactly what made this world read as noise with a height scale.
    """
    total = 0.0
    norm = 0.0
    coarse = 0.0
    coarse_norm = 0.0
    amp = 1.0
    weight = 1.0
    f = freq
    for i in range(octaves):
        n = 1.0 - abs(fast_noise2(x * f, z * f))
        n *= n * weight
        weight = n * gain
        if weight > 1.0:
            weight = 1.0
        total += n * amp
        norm += amp
        if i < RIDGE_SMOOTH:
            coarse += n * amp
            coarse_norm += amp
        amp *= persistence
        f *= 2.0
    return total / norm, coarse / coarse_norm


@njit(nogil=True, fastmath=True, cache=True)
def _amp_noise(x, z, freq, amps):
    """Octaves at the reference's own per-octave amplitudes.

    The real game does not use a geometric persistence: every climate noise
    ships an explicit amplitude list (`worldgen/noise/*.json`) and some entries
    are zero — continentalness triples the weight of its middle octaves and
    erosion switches its third off outright. That shape is a good part of why
    its coastlines read as coastlines, so it is copied rather than fitted.
    """
    total = 0.0
    norm = 0.0
    f = freq
    for i in range(amps.shape[0]):
        a = amps[i]
        if a != 0.0:
            total += fast_noise2(x * f, z * f) * a
            norm += a
        f *= 2.0
    return total / norm


@njit(nogil=True, fastmath=True, cache=True)
def _band(edges, value):
    """Which of len(edges)+1 bands *value* falls in."""
    i = 0
    while i < edges.shape[0] and value >= edges[i]:
        i += 1
    return i


@njit(nogil=True, fastmath=True, cache=True)
def climate_at(wx, wz):
    """The five fields, at one world column.

    fast_noise2 has one fixed permutation table, so the fields are decorrelated
    by sampling far apart rather than by reseeding. The offsets are deliberately
    not multiples of the table period.
    """
    # All five fields are read through the same displaced coordinate, exactly as
    # the reference's `shifted_noise` reads them all through the same shift_x /
    # shift_z. Sharing it is the point: a coastline, the erosion band behind it
    # and the biome on top all bend together, so the warp reads as the shape of
    # the place rather than as five fields disagreeing about where they are.
    warp_x = float(wx) + _amp_noise(float(wx) + 61700.5, float(wz) - 44300.5,
                                    1.0 / WARP_WL, WARP_AMP) * WARP_DIST
    warp_z = float(wz) + _amp_noise(float(wx) - 52100.5, float(wz) + 71900.5,
                                    1.0 / WARP_WL, WARP_AMP) * WARP_DIST
    x = warp_x
    z = warp_z

    cont = _amp_noise(x + 1000.5, z - 3000.5, 1.0 / CONT_WL, CONT_AMP) * CONT_GAIN
    ero = _amp_noise(x - 7300.5, z + 5100.5, 1.0 / ERO_WL, ERO_AMP) * CLIMATE_GAIN
    weird = _amp_noise(x + 12100.5, z + 9300.5, 1.0 / WEIRD_WL, WEIRD_AMP) * CLIMATE_GAIN
    temp = _amp_noise(x - 21700.5, z - 15300.5, 1.0 / TEMP_WL, TEMP_AMP) * CLIMATE_GAIN
    humid = _amp_noise(x + 31300.5, z - 25700.5, 1.0 / HUMID_WL, HUMID_AMP) * CLIMATE_GAIN

    if cont < -1.0:
        cont = -1.0
    elif cont > 1.0:
        cont = 1.0
    if ero < -1.0:
        ero = -1.0
    elif ero > 1.0:
        ero = 1.0
    if weird < -1.0:
        weird = -1.0
    elif weird > 1.0:
        weird = 1.0
    return cont, ero, weird, temp, humid


@njit(nogil=True, fastmath=True, cache=True)
def column_height(cont, ero, weird, wx, wz):
    """Surface height, river strength and biome altitude for one column.

    Returns floats — the caller rounds. River strength is 0 away from a river
    and 1 in the middle of one, and is what the biome picker uses to tell a
    river apart from the land it cuts through.

    The third value is the same terrain **without its own fine texture**, and it
    exists because a biome must not be chosen from a height that has any. The
    ridge field's last two octaves move the surface about ±5 blocks over 50, so
    a threshold anywhere near the ground gets crossed and re-crossed: a snow
    line drawn on the real height came out as snow and grass alternating every
    twenty blocks, and the same speckle produced groves, meadows and windswept
    hills a few blocks wide all over the map. Measured, the altitude rules cut
    the typical biome from 93 blocks across to 64.

    The reference has no such problem because it never asks: its biome `depth`
    is a spline of continentalness, erosion and folded ridges and nothing else
    (`density_function/overworld/offset.json`), and the 3D noise that gives the
    terrain its surface is applied afterwards. This is that idea with the two
    coarse ridge octaves left in, so the line still follows the massif it is
    drawn on rather than a spline that has never heard of it.
    """
    # The real game's fold: valleys where weirdness is near zero, ridges at
    # |w| = 2/3. One noise gives both, which is why they interlock. Here it sets
    # where a range runs at the scale of a whole massif, and — because the
    # rivers are also the w = 0 crossing — it puts them in the valley floors.
    pv = 1.0 - abs(3.0 * abs(weird) - 2.0)

    shelf = _spline(CONT_X, CONT_W, cont)
    base = _spline(CONT_X, CONT_Y, cont) + _spline(ERO_X, ERO_LIFT, ero) * shelf
    rug = _spline(ERO_X, ERO_Y, ero) * shelf
    ridge, ridge_coarse = _ridged(float(wx) + 41300.5, float(wz) - 33700.5,
                                  1.0 / RIDGE_WL, 4, 0.5, RIDGE_GAIN)
    rough = _octave2(float(wx) - 8800.5, float(wz) + 6100.5, 1.0 / ROUGH_WL, 2, 0.3)

    # One ruggedness number scales the whole relief, so erosion genuinely
    # decides the *character* of a region rather than just its height: at rug 3
    # this is a plain with a few metres of roll, at rug 40 the same expression is
    # an alpine ridge system.
    relief = pv * PV_WEIGHT + (ridge - RIDGE_MID) * RIDGE_WEIGHT
    smooth = pv * PV_WEIGHT + (ridge_coarse - RIDGE_MID) * RIDGE_WEIGHT
    # Valleys are cut shallower than peaks are raised. Symmetric relief digs the
    # low side down to bedrock and the result reads as a canyon maze; real ranges
    # rise out of a plain that stays roughly at the base height.
    if relief < 0.0:
        relief *= VALLEY_BIAS
    if smooth < 0.0:
        smooth *= VALLEY_BIAS

    # Roughness is only there to break contour lines, and contours only form
    # where the ground is nearly level — so it fades out as the relief grows.
    # At full strength on a mountainside it fights the slope instead and scales
    # the peak like a pine cone.
    height = base + rug * relief + rough * ROUGH_AMP * 14.0 / (10.0 + rug)
    alt = base + rug * smooth

    # A valley first, then the channel inside it. The valley is four times the
    # width of the water and eased in with a smoothstep, so its banks meet the
    # surrounding land flat; cutting the channel's width straight down to the
    # bed — which is what this did — leaves a slot with vertical walls.
    aw = abs(weird)
    river = 0.0
    if aw < RIVER_VALLEY and height > RIVER_BED:
        fade = (RIVER_MAX_H - height) / RIVER_FADE
        if fade > 1.0:
            fade = 1.0
        if fade > 0.0:
            v = 1.0 - aw / RIVER_VALLEY
            v = v * v * (3.0 - 2.0 * v)
            cut = v * fade
            height += (RIVER_BED - height) * cut
            alt += (RIVER_BED - alt) * cut       # the valley the biome sees too
            if aw < RIVER_WIDTH:
                river = (1.0 - aw / RIVER_WIDTH) * fade

    if height < MIN_TERRAIN:
        height = MIN_TERRAIN
    return height, river, alt


@njit(nogil=True, fastmath=True, cache=True)
def biome_at(temp, humid, cont, ero, weird, height, alt, river):
    """Which biome a column belongs to.

    `height` is the real surface and only decides whether the column is under
    water; `alt` is that surface without its own fine texture and is what every
    altitude rule below reads. See column_height for why they are two numbers.

    Order matters: water, then shore, then altitude, then the temperature x
    humidity grid. The real game does the same thing with a 6-dimensional
    nearest-neighbour search over ~200 entries; the early-outs here are the
    cases that search almost always lands on anyway.

    **Nothing is jittered here.** It used to be: a 430-block fractal and a
    per-block hash were added to temperature and humidity right at this point,
    to break up band edges that were otherwise smooth arcs. They broke them up
    far too well. A value jitter of a third of a band moves a column *bands* at
    a time, so a plains grew 20-block deserts inside it and every border became
    a 10-block stipple of two different top blocks — the ground reading as a
    quilt. The edges are frayed by the coordinate warp in `climate_at` now,
    which is where the reference does it and which cannot invent a biome that
    is not there, because it moves where the map is read rather than what it
    says.
    """
    pv = 1.0 - abs(3.0 * abs(weird) - 2.0)

    # **Sea and shore are decided by continentalness, not by height**, and that
    # is not a stylistic preference. A height window a few blocks tall, on
    # ground that has any roughness in it at all, is crossed and re-crossed
    # every few blocks: the beach used to come out as 13 700 separate runs
    # averaging 5 blocks — a speckled ribbon rather than a beach, and the same
    # for the deep-ocean line. Continentalness is the field the reference reads
    # for both (`beach` and `deep_ocean` are continentalness bands in its biome
    # parameter lists), it is lattice-smooth at a 2048-block wavelength, and the
    # band it draws is one continuous strip.
    if height < SEA_LEVEL:
        if river > 0.15 and height > SEA_LEVEL - 8:
            return RIVER
        if temp < TEMP_BANDS[0]:
            return FROZEN_OCEAN
        if cont < CONT_X[1]:
            return DEEP_OCEAN
        if temp > TEMP_BANDS[3]:
            return WARM_OCEAN
        return OCEAN

    if river > 0.15:
        return RIVER

    # The coast band, with a height guard that only bites where the coast rises
    # steeply out of the water — which is where a beach would be wrong anyway.
    if cont < CONT_X[3] and alt <= SEA_LEVEL + 8:
        if ero < -0.6:
            return STONY_SHORE
        if temp < TEMP_BANDS[0]:
            return SNOWY_BEACH
        if temp > TEMP_BANDS[3] and humid < HUMID_BANDS[1]:
            return DESERT
        return BEACH

    # The alpine cut-offs are measured against what the terrain actually builds,
    # not picked: these are about the top 1% and 5% of land columns by `alt`,
    # which is roughly the share of the real game's world that is peaks and
    # slopes. They have to be re-measured whenever the relief amplitude moves —
    # set against terrain that no longer reaches them, they match nothing and
    # the mountain biomes silently vanish, which is what test_worldgen's
    # "every biome appears" assert is there to catch.
    if alt > SEA_LEVEL + 41:
        if temp < TEMP_BANDS[1]:
            return JAGGED_PEAKS if pv > 0.35 else SNOWY_SLOPES
        if ero < -0.7 and pv > 0.25:
            return JAGGED_PEAKS
        return STONY_PEAKS

    if alt > SEA_LEVEL + 23 and temp < TEMP_BANDS[1]:
        return GROVE if humid > HUMID_BANDS[1] else SNOWY_SLOPES

    if ero < -0.62 and pv > 0.1:
        return WINDSWEPT_HILLS

    # Swamps are low and flat, and the erosion band is what carries "flat" —
    # a height window on its own speckled here too (median run: 2 blocks).
    if alt <= SEA_LEVEL + 10 and ero > 0.1 and temp > TEMP_BANDS[1] \
            and humid > HUMID_BANDS[3]:
        return SWAMP

    biome = LAND_GRID[_band(TEMP_BANDS, temp), _band(HUMID_BANDS, humid)]

    if biome == DESERT and weird > 0.35:
        return BADLANDS
    if biome == PLAINS and temp < TEMP_BANDS[2] and alt > SEA_LEVEL + 20 \
            and ero < -0.2:
        return MEADOW
    return biome


@njit(nogil=True, fastmath=True, cache=True)
def climate_interp(world_x, world_z):
    """Climate at one column, read off the same lattice the chunks use.

    Sampling the noise directly instead would give a *different* world: the
    splines turn a small climate difference into several blocks of height, so a
    spawn point picked from the raw fields lands inside a hill. Four samples
    where the kernel amortises one — this is the road not taken in the chunk
    loop, not a second implementation of it.
    """
    x0 = int(math.floor((world_x - GRID_ANCHOR) / GRID_STEP)) * GRID_STEP + GRID_ANCHOR
    z0 = int(math.floor((world_z - GRID_ANCHOR) / GRID_STEP)) * GRID_STEP + GRID_ANCHOR
    fx = (world_x - x0) / GRID_STEP
    fz = (world_z - z0) / GRID_STEP

    out = np.empty(5, dtype=np.float64)
    c00 = climate_at(x0, z0)
    c10 = climate_at(x0 + GRID_STEP, z0)
    c01 = climate_at(x0, z0 + GRID_STEP)
    c11 = climate_at(x0 + GRID_STEP, z0 + GRID_STEP)
    for k in range(5):
        a = c00[k] + (c01[k] - c00[k]) * fz
        b = c10[k] + (c11[k] - c10[k]) * fz
        out[k] = a + (b - a) * fx
    return out


def surface_height(world_x, world_z):
    """Ground level at one world column, as the chunk generator would build it."""
    return terrain_top(world_x, world_z)


def column_biome(world_x, world_z):
    """Biome id at one world column, for tests and telemetry."""
    c = climate_interp(world_x, world_z)
    height, river, alt = column_height(c[0], c[1], c[2], world_x, world_z)
    return biome_at(c[3], c[4], c[0], c[1], c[2], height, alt, river)


def find_spawn(world_x=8, world_z=8, reach=3000):
    """Eye position for a player starting near (world_x, world_z).

    Rings outward until a column stands clear of the water. Two fifths of this
    world is sea, so the fixed start the game used to have now lands on a sea
    bed about that often.
    """
    step = 24
    radius = 0
    while radius <= reach:
        points = max(1, int(radius / step) * 6)
        for i in range(points):
            angle = 2.0 * math.pi * i / points
            x = int(world_x + radius * math.cos(angle))
            z = int(world_z + radius * math.sin(angle))
            height = surface_height(x, z)
            if height > SEA_LEVEL + 1 and column_biome(x, z) != RIVER:
                return x + 0.5, height + 3.0, z + 0.5
        radius += step
    return world_x, SEA_LEVEL + 4.0, world_z


# ---------------------------------------------------------------------------
# Chunk-local writing helpers
# ---------------------------------------------------------------------------
@njit(nogil=True, fastmath=True, cache=True)
def _put(blocks, lx, ly, lz, block):
    """Write a block if it lands inside this chunk. Features straddle chunk
    seams by construction — both sides run the same code and each keeps its
    half — so the bounds check is the whole cross-chunk mechanism."""
    if 0 <= lx < CHUNK_SIZE and 0 <= lz < CHUNK_SIZE and 0 <= ly < CHUNK_HEIGHT:
        blocks[lx, ly, lz] = block


@njit(nogil=True, fastmath=True, cache=True)
def _get(blocks, lx, ly, lz):
    if 0 <= lx < CHUNK_SIZE and 0 <= lz < CHUNK_SIZE and 0 <= ly < CHUNK_HEIGHT:
        return blocks[lx, ly, lz]
    return AIR


@njit(nogil=True, fastmath=True, cache=True)
def _bilerp(grid, k, p, q):
    i = p // GRID_STEP
    j = q // GRID_STEP
    fx = (p - i * GRID_STEP) / GRID_STEP
    fz = (q - j * GRID_STEP) / GRID_STEP
    a = grid[k, i, j] + (grid[k, i, j + 1] - grid[k, i, j]) * fz
    b = grid[k, i + 1, j] + (grid[k, i + 1, j + 1] - grid[k, i + 1, j]) * fz
    return a + (b - a) * fx


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
@njit(nogil=True, fastmath=True, cache=True)
def _leaves(blocks, cx, y, cz, r, leaf, wx, wz, corner, round_=0):
    """One row of a canopy — the reference's `FoliagePlacer.placeLeavesRow`.

    A row is the **(2r+1) square**, not a disc. `placeLeavesRow` walks the whole
    square and the only thing that takes cells out of it is
    `shouldSkipLocation`, which for every placer used here reduces to "drop the
    four diagonal corners". `corner` is the chance each one goes: `1.0` is
    `SpruceFoliagePlacer`'s unconditional cut, `0.5` is `BlobFoliagePlacer`'s
    coin. Leaves only go where there is air, so a row never eats a trunk or the
    branch it hangs off.

    **The coin is where a forest's variety comes from.** It is drawn per corner,
    per row, so two oaks of the same height differ in up to 2^12 ways instead of
    being the same stamp — which is exactly what a wood of identical trees was.
    It is drawn from the *tree's* world position for the same reason every other
    number here is: both chunks either side of a seam have to derive the same
    tree, so a hash of the leaf cell's own absolute position would be fine but a
    counter or a chunk coordinate would not.

    This used to be a disc with a `fray` that thinned the whole rim. At r<=2 the
    disc and the square agree exactly (5 and 21 cells, which is the reference's
    own count), so the shape was only wrong from r=3 up — but the fray was wrong
    everywhere, because a rim eaten at a uniform rate reads as damage, where
    four missing corners read as a crown.

    `round_` cuts the square back to a disc and moves the coin to the whole rim,
    which is the one placer that does that: `CherryFoliagePlacer` has a
    `cornerHoleChance` and a circular bottom layer, and being able to see sky
    through it is the entire character of the tree.
    """
    if r < 0:
        return
    limit = r * r + r // 2
    for dx in range(-r, r + 1):
        for dz in range(-r, r + 1):
            if round_ != 0:
                d2 = dx * dx + dz * dz
                if d2 > limit:
                    continue
                if (d2 > limit - r and corner > 0.0
                        and fast_rand(wx + dx * 3.0, y * 7.0 + 3.0,
                                      wz + dz * 3.0) < corner):
                    continue
            elif r > 0 and (dx == r or dx == -r) and (dz == r or dz == -r):
                if corner >= 1.0:
                    continue
                if fast_rand(wx + dx * 3.0, y * 7.0 + 3.0,
                             wz + dz * 3.0) < corner:
                    continue
            if _get(blocks, cx + dx, y, cz + dz) == AIR:
                _put(blocks, cx + dx, y, cz + dz, leaf)


@njit(nogil=True, fastmath=True, cache=True)
def _trunk(blocks, lx, lz, y0, y1, log):
    for y in range(y0, y1 + 1):
        _put(blocks, lx, y, lz, log)


@njit(nogil=True, fastmath=True, cache=True)
def _blob(blocks, cx, attach_y, cz, r, leaf, wx, wz):
    """`blob_foliage_placer` — the oak / birch / jungle crown, and the ball on
    the end of a fancy oak's branch.

    Four rows down from the attachment at r-1, r-1, r, r, which is what its
    `max(radius - 1 - i/2, 0)` works out to at height 3. The attachment sits one
    block above the last log, which is the wiki's "canopies grow 1 block higher
    than the highest log block". The top row always loses its corners
    (`shouldSkipLocation`'s `localY == 0` term); the rest coin for them.
    """
    if r < 0:
        return
    _leaves(blocks, cx, attach_y, cz, r - 1, leaf, wx, wz, 1.0)
    _leaves(blocks, cx, attach_y - 1, cz, r - 1, leaf, wx, wz, 0.5)
    _leaves(blocks, cx, attach_y - 2, cz, r, leaf, wx, wz, 0.5)
    _leaves(blocks, cx, attach_y - 3, cz, r, leaf, wx, wz, 0.5)


@njit(nogil=True, fastmath=True, cache=True)
def _tip_blob(blocks, lx, lz, bx, by, bz, r, leaf, wx, wz):
    """The ball on the end of a branch, shrunk to whatever is left of PAD.

    Four shapes here reach sideways before they put leaves down — fancy oak,
    mega jungle, cherry, acacia — and a canopy that crosses PAD is one the
    neighbouring chunk never draws its half of, i.e. a hole at the seam. Rather
    than pick branch lengths that happen to fit, the ball is sized from the
    branch: reach two and it is a full blob, reach three and it is a small one.
    That is also why the branches may vary in length at all.
    """
    far = abs(bx - lx)
    if abs(bz - lz) > far:
        far = abs(bz - lz)
    if r > PAD - far:
        r = PAD - far
    _blob(blocks, bx, by, bz, r, leaf, wx, wz)


@njit(nogil=True, fastmath=True, cache=True)
def _acacia_plate(blocks, cx, cy, cz, r, leaf, wx, wz):
    """`acacia_foliage_placer`: a wide flat plate with a cross laid on top.

    Its `shouldSkipLocation` keeps, on the upper row, only the two axes and the
    3x3 core — so the plate's outline is a square and the layer above it is a
    plus. That silhouette, seen edge-on against a savanna sky, is the whole
    tree; two small round blobs with a gap between them (which is what the
    forking arms used to grow) is a different plant entirely.
    """
    if r < 1:
        return
    _leaves(blocks, cx, cy - 1, cz, r, leaf, wx, wz, 1.0)
    rr = r - 1
    for dx in range(-rr, rr + 1):
        for dz in range(-rr, rr + 1):
            if (dx > 1 or dx < -1 or dz > 1 or dz < -1) and dx != 0 and dz != 0:
                continue
            if _get(blocks, cx + dx, cy, cz + dz) == AIR:
                _put(blocks, cx + dx, cy, cz + dz, leaf)


@njit(nogil=True, fastmath=True, cache=True)
def place_boulder(blocks, lx, ground_y, lz, block, wx, wz):
    """A half-buried lump of rock. Squashed, and with its rim frayed by the
    same hash the canopies use, so it is a rock and not a ball."""
    r = 1 + int(fast_rand(wx, 41.0, wz) * 2)
    for dy in range(-2, r + 1):
        rr = r if dy <= 0 else r - dy
        if rr < 0:
            continue
        for dx in range(-rr, rr + 1):
            for dz in range(-rr, rr + 1):
                d2 = dx * dx + dz * dz
                if d2 > rr * rr + rr:
                    continue
                if d2 >= rr * rr and fast_rand(wx + dx, dy * 5.0, wz + dz) < 0.4:
                    continue
                _put(blocks, lx + dx, ground_y + dy, lz + dz, block)


@njit(nogil=True, fastmath=True, cache=True)
def place_tree(blocks, lx, base_y, lz, shape, log, leaf, wx, wz):
    """One tree, rooted at (lx, base_y, lz) in chunk-local space.

    lx/lz may sit outside the chunk: a tree rooted in the next chunk over still
    drops the half of its canopy that reaches across, and the neighbour writes
    the other half from the same seed. That symmetry is why the canopy is not
    clipped at the seam — and it is also why **every number here is drawn from
    `fast_rand` on the tree's world position and nothing else**. A draw that
    depended on the chunk, or on a counter, would give the two halves different
    trees.

    Nothing may reach further than PAD sideways. The crowns that sit over the
    trunk are inside it by radius; the four shapes that branch first and grow
    leaves afterwards go through `_tip_blob`, which sizes the ball from what the
    branch has already spent.
    """
    if shape == TS_NONE:
        return
    if base_y + 30 >= CHUNK_HEIGHT:
        return

    if shape == TS_FALLEN:
        # A dead trunk lying on the ground. The reference scatters these at
        # about 1% through every wooded biome and they do a lot of work: the eye
        # reads a forest floor with something on it as a forest rather than as
        # pillars on a lawn.
        length = 3 + int(fast_rand(wx, 61.0, wz) * 3)
        axis = int(fast_rand(wx, 67.0, wz) * 4)
        step_x = (1, 0, -1, 0)[axis]
        step_z = (0, 1, 0, -1)[axis]
        for i in range(length):
            x = lx + step_x * i
            z = lz + step_z * i
            _put(blocks, x, base_y, z, log)
            if fast_rand(wx + i, 71.0, wz) < 0.35:
                _put(blocks, x, base_y + 1, z, leaf)
        return

    if shape == TS_BUSH:
        # jungle_bush: one or two logs under a squat two-row blob. It is the
        # smallest thing the reference calls a tree and it is 45% of a jungle,
        # because a canopy all at one height needs something underneath it.
        height = 1 + int(fast_rand(wx, 73.0, wz) * 2)
        _trunk(blocks, lx, lz, base_y, base_y + height - 1, log)
        _leaves(blocks, lx, base_y + height, lz, 1, leaf, wx, wz, 1.0)
        _leaves(blocks, lx, base_y + height - 1, lz, 2, leaf, wx, wz, 0.5)
        _put(blocks, lx, base_y - 1, lz, DIRT)
        return

    if shape == TS_MUSHROOM or shape == TS_FLAT_MUSHROOM:
        # huge_red_mushroom is a dome on a short stem; huge_brown_mushroom is a
        # flat plate on a shorter one and half again as wide. They were the same
        # shape in two colours, which wasted the only two plants in the dark
        # forest that are not trees. We have no stem block, so the caller passes
        # bone — the closest texture in the table, and at any distance it reads
        # exactly right.
        if shape == TS_MUSHROOM:
            stem = 4 + int(fast_rand(wx, 77.0, wz) * 4)
            top = base_y + stem - 1
            _trunk(blocks, lx, lz, base_y, top, log)
            _leaves(blocks, lx, top, lz, 2, leaf, wx, wz, 0.0, 1)
            _leaves(blocks, lx, top + 1, lz, 2, leaf, wx, wz, 0.0, 1)
            _leaves(blocks, lx, top + 2, lz, 1, leaf, wx, wz, 1.0)
        else:
            stem = 3 + int(fast_rand(wx, 77.0, wz) * 3)
            top = base_y + stem - 1
            _trunk(blocks, lx, lz, base_y, top, log)
            _leaves(blocks, lx, top + 1, lz, 3, leaf, wx, wz, 0.0, 1)
        _put(blocks, lx, base_y - 1, lz, DIRT)
        return

    # --- everything below has a trunk and a crown ---------------------------
    # Logs in the trunk, from the reference's own trunk placers
    # (`configured_feature/{oak,birch,spruce,pine,...}.json`). Each is
    # `base + rand(a) [+ rand(b)]`, and where there are two draws they are two
    # draws here as well: summing two uniforms is a triangular distribution, so
    # a spruce is usually mid-height and only rarely at either end, which is not
    # what one wider uniform gives.
    if shape == TS_OAK:
        trunk = 4 + int(fast_rand(wx, 91.0, wz) * 3)           # 4 + rand(2)
    elif shape == TS_SWAMP_OAK:
        trunk = 5 + int(fast_rand(wx, 91.0, wz) * 4)           # 5 + rand(3)
    elif shape == TS_BIRCH:
        trunk = 5 + int(fast_rand(wx, 91.0, wz) * 3)           # 5 + rand(2)
    elif shape == TS_SUPER_BIRCH:
        trunk = (5 + int(fast_rand(wx, 91.0, wz) * 3)
                 + int(fast_rand(wx, 89.0, wz) * 7))           # 5 + rand(2)+rand(6)
    elif shape == TS_SPRUCE:
        trunk = (5 + int(fast_rand(wx, 91.0, wz) * 3)
                 + int(fast_rand(wx, 89.0, wz) * 2))           # 5 + rand(2)+rand(1)
    elif shape == TS_PINE:
        trunk = 6 + int(fast_rand(wx, 91.0, wz) * 5)           # 6 + rand(4)
    elif shape == TS_JUNGLE:
        trunk = 4 + int(fast_rand(wx, 91.0, wz) * 9)           # 4 + rand(8)
    elif shape == TS_ACACIA:
        trunk = (5 + int(fast_rand(wx, 91.0, wz) * 3)
                 + int(fast_rand(wx, 89.0, wz) * 3))           # 5 + rand(2)+rand(2)
    elif shape == TS_DARK_OAK:
        trunk = 6 + int(fast_rand(wx, 91.0, wz) * 3)           # "typically 6-8"
    elif shape == TS_FANCY:
        trunk = 6 + int(fast_rand(wx, 91.0, wz) * 7)           # 3 + rand(11), floored
    elif shape == TS_MEGA_SPRUCE or shape == TS_MEGA_PINE:
        trunk = 13 + int(fast_rand(wx, 91.0, wz) * 7)          # 13 + rand(2)+rand(14)
    elif shape == TS_MEGA_JUNGLE:
        trunk = 10 + int(fast_rand(wx, 91.0, wz) * 8)          # 10 + rand(2)+rand(19)
    else:                                    # TS_CHERRY
        trunk = 7 + int(fast_rand(wx, 91.0, wz) * 3)           # 7 + rand(1)

    # `trunk` is the log count, so the last log is one below base + trunk and
    # the crown attaches one above it — "canopies grow 1 block higher than the
    # highest log block".
    top = base_y + trunk - 1
    if top + 6 >= CHUNK_HEIGHT:
        return

    if (shape == TS_OAK or shape == TS_BIRCH or shape == TS_JUNGLE
            or shape == TS_SUPER_BIRCH or shape == TS_SWAMP_OAK):
        # blob_foliage_placer at radius 2 — or 3 for the swamp oak, which is the
        # whole difference between the wiki's 56-leaf oak and its 137-leaf one.
        _trunk(blocks, lx, lz, base_y, top, log)
        _blob(blocks, lx, top + 1, lz, 3 if shape == TS_SWAMP_OAK else 2,
              leaf, wx, wz)

    elif shape == TS_SPRUCE:
        # spruce_foliage_placer, written out. The radius does not taper
        # linearly: it climbs to a running cap, drops back, and the cap then
        # rises by one — a sawtooth, which is where a spruce's notched skirt
        # comes from. Three of its four numbers are drawn per tree, so no two
        # spruces in a stand have their notches at the same heights.
        #
        # What this replaced tapered linearly and then subtracted one on
        # alternate rows, which could and did land on r=0 *in the middle of the
        # crown* — and r=0 over a trunk places nothing at all, so a third of the
        # taiga had a bare ring sawn through it. The sawtooth cannot: it only
        # reaches 0 on its first reset, which is above the last log.
        _trunk(blocks, lx, lz, base_y, top, log)
        maxr = 2 + int(fast_rand(wx, 93.0, wz) * 2)            # UniformInt(2,3)
        off = int(fast_rand(wx, 95.0, wz) * 3)                 # UniformInt(0,2)
        fh = trunk - 1 - int(fast_rand(wx, 97.0, wz) * 2)      # height - trunkHeight
        if fh < 4:
            fh = 4
        r = int(fast_rand(wx, 99.0, wz) * 2)                   # nextInt(2)
        j = 1
        k = 0
        attach = top + 1
        for i in range(off, -fh - 1, -1):
            if attach + i > base_y:
                _leaves(blocks, lx, attach + i, lz, r, leaf, wx, wz, 1.0)
            if r >= j:
                r = k
                k = 1
                j = min(j + 1, maxr)
            else:
                r += 1

    elif shape == TS_PINE:
        # pine_foliage_placer: radius 1, three or four rows — a tuft on a bare
        # stem and deliberately nothing like the spruce beside it. The reference
        # makes them a 2:1 mix in taiga and the contrast is the point; giving
        # the pine a spruce's crown made that mix two of the same tree. Ten
        # leaves, which is the wiki's count for the matchstick spruce.
        _trunk(blocks, lx, lz, base_y, top, log)
        fh = 3 + int(fast_rand(wx, 93.0, wz) * 2)              # UniformInt(3,4)
        attach = top + 1
        r = 0
        for i in range(1, -fh, -1):
            _leaves(blocks, lx, attach + i, lz, r, leaf, wx, wz, 1.0)
            if r >= 1 and i == 2 - fh:
                r -= 1
            elif r < 1:
                r += 1

    elif shape == TS_ACACIA:
        # forking_trunk_placer: the stem runs up, then *leans* one block per
        # level in a cardinal direction, and a second fork peels off lower down
        # in a different one. The lean was eight-way here, and a trunk that
        # leans diagonally reads as a tree that has been pushed over.
        bend = trunk - 1 - int(fast_rand(wx, 97.0, wz) * 3)
        if bend < 2:
            bend = 2
        # The reference leans up to three blocks and then hangs a radius-3 plate
        # off the end of that, which reaches six. We have four, so the lean is
        # mostly one — the plate is the recognisable half of the tree and a
        # two-block lean costs it a ring.
        lean = 2 if fast_rand(wx, 99.0, wz) < 0.34 else 1
        d = int(fast_rand(wx, 101.0, wz) * 4)
        ax, az = lx, lz
        for y in range(base_y, top + 1):
            step = y - base_y - bend
            if 0 <= step < lean:
                ax += (1, 0, -1, 0)[d]
                az += (0, 1, 0, -1)[d]
            _put(blocks, ax, y, az, log)
        _acacia_plate(blocks, ax, top + 2, az, min(3, PAD - lean), leaf, wx, wz)
        d2 = int(fast_rand(wx, 103.0, wz) * 4)
        if d2 != d:
            # The second fork starts below the first's bend and is shorter, so
            # the two plates sit at different heights — that stagger is most of
            # what tells one acacia from the next.
            run = 1 + int(fast_rand(wx, 107.0, wz) * 2)
            fy = base_y + bend - 1 - int(fast_rand(wx, 109.0, wz) * 2)
            if fy < base_y + 2:
                fy = base_y + 2
            fx, fz = lx, lz
            for i in range(run):
                fx += (1, 0, -1, 0)[d2]
                fz += (0, 1, 0, -1)[d2]
                fy += 1
                _put(blocks, fx, fy, fz, log)
            _acacia_plate(blocks, fx, fy + 2, fz, min(3, PAD - run), leaf, wx, wz)

    elif shape == TS_DARK_OAK:
        # giant_trunk_placer, 2x2, plus the stubby branches the wiki calls
        # "irregular logs, representing large branches ... nearly always
        # present". dark_oak_foliage_placer then lays rows of 2, 3, 2 about both
        # diagonal corners and sometimes a fourth at 1 — drawing the crown twice
        # covers the square without a radius that would overrun PAD, and radius
        # 3 about lx+1 reaches exactly it.
        for ox2 in range(2):
            for oz2 in range(2):
                _trunk(blocks, lx + ox2, lz + oz2, base_y, top, log)
                _put(blocks, lx + ox2, base_y - 1, lz + oz2, DIRT)
        for b in range(1 + int(fast_rand(wx, 97.0, wz) * 2)):
            d = int(fast_rand(wx, 101.0 + b * 9.0, wz) * 4)
            sx = (1, 0, -1, 0)[d]
            sz = (0, 1, 0, -1)[d]
            bx = lx + (1 if sx > 0 else 0) + sx
            bz = lz + (1 if sz > 0 else 0) + sz
            _put(blocks, bx, top, bz, log)
            _put(blocks, bx, top - 1, bz, log)
        for i in range(3):
            r = (2, 3, 2)[i]
            _leaves(blocks, lx, top - 1 + i, lz, r, leaf, wx, wz, 0.5)
            _leaves(blocks, lx + 1, top - 1 + i, lz + 1, r, leaf, wx, wz, 0.5)
        if fast_rand(wx, 99.0, wz) < 0.5:
            _leaves(blocks, lx, top + 2, lz, 1, leaf, wx, wz, 1.0)
            _leaves(blocks, lx + 1, top + 2, lz + 1, 1, leaf, wx, wz, 1.0)
        return

    elif shape == TS_MEGA_SPRUCE or shape == TS_MEGA_PINE:
        # mega_pine_foliage_placer: the radius grows straight down the crown,
        # `radius + floor(k / crown * 3.5)`, so it is a cone and not a stack of
        # rings. The two configured features differ in one number and in nothing
        # else — the mega spruce's crown is 13-17 rows and skirts nearly to the
        # ground, the mega pine's is 3-7 and leaves a bare column under it.
        # 3.5 rounds up to a radius of 4; ours stops at 3, because 3 about the
        # far corner of a 2x2 trunk is already PAD.
        for ox2 in range(2):
            for oz2 in range(2):
                _trunk(blocks, lx + ox2, lz + oz2, base_y, top, log)
                _put(blocks, lx + ox2, base_y - 1, lz + oz2, DIRT)
        if shape == TS_MEGA_SPRUCE:
            crown = 13 + int(fast_rand(wx, 93.0, wz) * 5)      # UniformInt(13,17)
        else:
            crown = 3 + int(fast_rand(wx, 93.0, wz) * 5)       # UniformInt(3,7)
        if crown > trunk - 2:
            crown = trunk - 2
        attach = top + 1
        for k in range(crown, -1, -1):
            r = 1 + (k * 7) // (2 * crown)
            if r > 3:
                r = 3
            _leaves(blocks, lx, attach - k, lz, r, leaf, wx, wz, 1.0)
            _leaves(blocks, lx + 1, attach - k, lz + 1, r, leaf, wx, wz, 1.0)
        _leaves(blocks, lx, attach + 1, lz, 0, leaf, wx, wz, 1.0)
        _leaves(blocks, lx + 1, attach + 1, lz + 1, 0, leaf, wx, wz, 1.0)
        # alter_ground: the reference lays a disc of podzol under every mega
        # tree, and it is what stops a giant standing on a lawn.
        for dx in range(-2, 4):
            for dz in range(-2, 4):
                if dx * dx + dz * dz <= 7 or (dx - 1) ** 2 + (dz - 1) ** 2 <= 7:
                    if _get(blocks, lx + dx, base_y - 1, lz + dz) != AIR:
                        _put(blocks, lx + dx, base_y - 1, lz + dz, PODZOL)
        return

    elif shape == TS_MEGA_JUNGLE:
        # mega_jungle_trunk_placer: a 2x2 stem with branches off its upper half,
        # each carrying its own blob, and a crown over the top. Without the
        # branches it is a pole with a hat — which, at 10 to 17 logs, is most of
        # a jungle's skyline standing bare.
        for ox2 in range(2):
            for oz2 in range(2):
                _trunk(blocks, lx + ox2, lz + oz2, base_y, top, log)
                _put(blocks, lx + ox2, base_y - 1, lz + oz2, DIRT)
        for b in range(2 + int(fast_rand(wx, 97.0, wz) * 3)):
            d = int(fast_rand(wx, 101.0 + b * 7.0, wz) * 4)
            sx = (1, 0, -1, 0)[d]
            sz = (0, 1, 0, -1)[d]
            run = 1 + int(fast_rand(wx, 113.0 + b * 3.0, wz) * 2)
            by = top - 2 - int(fast_rand(wx, 107.0 + b * 5.0, wz) * (trunk // 2))
            bx = lx + (1 if sx > 0 else 0)
            bz = lz + (1 if sz > 0 else 0)
            for i in range(run):
                bx += sx
                bz += sz
                _put(blocks, bx, by, bz, log)
            _tip_blob(blocks, lx, lz, bx, by + 2, bz, 2, leaf, wx + b * 13, wz)
        _blob(blocks, lx, top + 2, lz, 2, leaf, wx, wz)
        _blob(blocks, lx + 1, top + 2, lz + 1, 2, leaf, wx, wz + 7)
        return

    elif shape == TS_FANCY:
        # fancy_trunk_placer: a tall stem with branches angling out of its upper
        # half, each ending in its own blob. This is the big oak, and it is the
        # one shape whose outline was already different every time. What it
        # gains here is branches of two lengths rather than one, and the wiki's
        # "leaves grow 3 blocks higher than the highest log" — which it calls out
        # for the fancy oak alone.
        _trunk(blocks, lx, lz, base_y, top, log)
        branches = 2 + int(fast_rand(wx, 97.0, wz) * 3)
        for b in range(branches):
            d = int(fast_rand(wx, 103.0 + b * 11.0, wz) * 8)
            sx = (1, 1, 0, -1, -1, -1, 0, 1)[d]
            sz = (0, 1, 1, 1, 0, -1, -1, -1)[d]
            run = 1 + int(fast_rand(wx, 109.0 + b * 7.0, wz) * 2)
            by = base_y + trunk // 2 + (b * (trunk // 2)) // branches
            bx, bz = lx, lz
            for i in range(run):
                bx += sx
                bz += sz
                by += 1
                _put(blocks, bx, by, bz, log)
            _tip_blob(blocks, lx, lz, bx, by + 2, bz, 2, leaf, wx + b * 17, wz)
        _leaves(blocks, lx, top + 3, lz, 1, leaf, wx, wz, 1.0)
        _leaves(blocks, lx, top + 2, lz, 2, leaf, wx, wz, 0.5)
        _leaves(blocks, lx, top + 1, lz, 3, leaf, wx, wz, 0.15, 1)
        _leaves(blocks, lx, top, lz, 3, leaf, wx, wz, 0.15, 1)
        _leaves(blocks, lx, top - 1, lz, 2, leaf, wx, wz, 0.5)

    else:                                    # TS_CHERRY
        # cherry_trunk_placer: one to three branches that run out horizontally
        # from the upper trunk, each carrying its own crown — the wiki's
        # "horizontally facing branches", and the reason a cherry "may
        # occasionally have multiple canopies". cherry_foliage_placer then lays
        # 1, 2, 3, 3, 2 over the stem with a quarter of the rim out, which is
        # what makes it lacy rather than a pink brick.
        _trunk(blocks, lx, lz, base_y, top, log)
        for b in range(1 + int(fast_rand(wx, 97.0, wz) * 3)):
            d = int(fast_rand(wx, 103.0 + b * 13.0, wz) * 4)
            sx = (1, 0, -1, 0)[d]
            sz = (0, 1, 0, -1)[d]
            by = top - 1 - int(fast_rand(wx, 107.0 + b * 5.0, wz) * 3)
            _put(blocks, lx + sx, by, lz + sz, log)
            _put(blocks, lx + sx * 2, by, lz + sz * 2, log)
            _tip_blob(blocks, lx, lz, lx + sx * 2, by + 3, lz + sz * 2, 2,
                      leaf, wx + b * 19, wz)
        _leaves(blocks, lx, top + 3, lz, 1, leaf, wx, wz, 0.25)
        _leaves(blocks, lx, top + 2, lz, 2, leaf, wx, wz, 0.25)
        _leaves(blocks, lx, top + 1, lz, 3, leaf, wx, wz, 0.25, 1)
        _leaves(blocks, lx, top, lz, 3, leaf, wx, wz, 0.25, 1)
        _leaves(blocks, lx, top - 1, lz, 2, leaf, wx, wz, 0.25)

    _put(blocks, lx, base_y - 1, lz, DIRT)


@njit(nogil=True, fastmath=True, cache=True)
def place_ores(blocks, chunk_x, chunk_z, top_y, mountain):
    """Scatter ore blobs through this chunk's stone.

    Position and blob shape come from the position hash rather than from a
    noise field: an ore vein is a handful of blocks and a noise threshold that
    fine costs a sample per block for something a hash decides just as well.
    Only stone is replaced, so a blob that lands in a cave or in the open air
    simply does not appear.
    """
    # Stone variants first, so an ore vein can still be cut into one.
    for k in range(ROCK_ID.shape[0]):
        y_max = ROCK_MAX_Y[k]
        if y_max > top_y:
            y_max = top_y
        if y_max <= ROCK_MIN_Y[k]:
            continue
        for attempt in range(ROCK_TRIES[k]):
            seed = chunk_x * 4096.0 + attempt * 29.0 + k * 811.0 + 100000.0
            x = int(fast_rand(seed, 3.0, chunk_z) * CHUNK_SIZE)
            z = int(fast_rand(seed, 7.0, chunk_z) * CHUNK_SIZE)
            y = ROCK_MIN_Y[k] + int(fast_rand(seed, 11.0, chunk_z) * (y_max - ROCK_MIN_Y[k]))
            r = ROCK_RADIUS[k]
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if dx * dx + dy * dy + dz * dz > r * r + 1:
                            continue
                        here = _get(blocks, x + dx, y + dy, z + dz)
                        if here == STONE or here == DEEPSLATE:
                            _put(blocks, x + dx, y + dy, z + dz, ROCK_ID[k])

    for k in range(ORE_ID.shape[0]):
        if ORE_MOUNTAIN_ONLY[k] != 0 and not mountain:
            continue
        y_max = ORE_MAX_Y[k]
        if y_max > top_y:
            y_max = top_y
        if y_max <= ORE_MIN_Y[k]:
            continue

        for attempt in range(ORE_TRIES[k]):
            seed = chunk_x * 4096.0 + attempt * 17.0 + k * 613.0
            x = int(fast_rand(seed, 3.0, chunk_z) * CHUNK_SIZE)
            z = int(fast_rand(seed, 7.0, chunk_z) * CHUNK_SIZE)
            y = ORE_MIN_Y[k] + int(fast_rand(seed, 11.0, chunk_z) * (y_max - ORE_MIN_Y[k]))
            r = ORE_RADIUS[k]

            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if dx * dx + dy * dy + dz * dz > r * r + 1:
                            continue
                        here = _get(blocks, x + dx, y + dy, z + dz)
                        # The reference's `ore_replaceables` tag covers the stone
                        # variants too, so a vein is not blocked by the granite
                        # blob that landed there first.
                        if (here == STONE or here == DEEPSLATE or here == GRANITE
                                or here == DIORITE or here == ANDESITE or here == TUFF):
                            # A vein that straddles the deepslate line wears both
                            # skins, as it does in the real game.
                            _put(blocks, x + dx, y + dy, z + dz,
                                 ORE_DEEP_ID[k] if (y + dy) < DEEPSLATE_LEVEL
                                 else ORE_ID[k])


# ---------------------------------------------------------------------------
# Villages
# ---------------------------------------------------------------------------
# The reference builds these from ~36 hand-made templates per style, wired
# together by a jigsaw over a street network (`structure/village/plains/houses`,
# `.../streets`, `.../town_centers`). We cannot read those at runtime, so the
# variety has to be generated — but the *shape* of the variety is copied: a
# village is a street grid with plots in its squares, and a plot holds one of
# several kinds of building, not one building with different dimensions.
#
# Nothing about a village is stored. Every number is a hash of a world position,
# so each of the chunks a building straddles derives exactly the same box.
@njit(nogil=True, fastmath=True, cache=True)
def village_site(region_x, region_z):
    """Where the village in this region would stand, before checking whether it
    can. Two hashes, so a chunk can reject far-off regions without touching the
    noise at all."""
    jx = int(fast_rand(region_x, 501.0, region_z) * VILLAGE_JITTER)
    jz = int(fast_rand(region_x, 907.0, region_z) * VILLAGE_JITTER)
    cx = region_x * VILLAGE_SPACING + jx
    cz = region_z * VILLAGE_SPACING + jz
    return cx * CHUNK_SIZE + 8, cz * CHUNK_SIZE + 8


@njit(nogil=True, fastmath=True, cache=True)
def terrain_top(wx, wz):
    """Ground level at one world column, off the lattice the chunks build from.

    Reading the raw noise instead gives a *different* world — the splines turn a
    small climate difference into several blocks of height — so everything
    outside the chunk loop that needs a real height comes through here.
    """
    c = climate_interp(wx, wz)
    height, _river, _alt = column_height(c[0], c[1], c[2], wx, wz)
    return int(height)


@njit(nogil=True, fastmath=True, cache=True)
def village_check(vx, vz):
    """(ok, floor_y, style) for a candidate site.

    Only the *plaza* has to be level now: buildings stand at the height of their
    own plot and the streets follow the ground, so a village can sit on rolling
    terrain the way the reference's `project_start_to_heightmap` lets it. What
    this still rejects is a site whose middle is a cliff or a river.
    """
    c = climate_interp(vx, vz)
    height, river, alt = column_height(c[0], c[1], c[2], vx, vz)
    biome = biome_at(c[3], c[4], c[0], c[1], c[2], height, alt, river)

    if B_VILLAGE[biome] == 0 or river > 0.05:
        return False, 0, 0

    centre = int(height)
    for i in range(4):
        dx = VILLAGE_PROBE if i == 0 else (-VILLAGE_PROBE if i == 1 else 0)
        dz = VILLAGE_PROBE if i == 2 else (-VILLAGE_PROBE if i == 3 else 0)
        h2 = terrain_top(vx + dx, vz + dz)
        if abs(h2 - centre) > VILLAGE_FLAT or h2 < SEA_LEVEL + 1:
            return False, 0, 0

    return True, centre + 1, B_VILLAGE_STYLE[biome]


@njit(nogil=True, fastmath=True, cache=True)
def _clear_above(blocks, lx, lz, from_y, to_y):
    for y in range(from_y, to_y + 1):
        if _get(blocks, lx, y, lz) != AIR:
            _put(blocks, lx, y, lz, AIR)


@njit(nogil=True, fastmath=True, cache=True)
def _terraform(blocks, lx, lz, floor_y, ground, clear_to):
    """Level one column: ground at floor_y, air above, fill the gap below."""
    _put(blocks, lx, floor_y, lz, ground)
    _clear_above(blocks, lx, lz, floor_y + 1, clear_to)
    # Fill down to whatever is already there. The loop stops at the first solid
    # block, so the bound only bites on a genuine cliff — and it has to reach
    # further than the slope a village is now allowed to stand on, or a plot on
    # a bank leaves the house on stilts.
    for y in range(floor_y - 1, floor_y - 25, -1):
        if y < 0:
            break
        here = _get(blocks, lx, y, lz)
        if here != AIR and here != WATER:
            break
        _put(blocks, lx, y, lz, ground)


@njit(nogil=True, fastmath=True, cache=True)
def _gable_roof(blocks, ox, oz, x0, z0, x1, z1, roof_y, roof, wall, ridge_x):
    """A stepped gable over the box, with the end walls filled in under it.

    A flat slab was what made every house the same house from outside. A roof
    with a ridge gives a building an orientation, and orientation is most of
    what tells two boxes apart at a distance.
    """
    if ridge_x:
        span = z1 - z0 + 2
    else:
        span = x1 - x0 + 2
    levels = span // 2 + 1
    for k in range(levels):
        y = roof_y + k
        for x in range(x0 - 1, x1 + 2):
            for z in range(z0 - 1, z1 + 2):
                if ridge_x:
                    inset = z - (z0 - 1)
                    other = (z1 + 1) - z
                    # The gable triangle stands on the building's own end wall,
                    # not on the eave overhang a block outside it — there it
                    # would be a wall hanging in the air.
                    at_end = x == x0 or x == x1
                else:
                    inset = x - (x0 - 1)
                    other = (x1 + 1) - x
                    at_end = z == z0 or z == z1
                if other < inset:
                    inset = other
                if inset == k:
                    _put(blocks, x - ox, y, z - oz, roof)
                elif inset > k and at_end:
                    _put(blocks, x - ox, y, z - oz, wall)


@njit(nogil=True, fastmath=True, cache=True)
def _build_house(blocks, ox, oz, x0, z0, x1, z1, floor_y, storeys, style, seed):
    """A walled building with windows, a door, a gabled roof and something
    inside it. Used for the house, big house and library kinds."""
    wall = V_WALL[style]
    post = V_POST[style]
    floor_block = V_FLOOR[style]
    roof = V_ROOF[style]
    accent = V_ACCENT[style]

    wall_h = 3 + storeys * 2 + int(fast_rand(seed, 97.0, seed) * 2)
    roof_y = floor_y + wall_h
    ridge_x = (x1 - x0) >= (z1 - z0)

    door_side = int(fast_rand(seed, 131.0, seed) * 4)
    if door_side == 0:
        door_x, door_z = (x0 + x1) // 2, z0
    elif door_side == 1:
        door_x, door_z = (x0 + x1) // 2, z1
    elif door_side == 2:
        door_x, door_z = x0, (z0 + z1) // 2
    else:
        door_x, door_z = x1, (z0 + z1) // 2

    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            lx = x - ox
            lz = z - oz
            _terraform(blocks, lx, lz, floor_y, floor_block, roof_y + 6)

            if not (x == x0 or x == x1 or z == z0 or z == z1):
                continue
            corner = (x == x0 or x == x1) and (z == z0 or z == z1)
            for y in range(floor_y + 1, roof_y):
                block = post if corner else wall
                if not corner:
                    # Windows on every storey's middle course. Beams on the
                    # course above tie the facade together — a flat plane of
                    # planks is what read as "a box" before.
                    level = (y - floor_y - 1) % (wall_h // storeys)
                    if level == 1 and (x + z) % 2 == 0:
                        block = GLASS
                    elif level == 2 and storeys > 1:
                        block = accent
                _put(blocks, lx, y, lz, block)

            if x == door_x and z == door_z:
                _put(blocks, lx, floor_y + 1, lz, AIR)
                _put(blocks, lx, floor_y + 2, lz, AIR)
                _put(blocks, lx, floor_y + 3, lz, accent)

    if storeys > 1:
        # An upper floor, so a tall house is two rooms rather than a hall.
        mid = floor_y + wall_h // 2
        for x in range(x0 + 1, x1):
            for z in range(z0 + 1, z1):
                _put(blocks, x - ox, mid, z - oz, V_ACCENT[style])

    _gable_roof(blocks, ox, oz, x0, z0, x1, z1, roof_y, roof, wall, ridge_x)

    # Furniture. Not decoration for its own sake: an empty shell reads as a
    # prop, and one crafting table in every house read as the same prop.
    fx = x0 + 1
    fz = z0 + 1
    _put(blocks, fx - ox, floor_y + 1, fz - oz, CRAFTING_TABLE)
    _put(blocks, x1 - 1 - ox, floor_y + 1, z1 - 1 - oz, BARREL)
    kind = fast_rand(seed, 149.0, seed)
    if kind < 0.35:
        _put(blocks, fx - ox, floor_y + 1, z1 - 1 - oz, BOOKSHELF)
        _put(blocks, fx - ox, floor_y + 2, z1 - 1 - oz, BOOKSHELF)
    elif kind < 0.75:
        bed = V_BED[style]
        _put(blocks, x1 - 1 - ox, floor_y + 1, fz - oz, bed)
        _put(blocks, x1 - 1 - ox, floor_y + 1, fz + 1 - oz, bed)
    _put(blocks, (x0 + x1) // 2 - ox, roof_y - 1, (z0 + z1) // 2 - oz, GLOWSTONE)


@njit(nogil=True, fastmath=True, cache=True)
def _build_farm(blocks, ox, oz, x0, z0, x1, z1, floor_y, style, seed):
    """Crop rows with a water channel, inside a low kerb — the reference's
    `plains_small_farm` / `large_farm`."""
    kerb = V_WALL[style]
    crop = V_CROP[style]
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            lx = x - ox
            lz = z - oz
            edge = x == x0 or x == x1 or z == z0 or z == z1
            _terraform(blocks, lx, lz, floor_y, COARSE_DIRT, floor_y + 6)
            if edge:
                _put(blocks, lx, floor_y, lz, kerb)
                continue
            row = (z - z0) % 3
            if row == 0:
                _put(blocks, lx, floor_y, lz, CLAY)
                _put(blocks, lx, floor_y + 1, lz, WATER)
            elif fast_rand(x, 157.0, z) > 0.18:
                _put(blocks, lx, floor_y + 1, lz, crop)


@njit(nogil=True, fastmath=True, cache=True)
def _build_pen(blocks, ox, oz, x0, z0, x1, z1, floor_y, style, seed):
    """An animal pen: a knee-high wall round bare ground, with a feed trough."""
    post = V_POST[style]
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            lx = x - ox
            lz = z - oz
            _terraform(blocks, lx, lz, floor_y, COARSE_DIRT, floor_y + 5)
            edge = x == x0 or x == x1 or z == z0 or z == z1
            gate = (x == (x0 + x1) // 2 and (z == z0 or z == z1))
            if edge and not gate:
                _put(blocks, lx, floor_y + 1, lz, post)
                _put(blocks, lx, floor_y + 2, lz, post)
    _put(blocks, (x0 + x1) // 2 - ox, floor_y + 1, (z0 + z1) // 2 - oz, HAY_BALE)


@njit(nogil=True, fastmath=True, cache=True)
def _build_tower(blocks, ox, oz, x0, z0, x1, z1, floor_y, style, seed):
    """A lookout. Every village needs one thing taller than the roofs or the
    skyline is flat wherever you stand."""
    wall = V_WALL[style]
    post = V_POST[style]
    height = 8 + int(fast_rand(seed, 163.0, seed) * 5)
    top = floor_y + height
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            lx = x - ox
            lz = z - oz
            _terraform(blocks, lx, lz, floor_y, V_FLOOR[style], top + 2)
            edge = x == x0 or x == x1 or z == z0 or z == z1
            if not edge:
                continue
            for y in range(floor_y + 1, top):
                _put(blocks, lx, y, lz, wall if (y - floor_y) % 4 != 0 else post)
            # Crenellations, gapped, so the top is a silhouette and not a rim.
            if (x + z) % 2 == 0:
                _put(blocks, lx, top, lz, post)
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            if x != x0 and x != x1 and z != z0 and z != z1:
                _put(blocks, x - ox, top - 1, z - oz, V_ACCENT[style])
    _put(blocks, (x0 + x1) // 2 - ox, top - 2, (z0 + z1) // 2 - oz, GLOWSTONE)
    _put(blocks, x0 + 1 - ox, floor_y + 1, z0 + 1 - oz, AIR)


@njit(nogil=True, fastmath=True, cache=True)
def build_village(blocks, chunk_x, chunk_z, vx, vz, floor_y, style, heights):
    """Write whatever of this village falls inside this chunk.

    Streets first, then buildings, so a wall overwrites the road it stands on
    rather than the other way round.
    """
    ox = chunk_x * CHUNK_SIZE
    oz = chunk_z * CHUNK_SIZE
    path = V_PATH[style]
    post = V_POST[style]

    # --- streets ----------------------------------------------------------
    # A grid of lines through the plot boundaries, with most of them switched
    # off by a hash: what is left is a main street and a couple of crossings,
    # which is the irregular layout the reference's jigsaw arrives at. Streets
    # follow the ground instead of levelling it, so a village can sit on a
    # slope without standing on a plinth.
    for lx in range(CHUNK_SIZE):
        for lz in range(CHUNK_SIZE):
            dx = ox + lx - vx
            dz = oz + lz - vz
            if abs(dx) > VILLAGE_RADIUS or abs(dz) > VILLAGE_RADIUS:
                continue

            gx = int(round(dx / VILLAGE_CELL))
            gz = int(round(dz / VILLAGE_CELL))
            on_x = (abs(dx - gx * VILLAGE_CELL) <= 1
                    and (gx == 0 or fast_rand(vx + gx, 211.0, vz) < 0.55))
            on_z = (abs(dz - gz * VILLAGE_CELL) <= 1
                    and (gz == 0 or fast_rand(vx, 223.0, vz + gz) < 0.55))
            plaza = abs(dx) <= 5 and abs(dz) <= 5
            if not (on_x or on_z or plaza):
                continue

            ground = heights[lx + PAD, lz + PAD]
            if ground < SEA_LEVEL:
                continue
            _put(blocks, lx, ground, lz, path)
            _clear_above(blocks, lx, lz, ground + 1, ground + 6)
            # Lamp posts down the street, spaced but not regular.
            if ((on_x or on_z) and not plaza
                    and fast_rand(ox + lx, 227.0, oz + lz) < 0.005):
                for y in range(ground + 1, ground + 4):
                    _put(blocks, lx, y, lz, post)
                _put(blocks, lx, ground + 4, lz, GLOWSTONE)

    # --- the well in the middle of the plaza -------------------------------
    for dx in range(-2, 3):
        for dz in range(-2, 3):
            lx = vx + dx - ox
            lz = vz + dz - oz
            if abs(dx) == 2 or abs(dz) == 2:
                if abs(dx) == 2 and abs(dz) == 2:
                    for y in range(floor_y, floor_y + 4):
                        _put(blocks, lx, y, lz, V_POST[style])
                    _put(blocks, lx, floor_y + 4, lz, V_ROOF[style])
                continue
            if dx == 0 and dz == 0:
                _put(blocks, lx, floor_y, lz, WATER)
                _put(blocks, lx, floor_y + 1, lz, WATER)
            else:
                _put(blocks, lx, floor_y, lz, COBBLESTONE)
                _put(blocks, lx, floor_y + 1, lz, COBBLESTONE)
    for dx in range(-2, 3):
        for dz in range(-2, 3):
            if abs(dx) == 2 or abs(dz) == 2:
                _put(blocks, vx + dx - ox, floor_y + 4, vz + dz - oz, V_ROOF[style])

    # --- buildings --------------------------------------------------------
    # Plots sit in the squares of the street grid, half a cell off the lines.
    reach = 1 + int(fast_rand(vx, 233.0, vz) * 2)      # a 3x3 or 5x5 of plots
    for gx in range(-reach, reach + 1):
        for gz in range(-reach, reach + 1):
            plot_x = vx + gx * VILLAGE_CELL + VILLAGE_CELL // 2
            plot_z = vz + gz * VILLAGE_CELL + VILLAGE_CELL // 2
            seed = plot_x * 3.0 + plot_z * 7.0
            if fast_rand(plot_x, 31.0, plot_z) < 0.28:
                continue      # an empty lot, so the grid does not read as a grid

            kind_roll = fast_rand(plot_x, 37.0, plot_z)
            if kind_roll < 0.32:
                kind = VB_HOUSE
                w, d = 5, 5
            elif kind_roll < 0.48:
                kind = VB_BIG_HOUSE
                w, d = 8, 7
            elif kind_roll < 0.64:
                kind = VB_FARM
                w, d = 8, 8
            elif kind_roll < 0.76:
                kind = VB_PEN
                w, d = 8, 7
            elif kind_roll < 0.84:
                kind = VB_TOWER
                w, d = 4, 4
            else:
                kind = VB_LIBRARY
                w, d = 7, 6
            w += int(fast_rand(plot_x, 53.0, plot_z) * 3)
            d += int(fast_rand(plot_x, 71.0, plot_z) * 3)
            if fast_rand(plot_x, 79.0, plot_z) < 0.5:
                w, d = d, w                      # half the buildings turned 90°

            # Jitter inside the plot, so the row of houses is not a row.
            plot_x += int(fast_rand(plot_x, 83.0, plot_z) * 5) - 2
            plot_z += int(fast_rand(plot_x, 89.0, plot_z) * 5) - 2

            x0 = plot_x - w // 2
            z0 = plot_z - d // 2
            x1 = x0 + w - 1
            z1 = z0 + d - 1

            # Skip the whole building when it cannot touch this chunk. This is
            # the test that keeps the pass off the profile — it runs before the
            # ground probe, which is the only expensive thing here.
            if (x1 < ox - 2 or x0 > ox + CHUNK_SIZE + 1
                    or z1 < oz - 2 or z0 > oz + CHUNK_SIZE + 1):
                continue

            # Each building stands at the height of its own plot, not the
            # village's. That is what lets a village follow rolling ground
            # instead of carving a mesa out of it — the reference does the same
            # with `project_start_to_heightmap`. Clamped, so a plot that landed
            # on a bank does not leave a house halfway up a wall.
            plot_y = terrain_top(plot_x, plot_z) + 1
            if plot_y < floor_y - 4:
                plot_y = floor_y - 4
            elif plot_y > floor_y + 4:
                plot_y = floor_y + 4
            if plot_y <= SEA_LEVEL:
                continue

            if kind == VB_FARM:
                _build_farm(blocks, ox, oz, x0, z0, x1, z1, plot_y, style, seed)
            elif kind == VB_PEN:
                _build_pen(blocks, ox, oz, x0, z0, x1, z1, plot_y, style, seed)
            elif kind == VB_TOWER:
                _build_tower(blocks, ox, oz, x0, z0, x1, z1, plot_y, style, seed)
            elif kind == VB_BIG_HOUSE:
                _build_house(blocks, ox, oz, x0, z0, x1, z1, plot_y, 2, style, seed)
            elif kind == VB_LIBRARY:
                _build_house(blocks, ox, oz, x0, z0, x1, z1, plot_y, 1, style, seed)
                for x in range(x0 + 1, x1):
                    _put(blocks, x - ox, plot_y + 1, z0 + 1 - oz, BOOKSHELF)
                    _put(blocks, x - ox, plot_y + 2, z0 + 1 - oz, BOOKSHELF)
            else:
                _build_house(blocks, ox, oz, x0, z0, x1, z1, plot_y, 1, style, seed)

# ---------------------------------------------------------------------------
# The chunk kernel
# ---------------------------------------------------------------------------
@njit(nogil=True, fastmath=True, cache=True)
def generate_chunk_fast(chunk_x, chunk_z, blocks):
    """Fill one chunk. *blocks* arrives zeroed (AIR).

    Eight passes, in an order the later ones depend on: climate -> columns ->
    caves -> ores -> find villages -> trees -> build villages. Where the
    villages are has to be known *before* the trees, or a street would have to
    delete a tree it found standing on it — and it could only delete the half in
    its own chunk, leaving the rest of the canopy hanging over the seam.
    """
    ox = chunk_x * CHUNK_SIZE
    oz = chunk_z * CHUNK_SIZE

    # --- 1. climate, on a 4-block grid across the padded span --------------
    grid = np.empty((5, GRID_N, GRID_N), dtype=np.float64)
    for i in range(GRID_N):
        for j in range(GRID_N):
            wx = ox - PAD + i * GRID_STEP
            wz = oz - PAD + j * GRID_STEP
            cont, ero, weird, temp, humid = climate_at(wx, wz)
            grid[0, i, j] = cont
            grid[1, i, j] = ero
            grid[2, i, j] = weird
            grid[3, i, j] = temp
            grid[4, i, j] = humid

    # --- 2. height and biome per column, chunk only ------------------------
    # The padded ring is *not* filled here. It exists only so a canopy rooted
    # next door crosses the seam, which is a handful of columns in a hundred;
    # computing all of it cost 47% of this loop to answer a question the tree
    # pass rejects with one hash. The ring is filled on demand there instead.
    heights = np.empty((SPAN, SPAN), dtype=np.int32)
    biomes = np.empty((SPAN, SPAN), dtype=np.int32)
    top_y = 0
    for p in range(PAD, PAD + CHUNK_SIZE):
        for q in range(PAD, PAD + CHUNK_SIZE):
            wx = ox - PAD + p
            wz = oz - PAD + q
            cont = _bilerp(grid, 0, p, q)
            ero = _bilerp(grid, 1, p, q)
            weird = _bilerp(grid, 2, p, q)
            temp = _bilerp(grid, 3, p, q)
            humid = _bilerp(grid, 4, p, q)

            height, river, alt = column_height(cont, ero, weird, wx, wz)
            h = int(height)
            if h >= CHUNK_HEIGHT - 12:
                h = CHUNK_HEIGHT - 12
            heights[p, q] = h
            biomes[p, q] = biome_at(temp, humid, cont, ero, weird, height,
                                    alt, river)
            if h > top_y:
                top_y = h

    # --- 3. caves, on a 4-block grid in all three axes ---------------------
    # The same idea as the climate grid and the reason it is here: a threshold
    # per block cost one 3D sample for every block underground, which was the
    # most expensive thing the generator did by an order of magnitude.
    cave_top = top_y - CAVE_ROOF
    ncy = 2 if cave_top < CAVE_FLOOR else (cave_top - CAVE_FLOOR) // CAVE_STEP + 2
    ncx = CHUNK_SIZE // CAVE_STEP + 1
    cave = np.empty((2, ncx, ncx, ncy), dtype=np.float64)
    for i in range(ncx):
        for j in range(ncx):
            for k in range(ncy):
                wx = (ox + i * CAVE_STEP) / CAVE_WL
                wz = (oz + j * CAVE_STEP) / CAVE_WL
                wy = (CAVE_FLOOR + k * CAVE_STEP) / CAVE_WL
                cave[0, i, j, k] = fast_noise3(wx, wy, wz)
                cave[1, i, j, k] = fast_noise3(wx + 71.5, wy + 13.5, wz - 47.5)

    # --- 4. columns --------------------------------------------------------
    for lx in range(CHUNK_SIZE):
        for lz in range(CHUNK_SIZE):
            h = heights[lx + PAD, lz + PAD]
            biome = biomes[lx + PAD, lz + PAD]
            underwater = h < SEA_LEVEL

            top = B_UNDER[biome] if underwater else B_TOP[biome]
            fill = B_UNDER[biome] if underwater else B_FILL[biome]
            fill_depth = 3 + int(fast_rand(ox + lx, 5.0, oz + lz) * 2)

            # Surface patches. One noise sample per column of the chunk proper —
            # not of the padded span, since a patch is not a feature and nothing
            # outside reads it. The hash term frays the patch edge so a disk
            # stops being a circle drawn on the ground.
            patch = _octave2(ox + lx + 5500.5, oz + lz - 2200.5,
                             1.0 / PATCH_WL, 2, 0.5)
            patch += (fast_rand(ox + lx, 17.0, oz + lz) - 0.5) * PATCH_DITHER
            if underwater:
                # The reference puts clay and sand under the water rather than
                # the biome's own bed; a lake floor of one block reads as tiling.
                if biome == WARM_OCEAN and patch > PATCH_B_T:
                    top = CORALS[int(fast_rand(ox + lx, 43.0, oz + lz) * 5)]
                elif patch > PATCH_A_T:
                    top = CLAY
                elif patch < PATCH_B_T:
                    top = SAND
            elif patch > PATCH_A_T:
                top = B_PATCH_A[biome]
            elif patch < PATCH_B_T:
                top = B_PATCH_B[biome]

            ci = lx // CAVE_STEP
            cj = lz // CAVE_STEP
            cfx = (lx - ci * CAVE_STEP) / CAVE_STEP
            cfz = (lz - cj * CAVE_STEP) / CAVE_STEP

            for y in range(h + 1):
                if y == 0:
                    blocks[lx, y, lz] = BEDROCK
                    continue
                if y < 3 and fast_rand(ox + lx, y, oz + lz) < 0.6:
                    blocks[lx, y, lz] = BEDROCK
                    continue

                if y == h:
                    block = top
                elif y > h - fill_depth:
                    block = fill
                elif y < DEEPSLATE_LEVEL:
                    block = DEEPSLATE
                else:
                    block = STONE

                # Badlands wear their strata on the outside, so the banding has
                # to replace the fill rather than sit under it.
                if biome == BADLANDS and not underwater and h - y < 18 and y > SEA_LEVEL:
                    if y > h - fill_depth:
                        band = (y + int(fast_rand(ox + lx, 3.0, oz + lz) * 3)) % 16
                        block = BADLANDS_BANDS[band]

                # Caves. Two interpolated fields, both near zero -> a tunnel.
                if CAVE_FLOOR <= y <= h - CAVE_ROOF:
                    ck = (y - CAVE_FLOOR) // CAVE_STEP
                    if ck < ncy - 1:
                        cfy = (y - CAVE_FLOOR - ck * CAVE_STEP) / CAVE_STEP
                        carve = True
                        for f in range(2):
                            c00 = cave[f, ci, cj, ck] + (cave[f, ci, cj, ck + 1] - cave[f, ci, cj, ck]) * cfy
                            c10 = cave[f, ci + 1, cj, ck] + (cave[f, ci + 1, cj, ck + 1] - cave[f, ci + 1, cj, ck]) * cfy
                            c01 = cave[f, ci, cj + 1, ck] + (cave[f, ci, cj + 1, ck + 1] - cave[f, ci, cj + 1, ck]) * cfy
                            c11 = cave[f, ci + 1, cj + 1, ck] + (cave[f, ci + 1, cj + 1, ck + 1] - cave[f, ci + 1, cj + 1, ck]) * cfy
                            a = c00 + (c01 - c00) * cfz
                            b = c10 + (c11 - c10) * cfz
                            if abs(a + (b - a) * cfx) > CAVE_WIDTH:
                                carve = False
                                break
                        if carve:
                            continue

                blocks[lx, y, lz] = block

            # Water fills whatever the column did not reach.
            if underwater:
                for y in range(h + 1, SEA_LEVEL):
                    blocks[lx, y, lz] = WATER

    # --- 5. ores -----------------------------------------------------------
    centre_biome = biomes[PAD + 8, PAD + 8]
    mountain = (centre_biome == WINDSWEPT_HILLS or centre_biome == JAGGED_PEAKS
                or centre_biome == STONY_PEAKS or centre_biome == SNOWY_SLOPES
                or centre_biome == GROVE or centre_biome == MEADOW)
    place_ores(blocks, chunk_x, chunk_z, top_y, mountain)

    # --- 6. which villages reach this chunk --------------------------------
    # Found before the trees, not after, because a village clears its ground and
    # a tree that was there first would lose its trunk to a street while its
    # canopy stayed — and worse, only in the chunks the street runs through, so
    # half a canopy would be left hanging at the seam. A village site is two
    # hashes to reject, so scanning here costs nothing and the list is reused by
    # the build pass below.
    n_villages = 0
    v_x = np.zeros(9, dtype=np.int64)
    v_z = np.zeros(9, dtype=np.int64)
    v_y = np.zeros(9, dtype=np.int64)
    v_style = np.zeros(9, dtype=np.int64)
    region_x = int(math.floor(chunk_x / VILLAGE_SPACING))
    region_z = int(math.floor(chunk_z / VILLAGE_SPACING))
    for rx in range(region_x - 1, region_x + 2):
        for rz in range(region_z - 1, region_z + 2):
            vx, vz = village_site(rx, rz)
            if (abs(vx - (ox + 8)) > VILLAGE_RADIUS + CHUNK_SIZE
                    or abs(vz - (oz + 8)) > VILLAGE_RADIUS + CHUNK_SIZE):
                continue
            ok, floor_y, style = village_check(vx, vz)
            if ok:
                v_x[n_villages] = vx
                v_z[n_villages] = vz
                v_y[n_villages] = floor_y
                v_style[n_villages] = style
                n_villages += 1

    # --- 7. trees, over the padded span so canopies cross chunk seams ------
    # Ascending p then q, which is ascending world x then z. Two chunks that
    # share a column must place the trees around it in the same order, because
    # leaves only go where there is air and so a later canopy defers to an
    # earlier one. Ascending world order is the same sequence seen from either
    # side; a chunk-relative order would not be.
    for p in range(SPAN):
        for q in range(SPAN):
            wx = ox - PAD + p
            wz = oz - PAD + q
            roll = fast_rand(wx, 17.0, wz) * 10000.0
            rock = fast_rand(wx, 19.0, wz) * 10000.0
            if roll >= TREE_RATE_MAX and rock >= ROCK_RATE_MAX:
                continue          # neither, in any biome — two hashes

            in_village = False
            for v in range(n_villages):
                if (abs(wx - v_x[v]) <= VILLAGE_RADIUS
                        and abs(wz - v_z[v]) <= VILLAGE_RADIUS):
                    in_village = True
                    break
            if in_village:
                continue

            inside = (PAD <= p < PAD + CHUNK_SIZE and PAD <= q < PAD + CHUNK_SIZE)
            if inside:
                h = heights[p, q]
                biome = biomes[p, q]
            else:
                cont = _bilerp(grid, 0, p, q)
                ero = _bilerp(grid, 1, p, q)
                weird = _bilerp(grid, 2, p, q)
                height, river, alt = column_height(cont, ero, weird, wx, wz)
                h = int(height)
                if h >= CHUNK_HEIGHT - 12:
                    h = CHUNK_HEIGHT - 12
                biome = biome_at(_bilerp(grid, 3, p, q), _bilerp(grid, 4, p, q),
                                 cont, ero, weird, height, alt, river)

            if h <= SEA_LEVEL:
                continue
            if rock < B_ROCKS[biome]:
                place_boulder(blocks, p - PAD, h, q - PAD, B_ROCK_BLOCK[biome],
                              wx, wz)
            if roll >= B_TREES[biome]:
                continue

            # Which of the biome's trees this one is. The reference's
            # random_selector, as a cumulative table.
            pick = fast_rand(wx, 23.0, wz)
            slot = 0
            while slot < B_TREE_N[biome] - 1 and pick > B_TREE_CUM[biome, slot]:
                slot += 1
            place_tree(blocks, p - PAD, h + 1, q - PAD, B_TREE_SHAPE[biome, slot],
                       B_TREE_LOG[biome, slot], B_TREE_LEAF[biome, slot], wx, wz)

    # --- 8. villages -------------------------------------------------------
    for v in range(n_villages):
        build_village(blocks, chunk_x, chunk_z, v_x[v], v_z[v], v_y[v],
                      v_style[v], heights)

    # --- 9. ground cover ---------------------------------------------------
    # Last, and chunk-only. Last because a plant defers to everything else: it
    # goes where the ground is still the biome's own and the cell above it is
    # still empty, so a trunk, a street or a house floor simply leaves no room
    # and nothing has to be deleted afterwards. Chunk-only because a plant is
    # one column wide — the padded ring exists for canopies, and nothing here
    # reaches across a seam.
    for lx in range(CHUNK_SIZE):
        for lz in range(CHUNK_SIZE):
            wx = ox + lx
            wz = oz + lz
            roll = fast_rand(wx, 29.0, wz) * 10000.0
            if roll >= PLANT_RATE_MAX:
                continue                      # one hash rejects most of the map
            biome = biomes[lx + PAD, lz + PAD]
            if roll >= B_PLANTS[biome]:
                continue

            h = heights[lx + PAD, lz + PAD]
            if h + 2 >= CHUNK_HEIGHT:
                continue

            # Seagrass grows in the water and everything else grows out of it.
            underwater = h < SEA_LEVEL - 1
            ground = blocks[lx, h, lz]
            above = AIR if not underwater else WATER
            if blocks[lx, h + 1, lz] != above:
                continue
            if underwater:
                if ground != SAND and ground != GRAVEL and ground != CLAY:
                    continue
            elif (ground != GRASS and ground != SNOWY_GRASS and ground != PODZOL
                    and ground != DIRT and ground != MUD and ground != SAND
                    and ground != RED_SAND):
                # Everything else is a village street, a rock, a patch of
                # gravel or bare stone — the ground the reference will not put a
                # plant on either.
                continue

            pick = fast_rand(wx, 31.0, wz)
            slot = 0
            while slot < B_PLANT_N[biome] - 1 and pick > B_PLANT_CUM[biome, slot]:
                slot += 1

            # A two-block plant goes down whole or not at all, the same rule the
            # doors follow: a lower half on its own is a different plant with
            # the top sawn off it.
            top = B_PLANT_TOP[biome, slot]
            if top != 0 and blocks[lx, h + 2, lz] != above:
                continue
            blocks[lx, h + 1, lz] = B_PLANT_ID[biome, slot]
            if top != 0:
                blocks[lx, h + 2, lz] = top


class AdvancedTerrainGenerator:
    """Owns nothing but the seed and the JIT warm-up."""

    def __init__(self, seed=WORLD_SEED):
        self.seed = seed
        print("Warming up Terrain Generator JIT...")
        # Same dtype as ModernChunk.blocks — numba specialises on it, so warming
        # the wrong one just compiles a second copy at the first real chunk.
        dummy = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=BLOCK_DTYPE)
        generate_chunk_fast(0, 0, dummy)
        print("Terrain Generator JIT Ready.")

    def generate_chunk_terrain(self, chunk_x, chunk_z, blocks):
        blocks.fill(AIR)
        generate_chunk_fast(chunk_x, chunk_z, blocks)


# Global terrain generator instance
terrain_generator = AdvancedTerrainGenerator()
