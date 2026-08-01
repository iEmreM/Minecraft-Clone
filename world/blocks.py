"""Block registry — the one table that says what a block is called and what it
looks like.

Everything downstream is derived from `_TABLE` below:

    FACE_LAYER   (id, face_id) -> atlas layer, the array the mesher indexes
    BLOCK_NAMES  id -> display name
    ICON_LAYER   id -> the layer the HUD draws for it
    CREATIVE     ids in group order, for the creative window
    TEXTURES     atlas layer order, which is what build_atlas.py bakes

Adding a block is one row here plus a rerun of `python build_atlas.py`.

Textures come from the real game's files under `referans/`, at their original
16x16. That folder is gitignored, so the atlas is baked once into `texture.png`
and committed — the game never reads `referans/` at runtime.
"""

import numpy as np

# Face order, fixed by fast_builder.emit_greedy_quad. `front` is the +Z face; we
# have no per-block rotation, so a furnace always faces the same way.
TOP, BOTTOM, FRONT, BACK, RIGHT, LEFT = range(6)

# Greyscale masters that the real game tints at runtime from the biome colormap.
# We have no biomes, so they are tinted once at atlas build time with the plains
# colours (referans/assets/minecraft/textures/colormap/grass.png, plains corner).
GRASS_TINT = (0x91, 0xBD, 0x59)
FOLIAGE_TINT = (0x77, 0xAB, 0x2F)
BIRCH_TINT = (0x80, 0xA7, 0x55)

TINTS = {
    'grass_block_top': GRASS_TINT,
    'grass_block_side_overlay': GRASS_TINT,
    'oak_leaves': FOLIAGE_TINT,
    'jungle_leaves': FOLIAGE_TINT,
    'acacia_leaves': FOLIAGE_TINT,
    'dark_oak_leaves': FOLIAGE_TINT,
    'birch_leaves': BIRCH_TINT,
}

# Textures that do not exist as a file: (base, overlay) stacked at build time,
# the overlay tinted by TINTS on the way. This is how the real game draws a
# grass block's side, and doing it here is what keeps the side's green matching
# the top's — `grass_block_side.png` ships with its own baked-in green that
# would not.
COMPOSITES = {
    'grass_block_side_tinted': ('dirt', 'grass_block_side_overlay'),
}

# (id, display name, textures)
#   1 name  -> all six faces
#   3 names -> (top, side, bottom)
#   4 names -> (top, side, bottom, front)
#
# IDs 1-10 are load-bearing: terrain generation and every saved chunk use them.
# The assert at the bottom of this file catches anyone renumbering them.
_TABLE = [
    ('Doğal', [
        (1, 'Grass Block', ('grass_block_top', 'grass_block_side_tinted', 'dirt')),
        (2, 'Dirt', 'dirt'),
        (3, 'Stone', 'stone'),
        (4, 'Sand', 'sand'),
        (5, 'Snow', 'snow'),
        (11, 'Cobblestone', 'cobblestone'),
        (12, 'Gravel', 'gravel'),
        (13, 'Andesite', 'andesite'),
        (14, 'Diorite', 'diorite'),
        (15, 'Granite', 'granite'),
        (16, 'Deepslate', ('deepslate_top', 'deepslate', 'deepslate_top')),
        (17, 'Cobbled Deepslate', 'cobbled_deepslate'),
        (18, 'Tuff', 'tuff'),
        (19, 'Calcite', 'calcite'),
        (20, 'Clay', 'clay'),
        (21, 'Red Sand', 'red_sand'),
        (22, 'Sandstone', ('sandstone_top', 'sandstone', 'sandstone_bottom')),
        (23, 'Red Sandstone', ('red_sandstone_top', 'red_sandstone', 'red_sandstone_bottom')),
        (24, 'Podzol', ('podzol_top', 'podzol_side', 'dirt')),
        (25, 'Mycelium', ('mycelium_top', 'mycelium_side', 'dirt')),
        (26, 'Moss Block', 'moss_block'),
        (27, 'Mud', 'mud'),
        (28, 'Packed Ice', 'packed_ice'),
        (29, 'Blue Ice', 'blue_ice'),
        (30, 'Obsidian', 'obsidian'),
        (31, 'Bedrock', 'bedrock'),
        (32, 'Netherrack', 'netherrack'),
        (33, 'Soul Sand', 'soul_sand'),
        (34, 'End Stone', 'end_stone'),
        (35, 'Magma Block', 'magma'),
        (36, 'Amethyst Block', 'amethyst_block'),
        (37, 'Basalt', ('basalt_top', 'basalt_side', 'basalt_top')),
    ]),
    ('Ahşap', [
        (7, 'Oak Log', ('oak_log_top', 'oak_log', 'oak_log_top')),
        (38, 'Birch Log', ('birch_log_top', 'birch_log', 'birch_log_top')),
        (39, 'Spruce Log', ('spruce_log_top', 'spruce_log', 'spruce_log_top')),
        (40, 'Jungle Log', ('jungle_log_top', 'jungle_log', 'jungle_log_top')),
        (41, 'Acacia Log', ('acacia_log_top', 'acacia_log', 'acacia_log_top')),
        (42, 'Dark Oak Log', ('dark_oak_log_top', 'dark_oak_log', 'dark_oak_log_top')),
        (43, 'Cherry Log', ('cherry_log_top', 'cherry_log', 'cherry_log_top')),
        (44, 'Oak Planks', 'oak_planks'),
        (45, 'Birch Planks', 'birch_planks'),
        (46, 'Spruce Planks', 'spruce_planks'),
        (47, 'Jungle Planks', 'jungle_planks'),
        (48, 'Acacia Planks', 'acacia_planks'),
        (49, 'Dark Oak Planks', 'dark_oak_planks'),
        (50, 'Cherry Planks', 'cherry_planks'),
        (51, 'Crimson Planks', 'crimson_planks'),
        (52, 'Warped Planks', 'warped_planks'),
        (53, 'Bamboo Planks', 'bamboo_planks'),
    ]),
    ('Yaprak', [
        (6, 'Oak Leaves', 'oak_leaves'),
        (54, 'Birch Leaves', 'birch_leaves'),
        (55, 'Spruce Leaves', 'spruce_leaves'),
        (56, 'Jungle Leaves', 'jungle_leaves'),
        (57, 'Acacia Leaves', 'acacia_leaves'),
        (58, 'Dark Oak Leaves', 'dark_oak_leaves'),
    ]),
    ('Cevher', [
        (59, 'Coal Ore', 'coal_ore'),
        (60, 'Iron Ore', 'iron_ore'),
        (61, 'Copper Ore', 'copper_ore'),
        (62, 'Gold Ore', 'gold_ore'),
        (63, 'Redstone Ore', 'redstone_ore'),
        (64, 'Lapis Ore', 'lapis_ore'),
        (65, 'Diamond Ore', 'diamond_ore'),
        (66, 'Emerald Ore', 'emerald_ore'),
        (67, 'Coal Block', 'coal_block'),
        (68, 'Iron Block', 'iron_block'),
        (69, 'Copper Block', 'copper_block'),
        (70, 'Gold Block', 'gold_block'),
        (71, 'Redstone Block', 'redstone_block'),
        (72, 'Lapis Block', 'lapis_block'),
        (73, 'Diamond Block', 'diamond_block'),
        (74, 'Emerald Block', 'emerald_block'),
    ]),
    ('Yapı', [
        (9, 'Stone Bricks', 'stone_bricks'),
        (10, 'Bricks', 'bricks'),
        (75, 'Smooth Stone', 'smooth_stone'),
        (76, 'Mossy Cobblestone', 'mossy_cobblestone'),
        (77, 'Mossy Stone Bricks', 'mossy_stone_bricks'),
        (78, 'Cracked Stone Bricks', 'cracked_stone_bricks'),
        (79, 'Chiseled Stone Bricks', 'chiseled_stone_bricks'),
        (80, 'Polished Andesite', 'polished_andesite'),
        (81, 'Polished Diorite', 'polished_diorite'),
        (82, 'Polished Granite', 'polished_granite'),
        (83, 'Polished Deepslate', 'polished_deepslate'),
        (84, 'Deepslate Bricks', 'deepslate_bricks'),
        (85, 'Deepslate Tiles', 'deepslate_tiles'),
        (86, 'Nether Bricks', 'nether_bricks'),
        (87, 'End Stone Bricks', 'end_stone_bricks'),
        (88, 'Quartz Block', ('quartz_block_top', 'quartz_block_side', 'quartz_block_bottom')),
        (89, 'Prismarine', 'prismarine'),
        (90, 'Purpur Block', 'purpur_block'),
        (91, 'Terracotta', 'terracotta'),
        (92, 'Packed Mud', 'packed_mud'),
    ]),
    ('Dekor', [
        (93, 'Crafting Table', ('crafting_table_top', 'crafting_table_side', 'oak_planks',
                                'crafting_table_front')),
        (94, 'Furnace', ('furnace_top', 'furnace_side', 'furnace_top', 'furnace_front')),
        (95, 'Bookshelf', ('oak_planks', 'bookshelf', 'oak_planks')),
        (96, 'TNT', ('tnt_top', 'tnt_side', 'tnt_bottom')),
        (97, 'Pumpkin', ('pumpkin_top', 'pumpkin_side', 'pumpkin_top')),
        (98, 'Carved Pumpkin', ('pumpkin_top', 'pumpkin_side', 'pumpkin_top', 'carved_pumpkin')),
        (99, 'Jack o\'Lantern', ('pumpkin_top', 'pumpkin_side', 'pumpkin_top', 'jack_o_lantern')),
        (100, 'Melon', ('melon_top', 'melon_side', 'melon_top')),
        (101, 'Hay Bale', ('hay_block_top', 'hay_block_side', 'hay_block_top')),
        (102, 'Glowstone', 'glowstone'),
        (103, 'Sea Lantern', 'sea_lantern'),
        (104, 'Note Block', 'note_block'),
        (105, 'Jukebox', ('jukebox_top', 'jukebox_side', 'jukebox_side')),
        (106, 'Sponge', 'sponge'),
        (107, 'Honeycomb Block', 'honeycomb_block'),
        (108, 'Dried Kelp Block', ('dried_kelp_top', 'dried_kelp_side', 'dried_kelp_bottom')),
    ]),
    ('Yün', [
        (109, 'White Wool', 'white_wool'),
        (110, 'Light Gray Wool', 'light_gray_wool'),
        (111, 'Gray Wool', 'gray_wool'),
        (112, 'Black Wool', 'black_wool'),
        (113, 'Brown Wool', 'brown_wool'),
        (114, 'Red Wool', 'red_wool'),
        (115, 'Orange Wool', 'orange_wool'),
        (116, 'Yellow Wool', 'yellow_wool'),
        (117, 'Lime Wool', 'lime_wool'),
        (118, 'Green Wool', 'green_wool'),
        (119, 'Cyan Wool', 'cyan_wool'),
        (120, 'Light Blue Wool', 'light_blue_wool'),
        (121, 'Blue Wool', 'blue_wool'),
        (122, 'Purple Wool', 'purple_wool'),
        (123, 'Magenta Wool', 'magenta_wool'),
        (124, 'Pink Wool', 'pink_wool'),
    ]),
]


def _expand(textures):
    """One / three / four texture names -> the six faces, in face_id order."""
    if isinstance(textures, str):
        return (textures,) * 6
    if len(textures) == 3:
        top, side, bottom = textures
        front = side
    else:
        top, side, bottom, front = textures
    faces = [None] * 6
    faces[TOP] = top
    faces[BOTTOM] = bottom
    faces[FRONT] = front
    faces[BACK] = faces[RIGHT] = faces[LEFT] = side
    return tuple(faces)


GROUPS = [(name, [b[0] for b in rows]) for name, rows in _TABLE]
BLOCK_NAMES = {bid: name for _, rows in _TABLE for bid, name, _ in rows}
BLOCK_FACES = {bid: _expand(tex) for _, rows in _TABLE for bid, _, tex in rows}

# Every block a player can pick, in the order the creative window lists them.
CREATIVE = [bid for _, ids in GROUPS for bid in ids]

# Atlas layer order. Sorted so the baked texture.png is stable across runs —
# a reordered atlas would silently repaint the world.
TEXTURES = sorted({name for faces in BLOCK_FACES.values() for name in faces})
_LAYER_OF = {name: i for i, name in enumerate(TEXTURES)}
LAYER_COUNT = len(TEXTURES)

# The array the mesher indexes: FACE_LAYER[block_type, face_id] -> atlas layer.
# Passed into build_chunk_mesh_fast as an argument rather than read as a numba
# global, because @njit(cache=True) bakes globals into the cached artifact and
# does not invalidate it when they change — a new block would keep the old
# texture until someone deleted __pycache__ by hand.
FACE_LAYER = np.zeros((max(BLOCK_FACES) + 1, 6), dtype=np.int32)
for _bid, _faces in BLOCK_FACES.items():
    for _f, _name in enumerate(_faces):
        FACE_LAYER[_bid, _f] = _LAYER_OF[_name]

# What the HUD draws for a block: its +Z face. That is the bark on a log, the
# door on a furnace and the grassy edge on a grass block — the side that reads
# as the block on a flat icon.
ICON_LAYER = {bid: int(FACE_LAYER[bid, FRONT]) for bid in BLOCK_FACES}

HOTBAR_DEFAULT = [1, 2, 3, 4, 9, 10, 44, 7, 6]

# IDs 1-10 are written into every generated chunk by terrain_generator and into
# every cached one. Renumbering them repaints the whole world, so fail loudly at
# import instead.
_LEGACY = {1: 'Grass Block', 2: 'Dirt', 3: 'Stone', 4: 'Sand', 5: 'Snow',
           6: 'Oak Leaves', 7: 'Oak Log', 9: 'Stone Bricks', 10: 'Bricks'}
assert all(BLOCK_NAMES.get(k) == v for k, v in _LEGACY.items()), \
    'legacy block IDs 1-10 changed meaning; terrain and saved chunks depend on them'
assert 8 not in BLOCK_NAMES, 'ID 8 is WATER — never meshed, keep it out of the table'
