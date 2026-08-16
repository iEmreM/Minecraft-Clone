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

# What ModernChunk.blocks is made of. See the note beside the ID assert at the
# bottom of this file for what widening it past 8 bits cost.
BLOCK_DTYPE = np.uint16

# Face order, fixed by fast_builder.emit_greedy_quad. `front` is the +Z face; we
# have no per-block rotation, so a furnace always faces the same way.
TOP, BOTTOM, FRONT, BACK, RIGHT, LEFT = range(6)

# Greyscale masters that the real game tints at runtime from the biome colormap.
# We have no biomes, so they are tinted once at atlas build time with the plains
# colours (referans/assets/minecraft/textures/colormap/grass.png, plains corner).
GRASS_TINT = (0x91, 0xBD, 0x59)
FOLIAGE_TINT = (0x77, 0xAB, 0x2F)
# Birch and spruce ignore the colormap in the real game too — they carry one
# constant each, so these are the real values, not plains-biome stand-ins.
BIRCH_TINT = (0x80, 0xA7, 0x55)
SPRUCE_TINT = (0x61, 0x99, 0x61)

TINTS = {
    'grass_block_top': GRASS_TINT,
    'grass_block_side_overlay': GRASS_TINT,
    'oak_leaves': FOLIAGE_TINT,
    'jungle_leaves': FOLIAGE_TINT,
    'acacia_leaves': FOLIAGE_TINT,
    'dark_oak_leaves': FOLIAGE_TINT,
    'mangrove_leaves': FOLIAGE_TINT,
    'birch_leaves': BIRCH_TINT,
    'spruce_leaves': SPRUCE_TINT,
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
        (125, 'Coarse Dirt', 'coarse_dirt'),
        (126, 'Rooted Dirt', 'rooted_dirt'),
        (127, 'Bone Block', ('bone_block_top', 'bone_block_side', 'bone_block_top')),
        # The reference model keeps the green top: there, a snow *layer* block
        # sits above and hides it. We have no snow layer, so this one block has
        # to carry the whole look and the top is snow.
        (128, 'Snowy Grass Block', ('snow', 'grass_block_snow', 'dirt')),
        (129, 'Dripstone Block', 'dripstone_block'),
        (130, 'Sculk', 'sculk'),
        (131, 'Muddy Mangrove Roots', ('muddy_mangrove_roots_top',
                'muddy_mangrove_roots_side', 'muddy_mangrove_roots_top')),
        (132, 'Soul Soil', 'soul_soil'),
        (279, 'Budding Amethyst', 'budding_amethyst'),
        (280, 'Pale Moss Block', 'pale_moss_block'),
        (281, 'Resin Block', 'resin_block'),
        (282, 'Wet Sponge', 'wet_sponge'),
        (283, 'Reinforced Deepslate', ('reinforced_deepslate_top',
                'reinforced_deepslate_side', 'reinforced_deepslate_bottom')),
        (284, 'Sculk Catalyst', ('sculk_catalyst_top', 'sculk_catalyst_side',
                'sculk_catalyst_bottom')),
        (285, 'Tube Coral Block', 'tube_coral_block'),
        (286, 'Brain Coral Block', 'brain_coral_block'),
        (287, 'Bubble Coral Block', 'bubble_coral_block'),
        (288, 'Fire Coral Block', 'fire_coral_block'),
        (289, 'Horn Coral Block', 'horn_coral_block'),
        (290, 'Dead Tube Coral Block', 'dead_tube_coral_block'),
        (291, 'Dead Brain Coral Block', 'dead_brain_coral_block'),
        (292, 'Dead Bubble Coral Block', 'dead_bubble_coral_block'),
        (293, 'Dead Fire Coral Block', 'dead_fire_coral_block'),
        (294, 'Dead Horn Coral Block', 'dead_horn_coral_block'),
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
        (133, 'Mangrove Log', ('mangrove_log_top', 'mangrove_log', 'mangrove_log_top')),
        (134, 'Pale Oak Log', ('pale_oak_log_top', 'pale_oak_log', 'pale_oak_log_top')),
        (135, 'Mangrove Planks', 'mangrove_planks'),
        (136, 'Pale Oak Planks', 'pale_oak_planks'),
        (137, 'Bamboo Block', ('bamboo_block_top', 'bamboo_block', 'bamboo_block_top')),
        (138, 'Stripped Oak Log', ('stripped_oak_log_top', 'stripped_oak_log',
                'stripped_oak_log_top')),
        (139, 'Stripped Birch Log', ('stripped_birch_log_top', 'stripped_birch_log',
                'stripped_birch_log_top')),
        (140, 'Stripped Spruce Log', ('stripped_spruce_log_top', 'stripped_spruce_log',
                'stripped_spruce_log_top')),
        (141, 'Stripped Jungle Log', ('stripped_jungle_log_top', 'stripped_jungle_log',
                'stripped_jungle_log_top')),
        (142, 'Stripped Acacia Log', ('stripped_acacia_log_top', 'stripped_acacia_log',
                'stripped_acacia_log_top')),
        (143, 'Stripped Dark Oak Log', ('stripped_dark_oak_log_top',
                'stripped_dark_oak_log', 'stripped_dark_oak_log_top')),
        (144, 'Stripped Cherry Log', ('stripped_cherry_log_top', 'stripped_cherry_log',
                'stripped_cherry_log_top')),
        (145, 'Stripped Mangrove Log', ('stripped_mangrove_log_top',
                'stripped_mangrove_log', 'stripped_mangrove_log_top')),
        (146, 'Stripped Pale Oak Log', ('stripped_pale_oak_log_top',
                'stripped_pale_oak_log', 'stripped_pale_oak_log_top')),
        # "Wood" is the six-sided bark block — same texture as the log's side, on
        # the ends too. No new atlas layers, just rows.
        (295, 'Oak Wood', 'oak_log'),
        (296, 'Birch Wood', 'birch_log'),
        (297, 'Spruce Wood', 'spruce_log'),
        (298, 'Jungle Wood', 'jungle_log'),
        (299, 'Acacia Wood', 'acacia_log'),
        (300, 'Dark Oak Wood', 'dark_oak_log'),
        (301, 'Cherry Wood', 'cherry_log'),
        (302, 'Mangrove Wood', 'mangrove_log'),
        (303, 'Pale Oak Wood', 'pale_oak_log'),
        (304, 'Stripped Oak Wood', 'stripped_oak_log'),
        (305, 'Stripped Birch Wood', 'stripped_birch_log'),
        (306, 'Stripped Spruce Wood', 'stripped_spruce_log'),
        (307, 'Stripped Jungle Wood', 'stripped_jungle_log'),
        (308, 'Stripped Acacia Wood', 'stripped_acacia_log'),
        (309, 'Stripped Dark Oak Wood', 'stripped_dark_oak_log'),
        (310, 'Stripped Cherry Wood', 'stripped_cherry_log'),
        (311, 'Stripped Mangrove Wood', 'stripped_mangrove_log'),
        (312, 'Stripped Pale Oak Wood', 'stripped_pale_oak_log'),
        (313, 'Stripped Bamboo Block', ('stripped_bamboo_block_top',
                'stripped_bamboo_block', 'stripped_bamboo_block_top')),
    ]),
    ('Yaprak', [
        (6, 'Oak Leaves', 'oak_leaves'),
        (54, 'Birch Leaves', 'birch_leaves'),
        (55, 'Spruce Leaves', 'spruce_leaves'),
        (56, 'Jungle Leaves', 'jungle_leaves'),
        (57, 'Acacia Leaves', 'acacia_leaves'),
        (58, 'Dark Oak Leaves', 'dark_oak_leaves'),
        (147, 'Cherry Leaves', 'cherry_leaves'),
        (148, 'Mangrove Leaves', 'mangrove_leaves'),
        (149, 'Pale Oak Leaves', 'pale_oak_leaves'),
        (150, 'Azalea Leaves', 'azalea_leaves'),
        (314, 'Flowering Azalea Leaves', 'flowering_azalea_leaves'),
        # Snow settles on whatever faces up, which for a canopy is its leaves.
        # The real game stacks a snow layer block on top; we have no thin
        # blocks, so the snow lives on the leaf's top face — the same trick
        # `grass_block_snow` uses there. Only the exposed tops of a canopy show
        # it, which is exactly where snow would be.
        (359, 'Snowy Spruce Leaves', ('snow', 'spruce_leaves', 'spruce_leaves')),
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
        (151, 'Deepslate Coal Ore', 'deepslate_coal_ore'),
        (152, 'Deepslate Iron Ore', 'deepslate_iron_ore'),
        (153, 'Deepslate Copper Ore', 'deepslate_copper_ore'),
        (154, 'Deepslate Gold Ore', 'deepslate_gold_ore'),
        (155, 'Deepslate Redstone Ore', 'deepslate_redstone_ore'),
        (156, 'Deepslate Lapis Ore', 'deepslate_lapis_ore'),
        (157, 'Deepslate Diamond Ore', 'deepslate_diamond_ore'),
        (158, 'Deepslate Emerald Ore', 'deepslate_emerald_ore'),
        (159, 'Nether Gold Ore', 'nether_gold_ore'),
        (160, 'Nether Quartz Ore', 'nether_quartz_ore'),
        (161, 'Raw Iron Block', 'raw_iron_block'),
        (162, 'Raw Copper Block', 'raw_copper_block'),
        (163, 'Raw Gold Block', 'raw_gold_block'),
        (164, 'Netherite Block', 'netherite_block'),
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
        (165, 'Mud Bricks', 'mud_bricks'),
        (166, 'Tuff Bricks', 'tuff_bricks'),
        (167, 'Polished Tuff', 'polished_tuff'),
        (168, 'Chiseled Tuff', ('chiseled_tuff_top', 'chiseled_tuff',
                'chiseled_tuff_top')),
        (169, 'Chiseled Deepslate', 'chiseled_deepslate'),
        (170, 'Cracked Deepslate Bricks', 'cracked_deepslate_bricks'),
        (171, 'Smooth Sandstone', 'sandstone_top'),
        (172, 'Cut Sandstone', ('sandstone_top', 'cut_sandstone', 'sandstone_top')),
        (173, 'Chiseled Sandstone', ('sandstone_top', 'chiseled_sandstone',
                'sandstone_top')),
        (174, 'Smooth Red Sandstone', 'red_sandstone_top'),
        (175, 'Cut Red Sandstone', ('red_sandstone_top', 'cut_red_sandstone',
                'red_sandstone_top')),
        (176, 'Chiseled Red Sandstone', ('red_sandstone_top', 'chiseled_red_sandstone',
                'red_sandstone_top')),
        (177, 'Quartz Bricks', 'quartz_bricks'),
        (178, 'Quartz Pillar', ('quartz_pillar_top', 'quartz_pillar',
                'quartz_pillar_top')),
        (179, 'Chiseled Quartz', ('chiseled_quartz_block_top', 'chiseled_quartz_block',
                'chiseled_quartz_block_top')),
        (180, 'Prismarine Bricks', 'prismarine_bricks'),
        (181, 'Dark Prismarine', 'dark_prismarine'),
        (182, 'Purpur Pillar', ('purpur_pillar_top', 'purpur_pillar',
                'purpur_pillar_top')),
        (315, 'Bamboo Mosaic', 'bamboo_mosaic'),
        (316, 'Chiseled Tuff Bricks', ('chiseled_tuff_bricks_top',
                'chiseled_tuff_bricks', 'chiseled_tuff_bricks_top')),
        (317, 'Cracked Deepslate Tiles', 'cracked_deepslate_tiles'),
        (318, 'Cracked Polished Blackstone Bricks',
                'cracked_polished_blackstone_bricks'),
        (319, 'Resin Bricks', 'resin_bricks'),
        (320, 'Chiseled Resin Bricks', 'chiseled_resin_bricks'),
        (321, 'Smooth Quartz', 'quartz_block_bottom'),
        (322, 'Target', ('target_top', 'target_side', 'target_top')),
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
        (183, 'Barrel', ('barrel_top', 'barrel_side', 'barrel_bottom')),
        (184, 'Blast Furnace', ('blast_furnace_top', 'blast_furnace_side',
                'blast_furnace_top', 'blast_furnace_front')),
        (185, 'Smoker', ('smoker_top', 'smoker_side', 'smoker_bottom', 'smoker_front')),
        (186, 'Dispenser', ('furnace_top', 'furnace_side', 'furnace_top',
                'dispenser_front')),
        (187, 'Cartography Table', ('cartography_table_top', 'cartography_table_side3',
                'dark_oak_planks', 'cartography_table_side3')),
        (188, 'Fletching Table', ('fletching_table_top', 'fletching_table_front',
                'birch_planks', 'fletching_table_front')),
        (189, 'Smithing Table', ('smithing_table_top', 'smithing_table_front',
                'smithing_table_bottom', 'smithing_table_front')),
        (190, 'Loom', ('loom_top', 'loom_side', 'loom_bottom', 'loom_front')),
        (191, 'Lodestone', ('lodestone_top', 'lodestone_side', 'lodestone_top')),
        (192, 'Redstone Lamp', 'redstone_lamp'),
        (193, 'Ochre Froglight', ('ochre_froglight_top', 'ochre_froglight_side',
                'ochre_froglight_top')),
        (194, 'Verdant Froglight', ('verdant_froglight_top', 'verdant_froglight_side',
                'verdant_froglight_top')),
        (195, 'Pearlescent Froglight', ('pearlescent_froglight_top',
                'pearlescent_froglight_side', 'pearlescent_froglight_top')),
        (196, 'Shroomlight', 'shroomlight'),
        (197, 'Brown Mushroom Block', 'brown_mushroom_block'),
        (198, 'Red Mushroom Block', 'red_mushroom_block'),
        (323, 'Dropper', ('furnace_top', 'furnace_side', 'furnace_top',
                'dropper_front')),
        (324, 'Bee Nest', ('bee_nest_top', 'bee_nest_side', 'bee_nest_bottom',
                'bee_nest_front')),
        (325, 'Beehive', ('beehive_end', 'beehive_side', 'beehive_end',
                'beehive_front')),
        (326, 'Creaking Heart', ('creaking_heart_top', 'creaking_heart',
                'creaking_heart_top')),
        (327, 'Respawn Anchor', ('respawn_anchor_top_off', 'respawn_anchor_side0',
                'respawn_anchor_bottom')),
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
    ('Beton', [
        (199, 'White Concrete', 'white_concrete'),
        (200, 'Light Gray Concrete', 'light_gray_concrete'),
        (201, 'Gray Concrete', 'gray_concrete'),
        (202, 'Black Concrete', 'black_concrete'),
        (203, 'Brown Concrete', 'brown_concrete'),
        (204, 'Red Concrete', 'red_concrete'),
        (205, 'Orange Concrete', 'orange_concrete'),
        (206, 'Yellow Concrete', 'yellow_concrete'),
        (207, 'Lime Concrete', 'lime_concrete'),
        (208, 'Green Concrete', 'green_concrete'),
        (209, 'Cyan Concrete', 'cyan_concrete'),
        (210, 'Light Blue Concrete', 'light_blue_concrete'),
        (211, 'Blue Concrete', 'blue_concrete'),
        (212, 'Purple Concrete', 'purple_concrete'),
        (213, 'Magenta Concrete', 'magenta_concrete'),
        (214, 'Pink Concrete', 'pink_concrete'),
        (328, 'White Concrete Powder', 'white_concrete_powder'),
        (329, 'Light Gray Concrete Powder', 'light_gray_concrete_powder'),
        (330, 'Gray Concrete Powder', 'gray_concrete_powder'),
        (331, 'Black Concrete Powder', 'black_concrete_powder'),
        (332, 'Brown Concrete Powder', 'brown_concrete_powder'),
        (333, 'Red Concrete Powder', 'red_concrete_powder'),
        (334, 'Orange Concrete Powder', 'orange_concrete_powder'),
        (335, 'Yellow Concrete Powder', 'yellow_concrete_powder'),
        (336, 'Lime Concrete Powder', 'lime_concrete_powder'),
        (337, 'Green Concrete Powder', 'green_concrete_powder'),
        (338, 'Cyan Concrete Powder', 'cyan_concrete_powder'),
        (339, 'Light Blue Concrete Powder', 'light_blue_concrete_powder'),
        (340, 'Blue Concrete Powder', 'blue_concrete_powder'),
        (341, 'Purple Concrete Powder', 'purple_concrete_powder'),
        (342, 'Magenta Concrete Powder', 'magenta_concrete_powder'),
        (343, 'Pink Concrete Powder', 'pink_concrete_powder'),
    ]),
    ('Terrakota', [
        (215, 'White Terracotta', 'white_terracotta'),
        (216, 'Light Gray Terracotta', 'light_gray_terracotta'),
        (217, 'Gray Terracotta', 'gray_terracotta'),
        (218, 'Black Terracotta', 'black_terracotta'),
        (219, 'Brown Terracotta', 'brown_terracotta'),
        (220, 'Red Terracotta', 'red_terracotta'),
        (221, 'Orange Terracotta', 'orange_terracotta'),
        (222, 'Yellow Terracotta', 'yellow_terracotta'),
        (223, 'Lime Terracotta', 'lime_terracotta'),
        (224, 'Green Terracotta', 'green_terracotta'),
        (225, 'Cyan Terracotta', 'cyan_terracotta'),
        (226, 'Light Blue Terracotta', 'light_blue_terracotta'),
        (227, 'Blue Terracotta', 'blue_terracotta'),
        (228, 'Purple Terracotta', 'purple_terracotta'),
        (229, 'Magenta Terracotta', 'magenta_terracotta'),
        (230, 'Pink Terracotta', 'pink_terracotta'),
    ]),
    ('Bakır', [
        (231, 'Exposed Copper', 'exposed_copper'),
        (232, 'Weathered Copper', 'weathered_copper'),
        (233, 'Oxidized Copper', 'oxidized_copper'),
        (234, 'Cut Copper', 'cut_copper'),
        (235, 'Exposed Cut Copper', 'exposed_cut_copper'),
        (236, 'Weathered Cut Copper', 'weathered_cut_copper'),
        (237, 'Oxidized Cut Copper', 'oxidized_cut_copper'),
        (238, 'Chiseled Copper', 'chiseled_copper'),
        (239, 'Copper Bulb', 'copper_bulb'),
        (344, 'Exposed Chiseled Copper', 'exposed_chiseled_copper'),
        (345, 'Weathered Chiseled Copper', 'weathered_chiseled_copper'),
        (346, 'Oxidized Chiseled Copper', 'oxidized_chiseled_copper'),
        (347, 'Exposed Copper Bulb', 'exposed_copper_bulb'),
        (348, 'Weathered Copper Bulb', 'weathered_copper_bulb'),
        (349, 'Oxidized Copper Bulb', 'oxidized_copper_bulb'),
    ]),
    ('Nether', [
        (240, 'Blackstone', ('blackstone_top', 'blackstone', 'blackstone_top')),
        (241, 'Polished Blackstone', 'polished_blackstone'),
        (242, 'Polished Blackstone Bricks', 'polished_blackstone_bricks'),
        (243, 'Chiseled Polished Blackstone', 'chiseled_polished_blackstone'),
        (244, 'Gilded Blackstone', 'gilded_blackstone'),
        (245, 'Polished Basalt', ('polished_basalt_top', 'polished_basalt_side',
                'polished_basalt_top')),
        (246, 'Smooth Basalt', 'smooth_basalt'),
        (247, 'Crimson Stem', ('crimson_stem_top', 'crimson_stem', 'crimson_stem_top')),
        (248, 'Warped Stem', ('warped_stem_top', 'warped_stem', 'warped_stem_top')),
        (249, 'Crimson Nylium', ('crimson_nylium', 'crimson_nylium_side', 'netherrack')),
        (250, 'Warped Nylium', ('warped_nylium', 'warped_nylium_side', 'netherrack')),
        (251, 'Nether Wart Block', 'nether_wart_block'),
        (252, 'Warped Wart Block', 'warped_wart_block'),
        (253, 'Red Nether Bricks', 'red_nether_bricks'),
        (254, 'Ancient Debris', ('ancient_debris_top', 'ancient_debris_side',
                'ancient_debris_top')),
        (255, 'Crying Obsidian', 'crying_obsidian'),
        (350, 'Chiseled Nether Bricks', 'chiseled_nether_bricks'),
        (351, 'Cracked Nether Bricks', 'cracked_nether_bricks'),
        # Hyphae is the stem's bark on all six faces, as Wood is to Log.
        (352, 'Crimson Hyphae', 'crimson_stem'),
        (353, 'Warped Hyphae', 'warped_stem'),
        (354, 'Stripped Crimson Stem', ('stripped_crimson_stem_top',
                'stripped_crimson_stem', 'stripped_crimson_stem_top')),
        (355, 'Stripped Warped Stem', ('stripped_warped_stem_top',
                'stripped_warped_stem', 'stripped_warped_stem_top')),
        (356, 'Stripped Crimson Hyphae', 'stripped_crimson_stem'),
        (357, 'Stripped Warped Hyphae', 'stripped_warped_stem'),
    ]),
    # Every block in this group is see-through, and that is not a cosmetic
    # label: TRANSPARENT below is built from it, the mesher routes their quads
    # into a second buffer, render_chunks draws that buffer in a blended pass
    # after the opaque one, and build_atlas.py keeps their alpha instead of
    # flattening it. Adding a see-through block anywhere else in this table
    # would draw it as an opaque block with a muddy texture.
    ('Şeffaf', [
        (256, 'Glass', 'glass'),
        (257, 'Tinted Glass', 'tinted_glass'),
        (258, 'Ice', 'ice'),
        (259, 'White Stained Glass', 'white_stained_glass'),
        (260, 'Light Gray Stained Glass', 'light_gray_stained_glass'),
        (261, 'Gray Stained Glass', 'gray_stained_glass'),
        (262, 'Black Stained Glass', 'black_stained_glass'),
        (263, 'Brown Stained Glass', 'brown_stained_glass'),
        (264, 'Red Stained Glass', 'red_stained_glass'),
        (265, 'Orange Stained Glass', 'orange_stained_glass'),
        (266, 'Yellow Stained Glass', 'yellow_stained_glass'),
        (267, 'Lime Stained Glass', 'lime_stained_glass'),
        (268, 'Green Stained Glass', 'green_stained_glass'),
        (269, 'Cyan Stained Glass', 'cyan_stained_glass'),
        (270, 'Light Blue Stained Glass', 'light_blue_stained_glass'),
        (271, 'Blue Stained Glass', 'blue_stained_glass'),
        (272, 'Purple Stained Glass', 'purple_stained_glass'),
        (273, 'Magenta Stained Glass', 'magenta_stained_glass'),
        (274, 'Pink Stained Glass', 'pink_stained_glass'),
        (275, 'Copper Grate', 'copper_grate'),
        (276, 'Exposed Copper Grate', 'exposed_copper_grate'),
        (277, 'Weathered Copper Grate', 'weathered_copper_grate'),
        (278, 'Oxidized Copper Grate', 'oxidized_copper_grate'),
        (358, 'Spawner', 'spawner'),
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

# The blocks you can see through, and the one place that says so.
TRANSPARENT = frozenset(bid for name, ids in GROUPS if name == 'Şeffaf' for bid in ids)
assert TRANSPARENT, 'the Şeffaf group is what defines the transparent blocks'

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

# Can this block hide the face of the block behind it? The mesher asks that of
# every neighbor, so it is a table lookup rather than a chain of comparisons.
# AIR and WATER never could; the see-through blocks cannot either, which is what
# keeps the terrain behind a glass wall meshed.
#
# It is a small enum rather than a flag, and every caller but one only wants the
# flag (`!= 0`):
#
#   0  see-through — AIR, WATER, the Şeffaf group
#   1  solid terrain
#   2  foliage — hides the face behind it, but is *not* cover
#
# The one caller that reads the 2 is `fast_builder.column_seal_limit`, and the
# distinction is load-bearing there. The far LOD fills in air that has
# SEAL_COVER solid blocks stacked over it, on the grounds that it must be a
# cave; a leaf column is not a cave roof, and a mega spruce's crown is thirteen
# rows deep, so counting foliage as cover grew a stone plinth under every
# distant giant — 42 visible blocks of it, measured. Nothing else needs a second
# table for that, and a second table would have to be threaded through
# build_chunk_mesh_fast and its seventeen call sites.
#
# Passed into build_chunk_mesh_fast as an argument for the same reason
# FACE_LAYER is — @njit(cache=True) freezes globals into the cached artifact.
FOLIAGE = frozenset(bid for name, ids in GROUPS if name == 'Yaprak' for bid in ids)
assert FOLIAGE, 'the Yaprak group is what defines the foliage blocks'

OPAQUE = np.ones(max(BLOCK_NAMES) + 1, dtype=np.uint8)
OPAQUE[0] = 0                       # AIR
OPAQUE[8] = 0                       # WATER — never meshed, see fast_builder
for _bid in TRANSPARENT:
    OPAQUE[_bid] = 0
for _bid in FOLIAGE:
    if OPAQUE[_bid]:                # a see-through leaf would stay 0
        OPAQUE[_bid] = 2

# Water has no row in the table (it is never meshed — see fast_builder), but the
# id is needed outside terrain generation: the player walks through it.
WATER = 8

HOTBAR_DEFAULT = [1, 2, 3, 4, 9, 10, 44, 7, 6]

# IDs 1-10 are written into every generated chunk by terrain_generator and into
# every cached one. Renumbering them repaints the whole world, so fail loudly at
# import instead.
_LEGACY = {1: 'Grass Block', 2: 'Dirt', 3: 'Stone', 4: 'Sand', 5: 'Snow',
           6: 'Oak Leaves', 7: 'Oak Log', 9: 'Stone Bricks', 10: 'Bricks'}
assert all(BLOCK_NAMES.get(k) == v for k, v in _LEGACY.items()), \
    'legacy block IDs 1-10 changed meaning; terrain and saved chunks depend on them'
assert 8 not in BLOCK_NAMES, 'ID 8 is WATER — never meshed, keep it out of the table'

# ModernChunk.blocks was uint8 until the see-through blocks needed IDs and 255
# was already taken. Widening it to uint16 doubles the largest allocation in the
# game — 64 KB -> 128 KB per chunk — and that is the entire price; measured, the
# mesher does not notice (0.561 -> 0.559 ms/chunk at lod 0, 0.482 -> 0.483 for
# terrain generation), because the part of the array a chunk actually touches is
# the ~40 populated levels, a few KB either way. Memory is ~7 MB more at render
# distance 6 and ~28 MB at 12, and P2-5 already stopped the cache from keeping
# untouched chunks. If block IDs ever pass 65535, the answer is a palette per
# chunk (referans.md §5.2, PalettedContainer), not uint32.
assert max(BLOCK_NAMES) < np.iinfo(BLOCK_DTYPE).max, \
    f'block IDs must fit {np.dtype(BLOCK_DTYPE).name} — ModernChunk.blocks is that type'
