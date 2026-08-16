"""Self-check for what the crosshair targets. Run: python test_interaction.py

The DDA raycast decides two things the player reads as one: which block gets
the black outline, and which cell a placed block goes into. Both fail as *looks
wrong* rather than as an exception — water that outlines like a solid, or a
block that stacks on the sea surface instead of sinking into it — and neither
is reachable from the other tests, which never open a window.

Nothing here needs GL. `raycast`, `add_block` and `remove_block` only touch
`camera.position`, `camera.front`, `block_reach` and the two block accessors,
so they are borrowed onto a stub world that supplies exactly those.
"""

import main
from main import REPLACEABLE, MinecraftModernGL
from world.blocks import FACING, LOWER, UPPER
from world.modern_chunk import AIR, GRASS, STONE, WATER

REACH = 8.0

OAK_DOOR = min(bid for bid in FACING if bid in UPPER)


class Vec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


class Cam:
    def __init__(self, eye, look):
        self.position = Vec(*eye)
        self.front = Vec(*look)
        self.yaw = 0.0      # looking down +X, so a door placed by yaw faces -X

    def intersects_block(self, x, y, z):
        return False        # never standing in the way, so placement is testable


class Player:
    """Just enough of MinecraftModernGL for the methods under test."""

    raycast = MinecraftModernGL.raycast
    add_block = MinecraftModernGL.add_block
    remove_block = MinecraftModernGL.remove_block
    orient = MinecraftModernGL.orient

    def __init__(self, world, eye, look):
        self.world = dict(world)
        self.camera = Cam(eye, look)
        self.block_reach = REACH
        self.selected_block_type = GRASS

    def get_block_at(self, x, y, z):
        return self.world.get((x, y, z), AIR)

    def set_block_at(self, x, y, z, block_type):
        self.world[(x, y, z)] = block_type


def sea(bed=19, surface=28):
    """One column of water over a stone bed, at x=z=0.

    The eye sits at y=26.5 throughout, which is inside the water and 6.5 blocks
    above the bed — comfortably inside `block_reach`, so a miss means the ray
    stopped, not that it ran out.
    """
    world = {(0, bed, 0): STONE}
    for y in range(bed + 1, surface + 1):
        world[(0, y, 0)] = WATER
    return world


def check_the_ray_goes_through_water():
    """Looking straight down at the sea, the outline lands on the bed."""
    hit = Player(sea(), eye=(0.5, 26.5, 0.5), look=(0, -1, 0)).raycast()
    assert hit['hit'], 'the ray stopped in the water and hit nothing'
    assert hit['block_type'] != WATER, 'the outline would be drawn around water'
    assert hit['block_pos'] == (0, 19, 0), f"targeted {hit['block_pos']}, not the bed"
    assert hit['prev_pos'] == (0, 20, 0), \
        f"the placement cell is {hit['prev_pos']}, not the water above the bed"
    print(f"through water: outline on {hit['block_pos']}, "
          f"placement into {hit['prev_pos']}")


def check_a_block_replaces_the_water():
    player = Player(sea(), eye=(0.5, 26.5, 0.5), look=(0, -1, 0))
    assert player.get_block_at(0, 20, 0) == WATER, 'the fixture is wrong'
    player.add_block()
    assert player.get_block_at(0, 20, 0) == GRASS, \
        'placing into water was refused — the block never landed'
    assert player.get_block_at(0, 19, 0) == STONE, 'the sea bed was overwritten'


def check_nothing_but_water_is_no_target():
    """Deep water with the bed out of reach: no hit, so no outline at all."""
    player = Player(sea(bed=-64), eye=(0.5, 26.5, 0.5), look=(0, -1, 0))
    assert not player.raycast()['hit'], 'water alone produced a target'

    before = dict(player.world)
    player.remove_block()
    assert player.world == before, 'water was mined'


def check_solids_still_stop_the_ray():
    """The control: without it, "it works" could just mean nothing is solid."""
    world = sea()
    world[(0, 25, 0)] = STONE                     # a ledge inside the water
    player = Player(world, eye=(0.5, 26.5, 0.5), look=(0, -1, 0))
    hit = player.raycast()
    assert hit['block_pos'] == (0, 25, 0), f'the ledge was missed ({hit})'

    player.add_block()
    assert player.get_block_at(0, 26, 0) == GRASS, 'nothing was placed on the ledge'

    player.world[(0, 26, 0)] = STONE              # now the cell is occupied
    player.add_block()
    assert player.get_block_at(0, 26, 0) == STONE, 'a placed block overwrote terrain'


def check_the_two_readings_agree():
    """The outline and the placement have to come off the same tuple.

    A ray that passes through a block the placement code then refuses to build
    in leaves the player aiming at terrain they cannot touch.
    """
    assert set(REPLACEABLE) == {AIR, WATER}, REPLACEABLE
    with open(main.__file__, encoding='utf-8') as handle:
        assert handle.read().count('REPLACEABLE') == 4, \
            'one of the three tests was spelled out by hand again'
    print(f'outline and placement share one rule: {REPLACEABLE}')


def check_a_door_is_two_blocks():
    """Placing one puts both halves down; breaking either takes both away.

    A door is two block ids because a cell holds one id and nothing else, so
    every half-measure here shows up in the world as a doorframe with no door
    in it or a top half hanging in the air.
    """
    world = {(0, 25, 0): STONE}                    # a ledge to place against
    player = Player(world, eye=(0.5, 26.5, 0.5), look=(0, -1, 0))
    player.selected_block_type = OAK_DOOR
    player.add_block()

    lower = player.get_block_at(0, 26, 0)
    upper = player.get_block_at(0, 27, 0)
    assert lower in FACING[OAK_DOOR], f'no door went down, got {lower}'
    assert UPPER[lower] == upper, f'the upper half is {upper}, not {UPPER[lower]}'
    assert LOWER[upper] == lower

    # Breaking either half takes the other with it, so both directions of the
    # UPPER/LOWER wiring are exercised. From inside the lower cell the ray hits
    # the lower half; from above it hits the upper one first.
    for eye in ((0.5, 26.5, 0.5), (0.5, 30.5, 0.5)):
        broken = Player(player.world, eye=eye, look=(0, -1, 0))
        broken.remove_block()
        assert broken.get_block_at(0, 27, 0) == AIR, f'{eye}: the upper half survived'
        assert broken.get_block_at(0, 26, 0) == AIR, f'{eye}: the lower half was left'
        assert broken.get_block_at(0, 25, 0) == STONE, f'{eye}: the ledge went too'

    # No headroom, no door at all — better than half of one.
    boxed = Player({(0, 25, 0): STONE, (0, 27, 0): STONE},
                   eye=(0.5, 26.5, 0.5), look=(0, -1, 0))
    boxed.selected_block_type = OAK_DOOR
    boxed.add_block()
    assert boxed.get_block_at(0, 26, 0) == AIR, \
        'a door went in with nowhere for its upper half'
    print(f'door: {OAK_DOOR} -> ({lower}, {upper}), both halves placed and broken')


def run():
    check_the_ray_goes_through_water()
    check_a_block_replaces_the_water()
    check_nothing_but_water_is_no_target()
    check_solids_still_stop_the_ray()
    check_a_door_is_two_blocks()
    check_the_two_readings_agree()
    print('\nok')


if __name__ == '__main__':
    run()
