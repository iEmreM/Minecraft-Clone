"""Self-check for the command console. Run: python test_commands.py

Every one of these fails as *the game does the wrong thing quietly*: a teleport
that lands somewhere other than where it says, a `/set` that reports a value it
did not write, a biome search that answers with the wrong biome, a bad argument
that unwinds into the game loop instead of printing in red. None of it is
reachable from the other tests, which never type anything.

Nothing here needs GL or a window. `commands.py` touches pygame and OpenGL
nowhere at all — it reaches into the game object for a camera, a chunk manager
and a renderer and nothing else — so the whole file runs against the stub
below, which supplies exactly those.
"""

import math

import commands
from commands import COMMANDS, SETTINGS, dispatch
from world.terrain_generator import BIOME_NAMES, column_biome, surface_height


class Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __repr__(self):
        return f'({self.x:.1f}, {self.y:.1f}, {self.z:.1f})'


class Cam:
    player_height = 1.8      # Camera's own; the y clamp reads it

    def __init__(self, x, y, z):
        self.position = Vec(x, y, z)
        self.velocity = Vec(1.0, -9.0, 2.0)   # falling, so a teleport must stop it
        self.flying = False
        self.on_ground = True

    def toggle_flying(self):
        self.flying = not self.flying


class ChunkManager:
    """The three things a command asks of it, and a record of what it asked."""

    def __init__(self, camera):
        self.camera = camera
        self.render_distance = 6
        self.enable_frustum_culling = True
        self.ensured = []
        self.player_when_ensured = []

    def world_to_chunk_coords(self, x, z):
        return int(x // 16), int(z // 16)

    def ensure_chunk(self, cx, cz):
        # The real one reads player_position — the very vector the camera holds
        # — for the new chunk's mesh priority and detail level, so *when* it is
        # called relative to the move is part of what teleport has to get right.
        self.ensured.append((cx, cz))
        pos = self.camera.position
        self.player_when_ensured.append((pos.x, pos.z))
        return True

    def set_render_distance(self, value):
        self.render_distance = value

    def toggle_frustum_culling(self):
        self.enable_frustum_culling = not self.enable_frustum_culling


class Renderer:
    def __init__(self):
        self.wireframe_mode = False

    def toggle_wireframe(self):
        self.wireframe_mode = not self.wireframe_mode


class Game:
    """Just enough of MinecraftModernGL for the commands under test."""

    def __init__(self, x=0.0, y=80.0, z=0.0):
        self.camera = Cam(x, y, z)
        self.chunk_manager = ChunkManager(self.camera)
        self.renderer = Renderer()
        self.render_distance = 6
        self.show_debug = False


def say(game, line):
    """Run a console line and return what it printed, echo dropped."""
    return dispatch(game, line)[1:]


def text(rows):
    return ' | '.join(row[0] for row in rows)


def failed(rows):
    return any(color == commands.ERROR for _, color in rows)


# ---------------------------------------------------------------------------

def check_the_registry_is_complete():
    """Every command is reachable, documented, and listed by /help.

    /help is generated from the same rows the dispatcher looks names up in, so
    the failure this catches is a command registered under a name nobody can
    type — an alias colliding with a real command, or a usage string whose
    first word is not the name.
    """
    listing = text(say(Game(), '/help'))
    for name, cmd in COMMANDS.items():
        assert cmd.usage.split()[0] == name, f'{name}: usage says {cmd.usage!r}'
        assert cmd.help.strip(), f'{name} has no help'
        assert commands.lookup(name) is cmd, f'{name} does not look itself up'
        assert commands.lookup('/' + name.upper()) is cmd, f'{name}: not case-free'
        assert f'/{cmd.usage}' in listing, f'{name} is missing from /help'
        for alias in cmd.aliases:
            assert commands.lookup(alias) is cmd, f'{alias} does not reach {name}'
            assert alias not in COMMANDS, f'{alias} shadows a real command'
        assert not failed(say(Game(), f'/help {name}')), f'/help {name} errored'

    print(f'registry: {len(COMMANDS)} commands, '
          f"{', '.join('/' + n for n in COMMANDS)}")


def check_bad_input_prints_instead_of_raising():
    """Nothing typed at the console may unwind into the game loop.

    The world only exists in memory, so a crash out of the frame loop is the
    player's session. Every one of these is a line a player will type sooner or
    later, and all of them have to come back as red text.
    """
    # nan and inf parse as floats, and NaN survives min/max untouched (every
    # comparison against it is False), so a clamp cannot catch one. A NaN that
    # reaches camera.position is a ValueError out of the collision sweep on the
    # *next* frame — a crash with nothing left on screen to say which line did
    # it — so these have to be refused at the parse, not clamped later.
    for line in ('/nosuchcommand', '/', '//', '/tp', '/tp here there',
                 '/tp 1 2 3 4', '/tp ~x 5 ~', '/tp 0 nan 0', '/tp ~ ~nan ~',
                 '/tp inf 5 0', '/tp 0 5 -inf', '/tp nan nan',
                 '/locate', '/locate nowhere',
                 '/locate snowy', '/set nosuchsetting', '/set fly maybe',
                 '/set renderdistance 900', '/set renderdistance six',
                 '/help nosuchcommand'):
        game = Game()
        rows = say(game, line)
        assert failed(rows), f'{line!r} was accepted: {text(rows)}'
        assert game.camera.position.x == 0.0 and game.camera.position.y == 80.0, \
            f'{line!r} moved the player before failing'
        pos = game.camera.position
        assert all(math.isfinite(v) for v in (pos.x, pos.y, pos.z)), \
            f'{line!r} left a non-finite coordinate: {pos}'

    assert dispatch(Game(), '   ') == [], 'a blank line printed something'
    print(f'bad input: {len(COMMANDS)} commands, no exception escaped')


def check_teleport_lands_where_it_says():
    """The coordinates in the answer are the coordinates the player is at.

    Block coordinates in, block centre out — a body 0.4 wide dropped on a block
    *edge* straddles two columns and can wedge in a wall that is only in one of
    them. And the destination chunk has to be built before the player arrives,
    or `get_block_at` reads air and they fall through the world.
    """
    game = Game()
    rows = say(game, '/tp 100 70 -200')
    pos = game.camera.position
    assert (pos.x, pos.y, pos.z) == (100.5, 70.0, -199.5), f'landed at {pos}'
    assert '100 70 -200' in text(rows), f'said {text(rows)}'
    assert game.chunk_manager.ensured == [(6, -13)], \
        f'built {game.chunk_manager.ensured}, not the chunk it landed in'

    # The player has to be moved *first*: the chunk manager sizes the new
    # chunk's mesh priority and its detail level off where the player is when
    # the request goes in, so building it while they are still at the old spot
    # queues the ground under their feet as the furthest, coarsest thing loaded.
    assert game.chunk_manager.player_when_ensured == [(100.5, -199.5)], \
        f'built the chunk from {game.chunk_manager.player_when_ensured}'

    # Falling has to stop, or the player carries the old velocity into a new
    # world and is driven into the ground the moment it loads.
    vel = game.camera.velocity
    assert (vel.x, vel.y, vel.z) == (0.0, 0.0, 0.0), f'still moving at {vel}'
    assert not game.camera.on_ground, 'claims to be standing before it has landed'

    # A negative coordinate has to floor, not truncate: int(-0.5) is 0 and
    # would put the player a block east of where they asked.
    game = Game()
    say(game, '/tp -0.5 70 -0.5')
    assert (game.camera.position.x, game.camera.position.z) == (-0.5, -0.5), \
        f'landed at {game.camera.position}'
    print('teleport: 100 70 -200 -> (100.5, 70.0, -199.5), chunk (6, -13) built')


def check_relative_coordinates():
    """`~` is where you are, `~5` is five blocks on — the real game's syntax."""
    game = Game(30.5, 64.0, -12.5)
    say(game, '/tp ~ ~10 ~')
    pos = game.camera.position
    assert (pos.x, pos.y, pos.z) == (30.5, 74.0, -12.5), f'~ moved to {pos}'

    # A fractional offset still lands on a block centre — -12.5 + 2.5 is -10.0,
    # which is the *edge* of block -10, and the whole point of centring is that
    # the player never stands on one.
    say(game, '/tp ~-6 ~ ~2.5')
    pos = game.camera.position
    assert (pos.x, pos.y, pos.z) == (24.5, 74.0, -9.5), f'~n moved to {pos}'
    print('relative: ~ holds, ~-6 / ~2.5 step from where the player is')


def check_the_surface_form_clears_the_ground():
    """`/tp x z` puts the player above the terrain, not inside it.

    Height comes from the same lattice the chunk generator builds off, so the
    ground under the destination is the ground that will be there. The feet
    have to clear the topmost solid block; landing level with it is standing
    inside it.
    """
    for x, z in ((0, 0), (713, -486), (-2400, 1750), (57, 57)):
        game = Game()
        rows = say(game, f'/tp {x} {z}')
        feet = game.camera.position.y - Cam.player_height
        ground = surface_height(x, z)
        # `ground` is the index of the topmost solid block, so its top face is
        # at ground + 1 and that is where feet stand. `feet > ground` would
        # tolerate anything in between — which is inside the block.
        assert feet >= ground + 1, (f'({x}, {z}): feet at {feet:.1f}, standing '
                                    f'on the block at {ground} means {ground + 1}')
        assert feet < ground + 4, f'({x}, {z}): dropped from {feet - ground:.1f}'
        assert BIOME_NAMES[column_biome(x, z)] in text(rows), \
            f'({x}, {z}) reported the wrong biome: {text(rows)}'

    # And the clamp under it is a real floor: bedrock is block 0, so an eye any
    # lower than one player-height above its top face reads air all the way down.
    game = Game()
    say(game, '/tp 0 -50 0')
    # Written without the subtraction: 2.8 - 1.8 is 0.9999999999999998, and the
    # invariant is about where the eye is put, not about float arithmetic.
    assert game.camera.position.y >= 1.0 + Cam.player_height, \
        f'clamped to {game.camera.position.y}, feet below the bedrock'
    print(f'surface form: /tp 713 -486 -> {surface_height(713, -486)} + clearance')


def check_biome_names_resolve():
    """What a player types has to reach the biome they meant.

    Exact before substring, or `Ocean` is an ambiguity between the four names
    that contain it rather than the ocean. Ambiguity that is real is an error
    naming the candidates, not a silent pick of the first one.
    """
    for typed, want in (('Plains', 'Plains'), ('plains', 'Plains'),
                        ('Ocean', 'Ocean'), ('deep_ocean', 'Deep Ocean'),
                        ('deepocean', 'Deep Ocean'), ('birch', 'Birch Forest'),
                        ('forest', 'Forest'), ('jagged', 'Jagged Peaks'),
                        ('BADLANDS', 'Badlands')):
        got = BIOME_NAMES[commands.biome_id(typed)]
        assert got == want, f'{typed!r} resolved to {got}, not {want}'

    # Ambiguity that is real: no exact name and more than one containing it.
    for typed in ('snowy', 'ocea', 'peaks'):
        try:
            commands.biome_id(typed)
        except commands.CommandError as exc:
            assert 'Which one?' in str(exc), f'{typed!r} said: {exc}'
        else:
            raise AssertionError(f'{typed!r} matched several and picked one')

    listed = text(say(Game(), '/locate list'))
    for name in BIOME_NAMES:
        assert name in listed, f'/locate list is missing {name}'
    print(f'names: exact beats substring, {len(BIOME_NAMES)} listed')


def check_locate_finds_the_real_biome():
    """The column it teleports to really is that biome.

    The search is a ring sweep on a 16-block step, and the step is what decides
    whether a *thin* biome can be found at all — a river is about ten blocks
    across. So this asks for every one of the 26 rather than a sample: a
    threshold drifting in the terrain, or a coarser step, silently turns one of
    them into "not within 4000 blocks".
    """
    worst = (0, '')
    for biome, name in enumerate(BIOME_NAMES):
        game = Game(0.0, 80.0, 0.0)
        rows = say(game, f'/locate {name}')
        assert not failed(rows), f'{name}: {text(rows)}'

        x = int(math.floor(game.camera.position.x))
        z = int(math.floor(game.camera.position.z))
        assert column_biome(x, z) == biome, \
            f'{name}: landed on {BIOME_NAMES[column_biome(x, z)]} at {x} {z}'
        assert game.chunk_manager.ensured, f'{name}: no chunk built to land in'

        distance = math.hypot(x, z)
        worst = max(worst, (distance, name))
    print(f'locate: all {len(BIOME_NAMES)} found from the origin, '
          f'furthest {worst[1]} at {worst[0]:.0f} blocks')


def check_settings_round_trip():
    """A setting reads back what was written, through the game's own toggles.

    `/set fly on` goes through `Camera.toggle_flying`, the same call TAB makes,
    so the two can never end up doing different things to the same flag. What
    that hides if it breaks is a setter that writes somewhere the getter does
    not read — the value reported after the write would still look right.
    """
    for name, setting in SETTINGS.items():
        game = Game()
        start = setting.get(game)
        assert not failed(say(game, f'/set {name}')), f'/set {name} could not read'

        if isinstance(start, bool):
            for word, want in (('on', True), ('off', False), ('toggle', None),
                               ('toggle', None)):
                before = setting.get(game)
                rows = say(game, f'/set {name} {word}')
                expect = (not before) if want is None else want
                assert not failed(rows), f'/set {name} {word}: {text(rows)}'
                assert setting.get(game) is expect, \
                    f'/set {name} {word} left it {setting.get(game)}'
                assert _shown(rows) == ('on' if expect else 'off'), \
                    f'/set {name} {word} reported {text(rows)}'
        else:
            say(game, f'/set {name} {commands.RENDER_MAX}')
            assert setting.get(game) == commands.RENDER_MAX
            assert game.chunk_manager.render_distance == commands.RENDER_MAX, \
                'the chunk manager kept the old render distance'

    listing = text(say(Game(), '/set'))
    for name in SETTINGS:
        assert name in listing, f'/set does not list {name}'
    print(f"settings: {', '.join(SETTINGS)} -- all read back what was written")


def _shown(rows):
    """The value out of a `name = value` line."""
    return rows[-1][0].split('=')[-1].split()[0]


def check_the_render_distance_setter_is_shared():
    """The + / - keys and `/set renderdistance` write both copies of it.

    `game.render_distance` is what the F3 screen and the key handler read;
    `chunk_manager.render_distance` is what actually loads chunks. They were
    only ever kept in step by one call site doing both.
    """
    game = Game()
    commands.set_render_distance(game, 17)
    assert game.render_distance == 17, 'the game kept the old distance'
    assert game.chunk_manager.render_distance == 17, 'the manager did not'

    say(game, '/set renderdistance 4')
    assert (game.render_distance, game.chunk_manager.render_distance) == (4, 4), \
        f'{game.render_distance} vs {game.chunk_manager.render_distance}'
    print('render distance: one setter, both copies')


def check_the_current_biome():
    """`/biome` names the column the player is standing in, not the one next to
    it — floor, not truncate.

    Checking only the *name* would not catch truncation: neighbouring columns
    are the same biome essentially always, so the coordinates it prints are what
    has to be checked. `int(-40.5)` is -40 and `floor(-40.5)` is -41, and the
    line would read plausibly either way.
    """
    for x, z in ((0.5, 0.5), (-40.5, 912.5), (2000.5, -3000.5)):
        game = Game(x, 80.0, z)
        rows = say(game, '/biome')
        want = BIOME_NAMES[column_biome(int(math.floor(x)), int(math.floor(z)))]
        assert text(rows).startswith(want), f'({x}, {z}): {text(rows)}, want {want}'
        assert text(rows).endswith(f'{math.floor(x):.0f} {math.floor(z):.0f}'), \
            f'({x}, {z}): reads the column at {text(rows).split()[-2:]}'
    print(f'biome: (-40.5, 912.5) -> {text(say(Game(-40.5, 80.0, 912.5), "/biome"))}')


def run():
    check_the_registry_is_complete()
    check_bad_input_prints_instead_of_raising()
    check_teleport_lands_where_it_says()
    check_relative_coordinates()
    check_the_surface_form_clears_the_ground()
    check_biome_names_resolve()
    check_locate_finds_the_real_biome()
    check_settings_round_trip()
    check_the_render_distance_setter_is_shared()
    check_the_current_biome()
    print('\nok')


if __name__ == '__main__':
    run()
