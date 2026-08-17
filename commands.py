"""The in-game command console — `/tp`, `/locate`, `/set`, and the registry
they hang off.

The console opens with **T** for an empty line and **/** for a line with the
slash already in it, which is the real game's split. Three files share the
work and none of them knows the other two's job:

* `main.py` owns the console's *state* — what is typed, what has been printed,
  the history — and the keyboard while it is open, exactly as it does for the
  creative picker.
* `engine/hud.py` draws it, through the same `_bake_column` the F3 screen uses.
* this file is the commands themselves, and nothing here touches pygame or
  OpenGL — which is why `test_commands.py` can run all of it against a stub
  game with no window open.

**Adding a command is one decorated function.** It takes the game and the
already-split argument list, and returns nothing, a line, or a list of lines;
a line is a bare string, or `(text, colour)` when it wants its own colour::

    @command('seed', 'Print the world seed.')
    def _seed(game, args):
        return f'Seed: {WORLD_SEED}'

Anything wrong with the arguments is a `CommandError`, which `dispatch` prints
in red along with the command's own usage line — so no command formats its own
error, and none of them can take the game down with it. `COMMANDS.md` is the
long-form documentation, and `/help` is generated from these same rows, so the
two cannot drift apart.
"""

import math
from collections import namedtuple

from world.modern_chunk import CHUNK_HEIGHT
from world.terrain_generator import (BIOME_NAMES, SPAWN_CLEARANCE, column_biome,
                                     ring_columns, surface_height)

# Colours the console prints in. Grey for the echo of what was typed, so a
# scrollback reads as a conversation rather than as a wall of answers.
OK = (232, 232, 232)
ERROR = (255, 120, 120)
ECHO = (168, 168, 176)

# The render-distance limits. They live here rather than in main.py's +/- keys
# because `/set renderdistance` and those keys have to agree about them.
RENDER_MIN, RENDER_MAX = 2, 96

# Eye height above the ground a teleport lands on is `terrain_generator`'s own
# SPAWN_CLEARANCE: a teleport and a spawn are the same act, and it is the same
# *terrain* height both of them read — a village floor or a boulder can stand a
# block or two above it, so both want the same short drop rather than a flush
# landing.

# How `/locate` sweeps. Rings outward from the player, ~4.5 us a sample, so an
# ordinary biome costs a few thousand samples and a few milliseconds; the worst
# case — asking for something that is not out there — is the full ~190 000 and
# about a second.
#
# The step is what decides whether a *thin* biome can be found at all: a river
# is about ten blocks across and a beach not much more, so sampling every 24
# blocks (which is what `find_spawn` does, and it only wants dry land) steps
# straight over them.
LOCATE_STEP = 16
LOCATE_REACH = 4000


class CommandError(Exception):
    """Bad arguments, unknown name, nothing found — the console prints these
    in red and carries on. Raising one is how a command reports a problem."""


Command = namedtuple('Command', 'name run usage help aliases')

COMMANDS = {}       # name -> Command, in the order they were registered
_ALIASES = {}       # alias -> name


def command(usage, help, aliases=()):
    """Register a command. The first word of *usage* is its name."""
    def register(fn):
        name = usage.split()[0]
        COMMANDS[name] = Command(name, fn, usage, help, tuple(aliases))
        for alias in aliases:
            _ALIASES[alias] = name
        return fn
    return register


def lookup(name):
    """The Command for a name or one of its aliases, or None."""
    name = name.lower().lstrip('/')
    return COMMANDS.get(_ALIASES.get(name, name))


def dispatch(game, text):
    """Run one console line. Returns `[(text, colour)]` for the console to show.

    Every failure path ends here rather than at a call site: a command that
    raises anything at all prints as an error instead of unwinding into the
    game loop, because the world only exists in memory and losing it to a typo
    in an argument would be losing the player's session.
    """
    text = text.strip()
    if not text:
        return []

    out = [(text, ECHO)]
    words = text.lstrip('/').split()
    if not words:
        return out + [('Type /help for the list of commands.', ERROR)]

    cmd = lookup(words[0])
    if cmd is None:
        return out + [(f'Unknown command: {words[0]}   (try /help)', ERROR)]

    try:
        # _lines is inside the guard too: a command that returns the wrong sort
        # of thing is the same slip as one that raises, and has to print the
        # same way rather than unwind from a line the try no longer covers.
        return out + _lines(cmd.run(game, words[1:]))
    except CommandError as exc:
        return out + [(str(exc), ERROR), (f'Usage: /{cmd.usage}', ERROR)]
    except Exception as exc:                       # noqa: BLE001 — see docstring
        return out + [(f'/{cmd.name} failed: {type(exc).__name__}: {exc}', ERROR)]


def _lines(result):
    """A command's return value as `(text, colour)` rows."""
    if result is None:
        return []
    if isinstance(result, str):
        return [(result, OK)]
    return [(row, OK) if isinstance(row, str) else tuple(row) for row in result]


# ---------------------------------------------------------------------------
# Teleporting
# ---------------------------------------------------------------------------

def teleport(game, x, z, y=None):
    """Put the player down on a block column, having first made sure it exists.

    *x* and *z* are **block** coordinates — the ones the F3 screen shows — and
    the player lands at the centre of that column. A body 0.4 wide dropped on a
    block *edge* straddles two columns, and if only one of them is solid it
    wedges against a wall it is standing half inside.

    *y* is the eye height, or None for the surface. Chunks load asynchronously
    and `get_block_at` reads air wherever none has arrived yet, so without
    `ensure_chunk` a player set down in unexplored terrain falls through the
    world until the queue catches up with them.
    """
    bx, bz = int(math.floor(x)), int(math.floor(z))
    if y is None:
        y = surface_height(bx, bz) + SPAWN_CLEARANCE

    # Bedrock is block 0, so its top face is y = 1 and standing on it puts the
    # eye a player's height above that. Clamping any lower only *looks* like a
    # floor: below bedrock every cell reads as air and the fall never ends.
    y = min(max(y, 1.0 + game.camera.player_height), float(CHUNK_HEIGHT))

    # Written in place: the chunk manager keeps a reference to this vector as
    # `player_position`, so rebinding the attribute would leave it pointing at
    # where the player used to be — and ensure_chunk, below, reads exactly that
    # to decide the landing chunk's mesh priority and detail level. Moving the
    # player first is what stops the chunk they are standing in from being
    # queued as the furthest and least detailed thing in the world.
    pos = game.camera.position
    pos.x, pos.y, pos.z = bx + 0.5, y, bz + 0.5
    vel = game.camera.velocity
    vel.x = vel.y = vel.z = 0.0
    game.camera.on_ground = False

    cx, cz = game.chunk_manager.world_to_chunk_coords(bx, bz)
    game.chunk_manager.ensure_chunk(cx, cz)

    return (f'Teleported to {bx} {y:.0f} {bz}'
            f'   ({BIOME_NAMES[column_biome(bx, bz)]})')


def _coord(token, base):
    """One `/tp` coordinate. `~` is where the player already is, `~5` is five
    blocks on from there — the real game's own relative syntax.

    `nan` and `inf` parse as floats and have to be refused here rather than
    downstream: NaN survives `min`/`max` untouched (every comparison against it
    is False), so a clamp cannot catch it, and a NaN in `camera.position` is a
    ValueError out of the collision sweep on the *next* frame — a crash with
    nothing left on screen to say which typed line caused it.
    """
    try:
        value = (base + (float(token[1:]) if len(token) > 1 else 0.0)
                 if token.startswith('~') else float(token))
    except ValueError:
        raise CommandError(f'"{token}" is not a coordinate.')
    if not math.isfinite(value):
        raise CommandError(f'"{token}" is not a place.')
    return value


@command('tp <x> [y] <z>',
         'Teleport. Leave y out to land on the surface at that column. '
         'A coordinate may be ~ for where you are now, or ~5 for five blocks '
         'on from it. You arrive at the centre of the column.',
         aliases=('teleport',))
def _tp(game, args):
    pos = game.camera.position
    if len(args) == 3:
        return teleport(game, _coord(args[0], pos.x), _coord(args[2], pos.z),
                        _coord(args[1], pos.y))
    if len(args) == 2:
        return teleport(game, _coord(args[0], pos.x), _coord(args[1], pos.z))
    raise CommandError('Give x y z, or x z to land on the surface.')


# ---------------------------------------------------------------------------
# Biomes
# ---------------------------------------------------------------------------

def _norm(name):
    """A biome name reduced to what a player is likely to type: `Deep Ocean`,
    `deep_ocean` and `deepocean` all come out the same."""
    return ''.join(ch for ch in name.lower() if ch.isalnum())


def biome_id(text):
    """The biome a typed name means.

    Exact match first, so `Ocean` is the ocean rather than an ambiguity between
    the four that have the word in them; then a substring, so `birch` is enough
    for the birch forest. Anything matching two or more says which.
    """
    key = _norm(text)
    if not key:
        raise CommandError('Give a biome name.   (try /locate list)')

    for i, name in enumerate(BIOME_NAMES):
        if _norm(name) == key:
            return i

    hits = [i for i, name in enumerate(BIOME_NAMES) if key in _norm(name)]
    if not hits:
        raise CommandError(f'No biome called "{text}".   (try /locate list)')
    if len(hits) > 1:
        raise CommandError('Which one? ' + ', '.join(BIOME_NAMES[i] for i in hits))
    return hits[0]


def nearest_biome(biome, origin_x, origin_z):
    """The nearest column of *biome*, or None if there is none within reach.

    ponytail: it runs on the main thread, so a miss stalls the game for about a
    second. Bounded reach is what keeps that a hitch instead of a hang; move it
    onto a worker if searches for absent biomes ever become the normal case.
    """
    for x, z in ring_columns(origin_x, origin_z, LOCATE_STEP, LOCATE_REACH):
        if column_biome(x, z) == biome:
            return x, z
    return None


@command('locate <biome>',
         'Teleport to the nearest column of a biome. Partial names work '
         f'(birch, jagged); /locate list names all {len(BIOME_NAMES)}.',
         aliases=('locatebiome',))
def _locate(game, args):
    if not args:
        raise CommandError('Give a biome name.   (try /locate list)')
    if len(args) == 1 and args[0].lower() == 'list':
        return [f'The {len(BIOME_NAMES)} biomes:'] + [
            ', '.join(BIOME_NAMES[i:i + 4]) for i in range(0, len(BIOME_NAMES), 4)]

    want = biome_id(' '.join(args))
    pos = game.camera.position
    found = nearest_biome(want, int(math.floor(pos.x)), int(math.floor(pos.z)))
    if found is None:
        raise CommandError(f'No {BIOME_NAMES[want]} within '
                           f'{LOCATE_REACH} blocks of here.')

    x, z = found
    distance = math.hypot(x - pos.x, z - pos.z)
    return [teleport(game, x, z),
            f'{BIOME_NAMES[want]} was {distance:.0f} blocks away.']


@command('biome', 'Name the biome the player is standing in.')
def _biome(game, args):
    pos = game.camera.position
    x, z = int(math.floor(pos.x)), int(math.floor(pos.z))
    return f'{BIOME_NAMES[column_biome(x, z)]}   at {x} {z}'


# ---------------------------------------------------------------------------
# Settings — everything a keyboard shortcut changes
# ---------------------------------------------------------------------------

Setting = namedtuple('Setting', 'name get set parse help')

SETTINGS = {}


def _setting(name, get, set_, parse, help):
    SETTINGS[name] = Setting(name, get, set_, parse, help)


def _flag(name, get, toggle, help):
    """A yes/no setting whose setter *is* the toggle its key press already
    calls, so `/set fly on` and TAB run the same code and print the same line.
    """
    def write(game, value):
        if get(game) != value:
            toggle(game)
    _setting(name, get, write, _parse_bool, help)


_TRUE = {'on', 'true', 'yes', 'enable', 'enabled', '1'}
_FALSE = {'off', 'false', 'no', 'disable', 'disabled', '0'}


def _parse_bool(word, current):
    word = word.lower()
    if word in _TRUE:
        return True
    if word in _FALSE:
        return False
    if word == 'toggle':
        return not current
    raise CommandError(f'"{word}" is not on, off or toggle.')


def _parse_render_distance(word, current):
    try:
        value = int(word)
    except ValueError:
        raise CommandError(f'"{word}" is not a whole number.')
    if not RENDER_MIN <= value <= RENDER_MAX:
        raise CommandError(f'Give a number between {RENDER_MIN} and {RENDER_MAX}.')
    return value


def set_render_distance(game, value):
    """The one setter — `/set renderdistance` and the +/- keys both come here.

    `game.render_distance` is what the F3 screen reads and the chunk manager's
    copy is what actually loads chunks; they were only ever kept in step by the
    key handler doing both, which is exactly the sort of thing a second caller
    quietly breaks.
    """
    game.render_distance = value
    game.chunk_manager.set_render_distance(value)


_setting('renderdistance',
         lambda game: game.render_distance, set_render_distance,
         _parse_render_distance,
         f'chunks loaded around you ({RENDER_MIN}-{RENDER_MAX}); keys + and -')

_flag('fly', lambda game: game.camera.flying,
      lambda game: game.camera.toggle_flying(), 'flight mode; key TAB')

_flag('culling', lambda game: game.chunk_manager.enable_frustum_culling,
      lambda game: game.chunk_manager.toggle_frustum_culling(),
      'frustum culling; key F')

_flag('wireframe', lambda game: game.renderer.wireframe_mode,
      lambda game: game.renderer.toggle_wireframe(), 'wireframe pass; key K')

_flag('debug', lambda game: game.show_debug,
      lambda game: setattr(game, 'show_debug', not game.show_debug),
      'the F3 screen; key F3')


def _show(value):
    if isinstance(value, bool):
        return 'on' if value else 'off'
    return str(value)


@command('set [setting] [value]',
         'Read or change a setting. With no arguments, lists them all with '
         'their current values; with a name only, reads that one. A yes/no '
         'setting takes on, off or toggle.',
         aliases=('config',))
def _set(game, args):
    if not args:
        return ['Settings (/set <name> <value>):'] + [
            f'  {s.name:<14} {_show(s.get(game)):<5} {s.help}'
            for s in SETTINGS.values()]

    setting = SETTINGS.get(args[0].lower())
    if setting is None:
        raise CommandError(f'No setting called "{args[0]}".   (try /set)')
    if len(args) == 1:
        return f'{setting.name} = {_show(setting.get(game))}   {setting.help}'

    setting.set(game, setting.parse(args[1], setting.get(game)))
    return f'{setting.name} = {_show(setting.get(game))}'


# ---------------------------------------------------------------------------
# Help — generated from the rows above, so it cannot drift from them
# ---------------------------------------------------------------------------

@command('help [command]', 'List the commands, or explain one of them.')
def _help(game, args):
    if args:
        cmd = lookup(args[0])
        if cmd is None:
            raise CommandError(f'No command called "{args[0]}".   (try /help)')
        lines = [f'/{cmd.usage}', f'  {cmd.help}']
        if cmd.aliases:
            lines.append('  also: ' + ', '.join('/' + a for a in cmd.aliases))
        return lines

    return ['Commands — /help <name> for detail, ESC closes the console:'] + [
        f'  /{cmd.usage}' for cmd in COMMANDS.values()]
