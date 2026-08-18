# Commands

An in-game console, opened the way the real game opens its chat. It is the way
to reach things a key press cannot say: *which* coordinates, *which* biome,
*which* value.

```
T          open an empty line
/          open a line with the slash already in it
Enter      run it and close
ESC        close without running
↑ ↓        walk back through what has been typed before
Backspace  delete a character
```

`T` is the one that always works. `/` is a *key*, and on a layout where the
slash is not a bare key press (a Turkish Q keyboard, for one) that key press
never arrives — open with `T` and type the slash, which goes through text input
and is layout-independent. The numpad `/` opens it too.

The console takes the whole keyboard and the mouse while it is open, exactly as
the creative picker does — WASD spells a command instead of walking, clicks and
the wheel do nothing, and the mouse comes back in the state it was found in. What a command prints stays on
screen for **8 seconds after the console closes**, which is the only way to read
it: Enter runs the line and closes the console in the same keystroke.

The console and the F3 screen are drawn in a **monospace** font (Consolas, or
whatever `engine/hud.MONO` finds first), and both halves of that show: `/set`
prints a space-padded table, and F3 prints numbers that are rewritten ten times
a second. It is also the fix for the lowercase `i` — pygame's built-in
`freesansbold` fuses the dot into the stem at these sizes, so every `i` read as
an `l`. The console's font is the larger of the two: it prints a sentence you
read once and dismiss, where F3 is a permanent corner readout.

Errors print in red and never interrupt the game. A typo cannot crash it: every
failure — an unknown name, an unparseable number, a bug in a command — comes
back as a red line. The world only exists in memory, so losing the frame loop
would be losing the session.

The leading `/` is optional (`tp 0 0` works), names are case-insensitive, and
`/help` is generated from the same rows the dispatcher looks names up in, so it
cannot fall out of date with what is actually there.

---

## `/tp` — teleport

```
/tp <x> <y> <z>
/tp <x> <z>
```

Aliases: `/teleport`

Coordinates are **block** coordinates — the ones the F3 screen shows on its
`Block:` line — and the player lands at the **centre** of that column. A body
0.4 blocks wide dropped on a block *edge* straddles two columns, and wedges
against a wall that is only in one of them.

Leaving `y` out lands on the surface at that column, four blocks above the
terrain height, which is a drop of about one block onto it. That clearance is
`terrain_generator.SPAWN_CLEARANCE`, the same number the spawn point uses, for
the same reason: the height it reads is the *terrain*, and a village floor or a
boulder can stand a block or two above it.

A coordinate may be relative, the real game's own `~` syntax:

| written | means |
| :--- | :--- |
| `100` | absolute — block 100 |
| `~` | where the player is now |
| `~12` | twelve blocks on from there |
| `~-3.5` | three and a half blocks back |

```
/tp 1200 90 -400        an exact spot
/tp 1200 -400           the same column, standing on the ground
/tp ~ ~40 ~             forty blocks straight up
/tp ~-64 ~ ~            sixty-four blocks west
```

**The destination chunk is built before the player arrives.** Chunks load on
background threads, and `get_block_at` reads air wherever none has arrived yet —
so without that, a player set down in unexplored terrain falls through the world
until the loader catches up, which after a jump of a few thousand blocks is a
queue with every chunk around the destination already in it. `/tp` therefore
generates its landing chunk synchronously (about 0.2 ms) and lets the mesh come
the ordinary way, so the ground is solid on the frame the player lands and
visible a few frames later.

A `y` you give is used as given, clamped only at the two ends of the world —
never below one player-height above the bedrock, where every cell reads as air
and the fall would not stop. Teleporting into the middle of a mountain puts you
inside it, as it does in the real game; reopen the console with `T` and `/tp`
again, or `/set fly on`, and you are out.

## `/locate` — find a biome or a structure and go there

```
/locate <name>
/locate list
```

Aliases: `/locatebiome`

Searches outward from the player for the nearest one and teleports to it,
landing on the surface exactly as `/tp <x> <z>` does. It reports where it went
and how far that was.

```
/locate jungle              → Teleported to -145 71 285   (Jungle)
                              Jungle was 320 blocks away.
/locate village             → Teleported to -456 44 -136   (Plains)
                              Village was 476 blocks away.
/locate jagged              → Jagged Peaks
/locate deep_ocean          → Deep Ocean
/locate list                → all 26 biomes, and the structures
```

Names are matched leniently: case, spaces, underscores and hyphens are all
ignored, so `Deep Ocean`, `deep_ocean` and `deepocean` are the same thing. A
**partial** name works when only one thing contains it — `birch`, `jagged`,
`badlands`. An exact name always wins over a partial one, so `ocean` is the
ocean rather than an ambiguity between the four names that contain the word.
Anything that genuinely matches several says which:

```
/locate snowy   → Which one? Snowy Beach, Snowy Taiga, Snowy Plains, Snowy Slopes
```

**Biomes and structures share one namespace and one matching rule**, which is
not tidiness. A future `Desert Pyramid` looked up in a table of its own first
would take `/locate desert` away from the biome of that name; matched together
the biome's exact name wins, and `pyramid` still reaches the structure.

The 26 biomes are Ocean, Deep Ocean, Frozen Ocean, Warm Ocean, Beach, Snowy
Beach, Stony Shore, River, Plains, Forest, Birch Forest, Dark Forest, Taiga,
Snowy Taiga, Snowy Plains, Savanna, Desert, Jungle, Swamp, Badlands, Windswept
Hills, Meadow, Grove, Snowy Slopes, Jagged Peaks and Stony Peaks. The structures
are **Village**.

**Cost, a biome.** The sweep rings outward on a 16-block step, out to 4000
blocks, and it shares that ring walk with the spawn search
(`terrain_generator.ring_columns`). The step is what decides whether a *thin*
biome can be found at all — a river is about ten blocks across and a beach not
much more.

A sample is about 4.5 µs. Measured from the origin, every one of the 26 biomes is
found: the common ones in 400–1500 samples and under 10 ms, the three furthest
(Warm Ocean, Badlands, Desert) in 10 000–22 000 and 50–100 ms, the last of them
1376 blocks out. The worst case is asking for something that is not out there:
the full sweep is ~190 000 samples, about a second, on the main thread. That
bound is deliberate — it is what keeps a miss a hitch instead of a hang.

**Cost, a structure.** Much less, because a structure is not swept for by column
at all. There is exactly one candidate per region — 16 chunks, 256 blocks, for a
village — so the whole 4000-block reach is 1089 candidates. Placing one is two
hashes; they are all placed, sorted by distance, and then *checked* (climate plus
four probe heights, the expensive half) only up to the first that stands.
Measured, a village is 2–11 checks and **0.6 ms**, nearly all of it the hashes
and the sort. Checking every candidate — which cannot happen for a village, and
is what a much rarer structure would cost — is 3.8 ms.

### Adding a structure

`/locate` reads `terrain_generator.STRUCTURES`, and a row there is all a new
building type needs to become findable — no command, no name list, no help text:

```python
STRUCTURES = {
    'village': Structure('Village', VILLAGE_SPACING, village_site, village_check),
}
```

`site(region_x, region_z)` returns the world column the structure would stand on
in that region, and `check(x, z)` returns a tuple whose **first item** says
whether it really does (whatever else the generator wants out of it rides behind,
unread by the search). That is how villages were already placed, and how the
reference places all of its structures, so the search is a walk over regions
rather than over columns.

## `/biome` — what am I standing in

```
/biome        → Jagged Peaks   at 4061 -4073
```

The same reading the F3 screen's `Biome:` line gives, without opening it.

## `/set` — the settings the shortcut keys change

```
/set                       list every setting with its current value
/set <name>                read one
/set <name> <value>        change one
```

Aliases: `/config`

| setting | values | key |
| :--- | :--- | :--- |
| `renderdistance` | 2 – 96 | `+` / `-` |
| `fly` | `on` `off` `toggle` | `TAB` |
| `culling` | `on` `off` `toggle` | `F` |
| `wireframe` | `on` `off` `toggle` | `K` |
| `debug` | `on` `off` `toggle` | `F3` |

`on` also accepts `true`, `yes`, `enable`, `enabled`, `1`; `off` likewise.

```
/set                    → renderdistance 6     chunks loaded around you (2-96)
                          fly            off   flight mode; key TAB
                          culling        on    frustum culling; key F
                          wireframe      off   wireframe pass; key K
                          debug          off   the F3 screen; key F3
/set renderdistance 24
/set fly on
/set culling toggle
```

**A setting's setter is the same call its key press makes.** `/set fly on` goes
through `Camera.toggle_flying`, `/set culling` through
`ThreadedChunkManager.toggle_frustum_culling`, `/set wireframe` through
`ModernGLRenderer.toggle_wireframe` — so the console and the keyboard cannot end
up doing different things to the same flag, and each still prints its own line to
stdout. Render distance is the one that had two copies to keep in step
(`game.render_distance`, which the F3 screen reads, and the chunk manager's,
which actually loads chunks); `commands.set_render_distance` is now the only
writer, and the `+` / `-` keys go through it too.

## `/help`

```
/help              list the commands
/help <command>    usage, description and aliases for one
```

---

## Adding a command

One decorated function in `commands.py`. There is no base class, no plugin
directory and no registration list to keep in step — the decorator *is* the
registration, and `/help` reads the same rows.

```python
@command('seed', 'Print the world seed.')
def _seed(game, args):
    return f'Seed: {WORLD_SEED}'
```

* **`usage`** — its first word is the command's name, so `'tp <x> [y] <z>'`
  registers `tp`. This is the line `/help` prints and the line an argument error
  prints under itself.
* **`help`** — one sentence or three, shown by `/help <name>`.
* **`aliases=('teleport',)`** — other names that reach it. They are looked up in
  a separate table, so an alias can never shadow a real command.
* **`game`** — the `MinecraftModernGL`. Commands reach it for the camera, the
  chunk manager, the renderer, and the two settings the game object owns itself
  (`render_distance`, `show_debug`). What they must *not* do is import pygame or
  moderngl: `commands.py` imports neither, which is why `test_commands.py` can
  run every one of them against a stub with no window open. Anything needing a
  surface or a GL call belongs in `engine/hud.py` behind a method.
* **`args`** — the line after the command name, already split on whitespace.
* **the return** — nothing, a string, or a list of strings. A line may be
  `(text, colour)` where it wants its own; `commands.OK`, `commands.ERROR` and
  `commands.ECHO` are the three in use.

Anything wrong with the arguments is a `CommandError`:

```python
raise CommandError(f'"{args[0]}" is not a coordinate.')
```

`dispatch` prints it in red followed by the command's own usage line, so no
command formats its own error, and none of them has to guard the game loop
against itself.

### Adding a setting

`/set` is table-driven the same way. A yes/no setting is one call, and it takes
the toggle the key press already calls rather than a second setter of its own:

```python
_flag('fly', lambda game: game.camera.flying,
      lambda game: game.camera.toggle_flying(), 'flight mode; key TAB')
```

Anything else takes a parser — `(word, current_value) -> value`, raising
`CommandError` on anything it cannot read:

```python
_setting('renderdistance',
         lambda game: game.render_distance, set_render_distance,
         _int_between(RENDER_MIN, RENDER_MAX),
         f'chunks loaded around you ({RENDER_MIN}-{RENDER_MAX}); keys + and -')
```

`/set` with no arguments, `/set <name>`, and the help text all come off that row,
so a new setting needs nothing else.

### Where the pieces live

| | |
| :--- | :--- |
| `commands.py` | the registry and the commands — no pygame, no OpenGL |
| `main.py` | the console's state and its keyboard: `open_console`, `console_key`, `submit_console`, `update_console` |
| `engine/hud.py` | `set_console` / `render_console`, through the same `_bake_column` the F3 screen uses |
| `test_commands.py` | runs all of it against a stub game — no window |

### Checking it

```bash
python test_commands.py
```

Registry completeness, the teleport landing where it says it did, `~` relative
coordinates, the surface form clearing the ground, names resolving exactly before
partially across both namespaces, **all 26 biomes actually being findable**,
`/locate village` landing on a column the generator really builds one on and on
the *nearest* such column, every setting reading back what was written through
the game's own toggles, and a list of malformed lines that all have to come back
as red text rather than as an exception.
