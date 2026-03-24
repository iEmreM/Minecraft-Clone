# Minecraft Clone

<!-- Media Placeholders -->
<div align="center">
  <img src="./images/sc1.png" alt="Screenshot 1" width="48%">
  <img src="./images/sc2.png" alt="Screenshot 2" width="48%">
</div>

A comprehensive voxel-based engine built with Python, delivering a Minecraft-like experience. The project focuses on high performance and smooth gameplay by leveraging modern OpenGL features, multi-threading, and meshing algorithms.

## Features

- **Infinite world:** Terrain is procedurally generated using fast noise and loads dynamically as you explore — there are no borders or pre-built maps.
- **Block interaction:** Raycasted placement and removal with an 8-block reach, supporting eight distinct block types: Grass, Dirt, Stone, Sand, Snow, Leaves, Wood, and Water.
- **Walking & flying:** Switch between grounded movement with gravity and collision, and a free-flying mode useful for building or just exploring quickly.
- **Performance-first rendering:** Render distance is adjustable at runtime (2–96 chunks), chunk geometry is built in background threads so the game never freezes mid-exploration, and both frustum and occlusion culling trim down what actually gets sent to the GPU.
- **Visual detail:** Custom GLSL shaders, a procedural sky, ambient occlusion on block faces, and a transparent animated water surface.

## Controls

| Action | Key / Input |
| :--- | :--- |
| Move | `W` `A` `S` `D` |
| Look | `Mouse` |
| Jump (Walk) / Ascend (Fly) | `Space` |
| Run (Walk) / Descend (Fly) | `Shift` |
| Toggle Fly Mode | `TAB` |
| Toggle Mouse Capture | `ESC` |
| Remove Block | `Left Click` |
| Place Block | `Right Click` |
| Select Block Type | `1` - `8` |
| Adjust Render Distance | `+` / `-` |
| Toggle Frustum Culling | `F` |
| Toggle Occlusion Culling | `O` |

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/iEmreM/Minecraft-Clone.git
   cd Minecraft-Clone
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the game:**
   ```bash
   python main.py
   ```

## Technologies & Technical Architecture

The architecture is designed to overcome Python's performance bottlenecks when handling large arrays of volumetric data, employing a highly modular and optimized tech stack:

- **Core Engine (Python):** The foundation of the game logic and state management.
- **Graphics Pipeline (ModernGL):** A high-performance Python wrapper over modern OpenGL core contexts, replacing legacy fixed-function pipelines with programmable shaders.
- **Window & Input (Pygame):** Handing cross-platform window initialization, event polling, and mouse capturing.
- **Mathematical Operations (NumPy & PyGLM):** Used for intensive matrix transformations and efficient manipulation of large vertex buffers.
- **JIT Compilation (Numba):** Just-In-Time compilation accelerates heavy CPU-bound tasks such as terrain generation and voxel meshing, achieving near-C speeds.

### Key Subsystems
- **ThreadedChunkManager:** Chunk generation and mesh building run on worker threads, so the main loop stays smooth even when loading new areas of the world.
- **FastBuilder:** Uses greedy meshing — adjacent faces of the same block type are merged into a single quad rather than drawn individually, which significantly cuts vertex count and GPU load.
- **Culling Pipeline:** Frustum culling discards chunks entirely outside the camera's view, and occlusion culling skips chunks hidden behind solid geometry. Both can be toggled at runtime for comparison.

## Project Structure

```text
Minecraft-Clone/
├── engine/                 # Core engine components (rendering, camera systems, culling)
├── shaders/                # Custom GLSL vertex and fragment shaders 
├── world/                  # Voxel generation, chunk management, multi-threading
├── main.py                 # Application entry point and game loop
├── requirements.txt        # Python dependency manifest
├── texture.png             # Block texture atlas (spritesheet)
└── README.md               # Project documentation
```

## Contributing

Pull requests are welcome. If you have a bug fix, a performance improvement, or a new feature in mind, feel free to open one. For larger changes, opening an issue first to discuss the approach tends to save time.

## License

Distributed under the terms specified in the `LICENSE` file.
