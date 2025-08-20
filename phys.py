# sand_sim_numpy_pygame.py
# Run:
#   pip install pygame numpy numba   # numba is optional; see note below
#   python sand_sim_numpy_pygame.py
#
# Notes:
# - Y increases DOWN (top row is y=0). "Above" means y-1.
# - Controls: Q=Sand, W=Water, E=Wall, T=Plant-dry, Y=Smoke, F=Fire, R=Eraser, C=clear, [ / ] brush size, Esc=quit
# - Matches your plant rules:
#   (1) Plant-dry -> Plant-wet if touching WATER and SAND simultaneously; consumes ONE adjacent WATER.
#   (2) Plant-wet transposes with Plant-dry directly above (runs before sand/water).
#   (3) Plant-wet growth: if empty above:
#       - if (row % 4) != 0, add Plant-dry straight above (y-1)
#       - if (row % 4) == 0, add Plant-dry to up-left, up, up-right (y-1, x-1..x+1)

import sys
import math
import random
from typing import Tuple

import numpy as np
import pygame

# Optional: JIT accelerate the step() with Numba if available
try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def wrap(f): return f
        return wrap

# ---------- World settings ----------
W, H = 256, 160            # grid width/height (cells). y increases downward
PIXEL_SCALE = 4            # upscaling for the window
STEPS_PER_FRAME = 2
BRUSH_MIN, BRUSH_MAX = 1, 16

# ---------- Materials ----------
EMPTY, WALL, SAND, WATER, PLANT_DRY, PLANT_WET, SMOKE, FIRE, STEAM = 0, 1, 2, 3, 4, 5, 6, 7, 8

PALETTE = np.array([
    [0.00, 0.00, 0.00],  # EMPTY
    [0.10, 0.10, 0.10],  # WALL
    [0.90, 0.80, 0.50],  # SAND
    [0.30, 0.55, 0.95],  # WATER
    [0.20, 0.80, 0.25],  # PLANT_DRY
    [0.10, 0.55, 0.15],  # PLANT_WET
    [0.60, 0.60, 0.60],  # SMOKE
    [1.00, 0.30, 0.00],  # FIRE
    [0.85, 0.85, 0.90],  # STEAM
], dtype=np.float32)

FIRE_LIFE_INIT = 5
STEAM_LIFE_INIT = 20

MAT_NAME = {
    EMPTY: "Empty",
    SAND: "Sand",
    WATER: "Water",
    WALL: "Wall",
    PLANT_DRY: "Plant-dry",
    PLANT_WET: "Plant-wet",
    SMOKE: "Smoke",
    FIRE: "Fire",
    STEAM: "Steam",
}

# ---------- Core simulation ----------
@njit(cache=True, fastmath=True)
def in_bounds(x: int, y: int, W: int, H: int) -> bool:
    return 0 <= x < W and 0 <= y < H

@njit(cache=True, fastmath=True)
def paint_circle(grid: np.ndarray, cx: int, cy: int, r: int, mat: np.uint8):
    H, W = grid.shape[0], grid.shape[1]  # careful: we store as [H,W]
    r2 = r * r
    for dy in range(-r, r + 1):
        yy = cy + dy
        if yy < 0 or yy >= H:  # H rows
            continue
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r2:
                xx = cx + dx
                if xx < 0 or xx >= W:
                    continue
                grid[yy, xx] = mat

@njit(cache=True)
def step(grid: np.ndarray, grid_next: np.ndarray, kill_water: np.ndarray,
         fire_life: np.ndarray, fire_life_next: np.ndarray,
         steam_life: np.ndarray, steam_life_next: np.ndarray):
    """
    grid, grid_next: uint8 arrays [H, W] (y, x)
    kill_water: uint8 mask [H, W] (0/1)
    fire_life/steam_life: uint8 arrays for remaining lifetime
    """
    H, W = grid.shape
    # reset next + mask
    for y in range(H):
        for x in range(W):
            grid_next[y, x] = EMPTY
            kill_water[y, x] = 0
            fire_life_next[y, x] = 0
            steam_life_next[y, x] = 0

    # ===== Phase 1: Plants (before sand/water) =====
    for y in range(H):
        for x in range(W):
            m = grid[y, x]

            if m == WALL:
                grid_next[y, x] = WALL
                continue

            # Rule 2: PLANT_WET transposes with PLANT_DRY directly above (y-1)
            if m == PLANT_WET:
                if y - 1 >= 0 and grid[y - 1, x] == PLANT_DRY:
                    grid_next[y, x] = PLANT_DRY
                    grid_next[y - 1, x] = PLANT_WET
                    # skip growth if we transposed this tick
                    continue

                # Rule 3: growth upward if empty above
                if y - 1 >= 0:
                    if (y % 4) == 0:
                        # up-left, up, up-right (keypad 7,8,9 relative to 5)
                        for dx in (-1, 0, 1):
                            nx = x + dx
                            ny = y - 1
                            if 0 <= nx < W and grid[ny, nx] == EMPTY and grid_next[ny, nx] == EMPTY:
                                grid_next[ny, nx] = PLANT_DRY
                    else:
                        ny, nx = y - 1, x
                        if grid[ny, nx] == EMPTY and grid_next[ny, nx] == EMPTY:
                            grid_next[ny, nx] = PLANT_DRY

                # keep wet where it is unless already written
                if grid_next[y, x] == EMPTY:
                    grid_next[y, x] = PLANT_WET
                continue

            if m == PLANT_DRY:
                # Rule 1: becomes wet if touching WATER and SAND simultaneously.
                # Also consume ONE adjacent water cell (first found).
                found_sand = False
                wx, wy = -1, -1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = x + dx
                        ny = y + dy
                        if not in_bounds(nx, ny, W, H):
                            continue
                        n = grid[ny, nx]
                        if n == SAND:
                            found_sand = True
                        elif n == WATER and wx < 0:
                            wx, wy = nx, ny

                if found_sand and wx >= 0:
                    grid_next[y, x] = PLANT_WET
                    kill_water[wy, wx] = 1  # mark that water cell for removal
                else:
                    grid_next[y, x] = PLANT_DRY
                continue

            # others handled later

    # ===== Phase 2: Sand & Water physics =====
    for y in range(H):
        for x in range(W):
            if grid_next[y, x] != EMPTY:
                continue
            m = grid[y, x]
            if m == EMPTY or m == WALL or m == PLANT_DRY or m == PLANT_WET:
                # already handled or static
                continue

            if m == SAND:
                moved = False
                # down (y+1)
                if y + 1 < H and grid[y + 1, x] == EMPTY and grid_next[y + 1, x] == EMPTY:
                    grid_next[y + 1, x] = SAND
                    moved = True
                else:
                    # diagonals (random order to reduce bias)
                    d = 1 if random.random() < 0.5 else -1
                    nx = x + d
                    if y + 1 < H and 0 <= nx < W and grid[y + 1, nx] == EMPTY and grid_next[y + 1, nx] == EMPTY:
                        grid_next[y + 1, nx] = SAND
                        moved = True
                    else:
                        nx2 = x - d
                        if y + 1 < H and 0 <= nx2 < W and grid[y + 1, nx2] == EMPTY and grid_next[y + 1, nx2] == EMPTY:
                            grid_next[y + 1, nx2] = SAND
                            moved = True

                if not moved:
                    if grid_next[y, x] == EMPTY:
                        grid_next[y, x] = SAND
                continue

            if m == WATER:
                # consumed by plant this tick?
                if kill_water[y, x] == 1:
                    # treat as empty: don't write anything
                    continue
                extinguish = False
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = x + dx
                        ny = y + dy
                        if not in_bounds(nx, ny, W, H):
                            continue
                        if grid[ny, nx] == FIRE:
                            extinguish = True
                            grid_next[ny, nx] = SMOKE
                if extinguish:
                    grid_next[y, x] = STEAM
                    steam_life_next[y, x] = STEAM_LIFE_INIT
                    continue
                moved = False
                # down
                if y + 1 < H and grid[y + 1, x] == EMPTY and grid_next[y + 1, x] == EMPTY:
                    grid_next[y + 1, x] = WATER
                    moved = True
                else:
                    d = 1 if random.random() < 0.5 else -1
                    nx = x + d
                    # diagonals down
                    if y + 1 < H and 0 <= nx < W and grid[y + 1, nx] == EMPTY and grid_next[y + 1, nx] == EMPTY:
                        grid_next[y + 1, nx] = WATER
                        moved = True
                    else:
                        nx2 = x - d
                        if y + 1 < H and 0 <= nx2 < W and grid[y + 1, nx2] == EMPTY and grid_next[y + 1, nx2] == EMPTY:
                            grid_next[y + 1, nx2] = WATER
                            moved = True
                        # sideways
                        elif 0 <= nx < W and grid[y, nx] == EMPTY and grid_next[y, nx] == EMPTY:
                            grid_next[y, nx] = WATER
                            moved = True
                        elif 0 <= nx2 < W and grid[y, nx2] == EMPTY and grid_next[y, nx2] == EMPTY:
                            grid_next[y, nx2] = WATER
                            moved = True

                if not moved:
                    if grid_next[y, x] == EMPTY:
                        grid_next[y, x] = WATER
                continue

            if m == FIRE:
                life = fire_life[y, x] - 1
                extinguished = False
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = x + dx
                        ny = y + dy
                        if not in_bounds(nx, ny, W, H):
                            continue
                        n = grid[ny, nx]
                        if n == WATER:
                            extinguished = True
                            grid_next[ny, nx] = STEAM
                            steam_life_next[ny, nx] = STEAM_LIFE_INIT
                        elif n == PLANT_DRY or n == PLANT_WET:
                            grid_next[ny, nx] = FIRE
                            fire_life_next[ny, nx] = FIRE_LIFE_INIT
                if extinguished:
                    grid_next[y, x] = SMOKE
                    continue
                moved = False
                if y - 1 >= 0 and grid[y - 1, x] == EMPTY and grid_next[y - 1, x] == EMPTY:
                    grid_next[y - 1, x] = FIRE
                    fire_life_next[y - 1, x] = life
                    moved = True
                else:
                    d = 1 if random.random() < 0.5 else -1
                    nx = x + d
                    if y - 1 >= 0 and 0 <= nx < W and grid[y - 1, nx] == EMPTY and grid_next[y - 1, nx] == EMPTY:
                        grid_next[y - 1, nx] = FIRE
                        fire_life_next[y - 1, nx] = life
                        moved = True
                    else:
                        nx2 = x - d
                        if y - 1 >= 0 and 0 <= nx2 < W and grid[y - 1, nx2] == EMPTY and grid_next[y - 1, nx2] == EMPTY:
                            grid_next[y - 1, nx2] = FIRE
                            fire_life_next[y - 1, nx2] = life
                            moved = True
                        elif 0 <= nx < W and grid[y, nx] == EMPTY and grid_next[y, nx] == EMPTY:
                            grid_next[y, nx] = FIRE
                            fire_life_next[y, nx] = life
                            moved = True
                        elif 0 <= nx2 < W and grid[y, nx2] == EMPTY and grid_next[y, nx2] == EMPTY:
                            grid_next[y, nx2] = FIRE
                            fire_life_next[y, nx2] = life
                            moved = True
                if not moved:
                    if life > 0:
                        grid_next[y, x] = FIRE
                        fire_life_next[y, x] = life
                    else:
                        grid_next[y, x] = SMOKE
                continue

            if m == STEAM:
                life = steam_life[y, x] - 1
                moved = False
                if y - 1 >= 0 and grid[y - 1, x] == EMPTY and grid_next[y - 1, x] == EMPTY:
                    grid_next[y - 1, x] = STEAM
                    steam_life_next[y - 1, x] = life
                    moved = True
                else:
                    d = 1 if random.random() < 0.5 else -1
                    nx = x + d
                    if y - 1 >= 0 and 0 <= nx < W and grid[y - 1, nx] == EMPTY and grid_next[y - 1, nx] == EMPTY:
                        grid_next[y - 1, nx] = STEAM
                        steam_life_next[y - 1, nx] = life
                        moved = True
                    else:
                        nx2 = x - d
                        if y - 1 >= 0 and 0 <= nx2 < W and grid[y - 1, nx2] == EMPTY and grid_next[y - 1, nx2] == EMPTY:
                            grid_next[y - 1, nx2] = STEAM
                            steam_life_next[y - 1, nx2] = life
                            moved = True
                        elif 0 <= nx < W and grid[y, nx] == EMPTY and grid_next[y, nx] == EMPTY:
                            grid_next[y, nx] = STEAM
                            steam_life_next[y, nx] = life
                            moved = True
                        elif 0 <= nx2 < W and grid[y, nx2] == EMPTY and grid_next[y, nx2] == EMPTY:
                            grid_next[y, nx2] = STEAM
                            steam_life_next[y, nx2] = life
                            moved = True
                if not moved:
                    if life > 0:
                        grid_next[y, x] = STEAM
                        steam_life_next[y, x] = life
                    else:
                        grid_next[y, x] = WATER
                continue

            if m == SMOKE:
                moved = False
                # up
                if y - 1 >= 0 and grid[y - 1, x] == EMPTY and grid_next[y - 1, x] == EMPTY:
                    grid_next[y - 1, x] = SMOKE
                    moved = True
                else:
                    d = 1 if random.random() < 0.5 else -1
                    nx = x + d
                    # diagonals up
                    if y - 1 >= 0 and 0 <= nx < W and grid[y - 1, nx] == EMPTY and grid_next[y - 1, nx] == EMPTY:
                        grid_next[y - 1, nx] = SMOKE
                        moved = True
                    else:
                        nx2 = x - d
                        if y - 1 >= 0 and 0 <= nx2 < W and grid[y - 1, nx2] == EMPTY and grid_next[y - 1, nx2] == EMPTY:
                            grid_next[y - 1, nx2] = SMOKE
                            moved = True
                        # sideways
                        elif 0 <= nx < W and grid[y, nx] == EMPTY and grid_next[y, nx] == EMPTY:
                            grid_next[y, nx] = SMOKE
                            moved = True
                        elif 0 <= nx2 < W and grid[y, nx2] == EMPTY and grid_next[y, nx2] == EMPTY:
                            grid_next[y, nx2] = SMOKE
                            moved = True

                if not moved:
                    if grid_next[y, x] == EMPTY:
                        grid_next[y, x] = SMOKE
                continue

    # commit
    for y in range(H):
        for x in range(W):
            grid[y, x] = grid_next[y, x]
            fire_life[y, x] = fire_life_next[y, x]
            steam_life[y, x] = steam_life_next[y, x]

# ---------- Rendering ----------
def render_surface(grid: np.ndarray) -> pygame.Surface:
    """Map grid -> RGB surface (unscaled)."""
    # grid is [H,W]; palette index to [H,W,3] floats 0..1
    rgb = (PALETTE[grid] * 255.0).astype(np.uint8)  # shape [H,W,3]
    surf = pygame.image.frombuffer(rgb.tobytes(), (grid.shape[1], grid.shape[0]), "RGB")
    return surf.convert()

# ---------- App ----------
def main():
    pygame.init()
    size = (W * PIXEL_SCALE, H * PIXEL_SCALE)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Falling Sand — NumPy/Pygame")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    # Grid is stored as [H, W] for convenient row-major access
    grid = np.zeros((H, W), dtype=np.uint8)
    grid_next = np.zeros_like(grid)
    kill_water = np.zeros_like(grid)
    fire_life = np.zeros_like(grid)
    fire_life_next = np.zeros_like(grid)
    steam_life = np.zeros_like(grid)
    steam_life_next = np.zeros_like(grid)

    # Optional: a floor at the bottom
    grid[H - 1, :] = WALL

    brush_radius = 6
    current_mat = SAND

    running = True
    while running:
        # --- input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_c:
                    grid[:, :] = EMPTY
                    grid_next[:, :] = EMPTY
                    kill_water[:, :] = 0
                    fire_life[:, :] = 0
                    fire_life_next[:, :] = 0
                    steam_life[:, :] = 0
                    steam_life_next[:, :] = 0
                elif k == pygame.K_LEFTBRACKET:
                    brush_radius = max(BRUSH_MIN, brush_radius - 1)
                elif k == pygame.K_RIGHTBRACKET:
                    brush_radius = min(BRUSH_MAX, brush_radius + 1)

        # Poll for material hotkeys (letter bindings, more reliable)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            current_mat = SAND
        elif keys[pygame.K_w]:
            current_mat = WATER
        elif keys[pygame.K_e]:
            current_mat = WALL
        elif keys[pygame.K_t]:
            current_mat = PLANT_DRY
        elif keys[pygame.K_y]:
            current_mat = SMOKE
        elif keys[pygame.K_f]:
            current_mat = FIRE
        elif keys[pygame.K_r]:
            current_mat = EMPTY

        # Paint with mouse
        mx, my = pygame.mouse.get_pos()
        gx = int(mx / PIXEL_SCALE)
        gy = int(my / PIXEL_SCALE)
        buttons = pygame.mouse.get_pressed(3)
        if (0 <= gx < W) and (0 <= gy < H):
            if buttons[0]:  # LMB — paint
                paint_circle(grid, gx, gy, brush_radius, np.uint8(current_mat))
                if current_mat == FIRE:
                    paint_circle(fire_life, gx, gy, brush_radius, np.uint8(FIRE_LIFE_INIT))
                else:
                    paint_circle(fire_life, gx, gy, brush_radius, np.uint8(0))
                paint_circle(steam_life, gx, gy, brush_radius, np.uint8(0))
            elif buttons[2]:  # RMB — erase
                paint_circle(grid, gx, gy, brush_radius, np.uint8(EMPTY))
                paint_circle(fire_life, gx, gy, brush_radius, np.uint8(0))
                paint_circle(steam_life, gx, gy, brush_radius, np.uint8(0))

        # --- simulation ---
        for _ in range(STEPS_PER_FRAME):
            step(grid, grid_next, kill_water,
                 fire_life, fire_life_next,
                 steam_life, steam_life_next)

        # --- draw ---
        surf = render_surface(grid)
        if PIXEL_SCALE != 1:
            surf = pygame.transform.scale(surf, size)
        screen.blit(surf, (0, 0))

        # HUD
        hud_lines = [
            f"Material: {MAT_NAME.get(int(current_mat), 'Unknown')}  (Q:Sand W:Water E:Wall T:Plant Y:Smoke F:Fire R:Eraser)",
            f"Brush: {brush_radius}  ([ / ])    Grid: {W}x{H}   Steps/frame: {STEPS_PER_FRAME}",
            "LMB paint  RMB erase  C clear  Esc quit",
        ]
        y = 6
        for line in hud_lines:
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (6, y))
            y += 18

        pygame.display.flip()
        clock.tick(60)  # cap to ~60 FPS; adjust as needed

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
