"""
Snake Game Environment — Gymnasium-compatible
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from enum import IntEnum


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


# (row_delta, col_delta) for each direction
DELTAS = {
    Direction.UP:    (-1,  0),
    Direction.RIGHT: ( 0,  1),
    Direction.DOWN:  ( 1,  0),
    Direction.LEFT:  ( 0, -1),
}

# Actions: 0=straight, 1=turn right, 2=turn left
TURN_RIGHT = {
    Direction.UP:    Direction.RIGHT,
    Direction.RIGHT: Direction.DOWN,
    Direction.DOWN:  Direction.LEFT,
    Direction.LEFT:  Direction.UP,
}
TURN_LEFT = {
    Direction.UP:    Direction.LEFT,
    Direction.LEFT:  Direction.DOWN,
    Direction.DOWN:  Direction.RIGHT,
    Direction.RIGHT: Direction.UP,
}


class SnakeEnv(gym.Env):
    """
    Snake environment with an 11-feature observation vector suitable for DQN.

    Observation (11 floats):
        Danger: straight, right, left
        Direction: up, right, down, left (one-hot)
        Food: up, right, down, left (relative booleans)

    Actions: 0=straight, 1=turn_right, 2=turn_left
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 10}

    def __init__(self, grid_size: int = 20, render_mode: str | None = None):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32
        )

        self._snake: deque[tuple[int, int]] = deque()
        self._direction = Direction.RIGHT
        self._food: tuple[int, int] = (0, 0)
        self._steps = 0
        self._steps_without_food = 0

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mid = self.grid_size // 2
        self._direction = Direction.RIGHT
        self._snake = deque([(mid, mid - 1), (mid, mid), (mid, mid + 1)])
        self._place_food()
        self._steps = 0
        self._steps_without_food = 0

        return self._get_obs(), {}

    # ------------------------------------------------------------------ step
    def step(self, action: int):
        self._steps += 1
        self._steps_without_food += 1

        # Resolve new direction
        if action == 1:
            self._direction = TURN_RIGHT[self._direction]
        elif action == 2:
            self._direction = TURN_LEFT[self._direction]

        dr, dc = DELTAS[self._direction]
        head_r, head_c = self._snake[-1]
        new_head = (head_r + dr, head_c + dc)

        # Collision check
        terminated = self._is_collision(new_head)
        if terminated:
            reward = -10.0
            return self._get_obs(), reward, terminated, False, {"score": len(self._snake) - 3}

        # Timeout to prevent infinite loops
        truncated = self._steps_without_food > 100 * self.grid_size

        self._snake.append(new_head)

        if new_head == self._food:
            reward = 10.0
            self._steps_without_food = 0
            self._place_food()
        else:
            self._snake.popleft()
            reward = 0.0

        return self._get_obs(), reward, False, truncated, {"score": len(self._snake) - 3}

    # ----------------------------------------------------------- observation
    def _get_obs(self) -> np.ndarray:
        head = self._snake[-1]
        dir_ = self._direction

        def danger(d: Direction) -> float:
            dr, dc = DELTAS[d]
            nxt = (head[0] + dr, head[1] + dc)
            return float(self._is_collision(nxt))

        # Straight / right / left relative to current heading
        d_straight = danger(dir_)
        d_right    = danger(TURN_RIGHT[dir_])
        d_left     = danger(TURN_LEFT[dir_])

        # One-hot direction
        dir_up    = float(dir_ == Direction.UP)
        dir_right = float(dir_ == Direction.RIGHT)
        dir_down  = float(dir_ == Direction.DOWN)
        dir_left  = float(dir_ == Direction.LEFT)

        # Food relative position
        food_r, food_c = self._food
        h_r, h_c = head
        food_up    = float(food_r < h_r)
        food_right = float(food_c > h_c)
        food_down  = float(food_r > h_r)
        food_left  = float(food_c < h_c)

        return np.array([
            d_straight, d_right, d_left,
            dir_up, dir_right, dir_down, dir_left,
            food_up, food_right, food_down, food_left,
        ], dtype=np.float32)

    # --------------------------------------------------------------- helpers
    def _is_collision(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
            return True
        # Exclude tail tip — it will move away this step (except when eating)
        body = list(self._snake)[1:]
        return pos in body

    def _place_food(self):
        snake_set = set(self._snake)
        while True:
            pos = (
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size),
            )
            if pos not in snake_set:
                self._food = pos
                break

    # ---------------------------------------------------------------- render
    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            return self._render_rgb()

    def _render_ansi(self) -> str:
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for r, c in self._snake:
            grid[r][c] = "O"
        grid[self._snake[-1][0]][self._snake[-1][1]] = "H"
        grid[self._food[0]][self._food[1]] = "X"
        rows = ["".join(row) for row in grid]
        return "\n".join(["+" + "-" * self.grid_size + "+"] + [f"|{r}|" for r in rows] + ["+" + "-" * self.grid_size + "+"])

    def _render_rgb(self) -> np.ndarray:
        cell = 20
        size = self.grid_size * cell
        img = np.zeros((size, size, 3), dtype=np.uint8)
        # Background
        img[:] = [30, 30, 30]
        # Snake body
        for r, c in list(self._snake)[:-1]:
            img[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = [0, 200, 0]
        # Snake head
        hr, hc = self._snake[-1]
        img[hr*cell:(hr+1)*cell, hc*cell:(hc+1)*cell] = [0, 255, 0]
        # Food
        fr, fc = self._food
        img[fr*cell:(fr+1)*cell, fc*cell:(fc+1)*cell] = [220, 50, 50]
        return img
