import math
import random
from typing import Dict, Any, Tuple

import torch
import numpy as np

import genesis as gs
from genesis.utils.geom import quat_to_xyz


def _gs_tensor(x, like_shape=(1,)):
    return torch.tensor(x, device=gs.device, dtype=gs.tc_float).reshape((*like_shape, -1))


def _safe_exp(x: torch.Tensor, limit: float = 60.0) -> torch.Tensor:
    return torch.exp(torch.clamp(x, -limit, limit))


class BoxMaze:
    def __init__(self, grid_size: Tuple[int, int] = (9, 9), corridor: float = 0.5, wall_thickness: float = 0.05,
                 wall_height: float = 1.0, seed: int | None = None, sparsify_factor: int = 10):
        self.grid_h, self.grid_w = grid_size
        self.corridor = corridor
        self.wall_t = wall_thickness
        self.wall_h = wall_height
        self.rng = random.Random(seed)
        self.grid = np.ones((self.grid_h, self.grid_w), dtype=np.int32)
        self.sparsify_factor = max(1, sparsify_factor)

    def _carve(self, i: int, j: int):
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.rng.shuffle(dirs)
        for di, dj in dirs:
            ni, nj = i + 2 * di, j + 2 * dj
            if 1 <= ni < self.grid_h - 1 and 1 <= nj < self.grid_w - 1 and self.grid[ni, nj] == 1:
                self.grid[i + di, j + dj] = 0
                self.grid[ni, nj] = 0
                self._carve(ni, nj)

    def generate(self):
        self.grid.fill(1)
        start_i = self.rng.randrange(1, self.grid_h, 2)
        start_j = self.rng.randrange(1, self.grid_w, 2)
        self.grid[start_i, start_j] = 0
        self._carve(start_i, start_j)
        return self.grid

    def to_boxes(self):
        boxes = []
        cell = self.corridor + self.wall_t
        # outer border
        half_w = (self.grid_w * cell) / 2.0
        half_h = (self.grid_h * cell) / 2.0
        # perimeter walls (always keep)
        boxes.append(dict(pos=(0.0, -half_h - self.wall_t / 2, self.wall_h / 2),
                          size=(self.grid_w * cell + 2 * self.wall_t, self.wall_t, self.wall_h)))
        boxes.append(dict(pos=(0.0, half_h + self.wall_t / 2, self.wall_h / 2),
                          size=(self.grid_w * cell + 2 * self.wall_t, self.wall_t, self.wall_h)))
        boxes.append(dict(pos=(-half_w - self.wall_t / 2, 0.0, self.wall_h / 2),
                          size=(self.wall_t, self.grid_h * cell + 2 * self.wall_t, self.wall_h)))
        boxes.append(dict(pos=(half_w + self.wall_t / 2, 0.0, self.wall_h / 2),
                          size=(self.wall_t, self.grid_h * cell + 2 * self.wall_t, self.wall_h)))
        # internal cells as walls (sparsified)
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                if self.grid[i, j] == 1:
                    if self.rng.randrange(self.sparsify_factor) != 0:
                        continue
                    cx = (j - self.grid_w / 2.0 + 0.5) * cell
                    cy = (i - self.grid_h / 2.0 + 0.5) * cell
                    boxes.append(dict(pos=(cx, cy, self.wall_h / 2), size=(self.corridor, self.corridor, self.wall_h)))
        return boxes


class MazeEnv:
    def __init__(self, num_envs: int = 1, grid_size: Tuple[int, int] = (11, 11), seed: int | None = None,
                 show_viewer: bool = False):
        self.num_envs = num_envs
        self.dt = 0.01
        self.max_FPS = 60
        self.base_hover_rpm = 14468.429183500699

        self.grid_size = grid_size
        self.seed = seed
        self.rng = random.Random(seed)

        # observation: F, R, L
        self.num_obs = 3
        self.num_privileged_obs = None
        self.num_actions = 3  # pitch, yaw, roll

        self._build_scene(show_viewer)

        # set runner device
        self.device = gs.device

        # buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)

        self.prev_rep_eq = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)

        # reward hyperparameters from paper (scaled to avoid overflow)
        self.beta = 0.2
        self.eta = 0.5
        self.kappa = 1.6
        self.omega = 0.5
        self.rho = 1.0
        self.mu = 0.05
        self.delta_const = 1.0
        self.F_threshold = 1.5  # front sector safe dist in meters
        self.step_reward_scale = self.dt

        self.max_episode_length = int(30.0 / self.dt)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_step = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.extras = {"observations": {}}

    def _build_scene(self, show_viewer: bool):
        gs.init(backend=gs.cpu, logging_level="warning")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, -2.0, 1.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=45,
                max_FPS=self.max_FPS,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(10, self.num_envs)))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())

        self.maze = BoxMaze(grid_size=self.grid_size, seed=self.seed, sparsify_factor=10)
        self.maze.generate()
        for wall in self.maze.to_boxes():
            self.scene.add_entity(
                gs.morphs.Box(
                    size=tuple(wall["size"]),
                    pos=tuple(wall["pos"]),
                    fixed=True,
                )
            )

        self.drone = self.scene.add_entity(
            gs.morphs.Drone(
                file="urdf/drones/cf2x.urdf",
                pos=(0.0, 0.0, 0.35),
            )
        )

        if show_viewer:
            try:
                self.scene.viewer.follow_entity(self.drone)
            except AttributeError:
                pass

        self.lidar = self.scene.add_sensor(
            gs.sensors.Lidar(
                pattern=gs.sensors.raycaster.SphericalPattern(fov=(180.0, 10.0), n_points=(181, 1)),
                entity_idx=self.drone.idx,
                pos_offset=(0.0, 0.0, 0.05),
                euler_offset=(0.0, 0.0, 0.0),
                return_world_frame=True,
                draw_debug=True,
            )
        )

        self.scene.build(n_envs=self.num_envs)

    def _read_lidar(self):
        data = self.lidar.read()
        dists = data.distances.reshape(-1)
        dists = torch.clamp(dists, 0.0, 6.0)
        return dists

    def _compute_frontal_sectors(self, dists: torch.Tensor):
        front_idx = slice(80, 101)
        left_idx = slice(0, 80)
        right_idx = slice(101, 181)
        F = torch.min(dists[front_idx]).reshape(1)
        L = torch.min(dists[left_idx]).reshape(1)
        R = torch.min(dists[right_idx]).reshape(1)
        return F, R, L

    def _map_actions_to_rpms(self, actions: torch.Tensor) -> torch.Tensor:
        pitch = torch.clamp(actions[:, 0], 0.0, 1.5)
        yaw = torch.clamp(actions[:, 1], -10.0, 10.0)
        roll = torch.clamp(actions[:, 2], -1.0, 1.0)

        pitch_k = 300.0
        yaw_k = 150.0
        roll_k = 300.0

        dp = pitch * pitch_k
        dy = yaw / 10.0 * yaw_k
        dr = roll * roll_k

        base = torch.full((self.num_envs, 4), self.base_hover_rpm, device=gs.device, dtype=gs.tc_float)
        base[:, 0] += dp
        base[:, 1] += dp
        base[:, 2] -= dp
        base[:, 3] -= dp
        base[:, 0] -= dy
        base[:, 1] += dy
        base[:, 2] -= dy
        base[:, 3] += dy
        base[:, 0] += dr
        base[:, 2] += dr
        base[:, 1] -= dr
        base[:, 3] -= dr

        base = torch.clamp(base, 0.0, 25000.0)
        return base

    def _reward_hybrid(self, F: torch.Tensor, R: torch.Tensor, L: torch.Tensor, rep_eq_prev: torch.Tensor):
        eps = torch.tensor(1e-3, device=gs.device, dtype=gs.tc_float)
        # reward_1
        below = F < self.F_threshold
        r1_pos = self.beta * (self.F_threshold - F) ** 2
        inv = 1.0 / torch.clamp(F - self.F_threshold, min=eps)
        r1_neg = -self.eta * _safe_exp(inv)
        r1 = torch.where(below, r1_pos, r1_neg)
        # reward_2
        d0 = torch.tensor(1.0, device=gs.device, dtype=gs.tc_float)
        lam = torch.tensor(1.0, device=gs.device, dtype=gs.tc_float)

        def U_rep(d):
            d_clamped = torch.clamp(d, min=eps)
            val = lam * (1.0 / d_clamped - 1.0 / d0) ** 2 + 1.0
            return torch.where(d <= d0, val, torch.tensor(1.0, device=gs.device, dtype=gs.tc_float))

        U_R = U_rep(R)
        U_L = U_rep(L)
        ratio1 = U_R / torch.clamp(U_L, min=eps)
        ratio2 = U_L / torch.clamp(U_R, min=eps)
        rep_eq = torch.where(U_R > U_L, ratio1, ratio2)
        delta_rep = rep_eq - rep_eq_prev

        cond1 = (rep_eq < self.kappa) & (delta_rep < 0)
        cond2 = (rep_eq < self.kappa) & (delta_rep > 0)
        exponent2 = -1.0 / torch.clamp(torch.abs(rep_eq - self.rho), min=eps)
        r2_neg = -self.omega * _safe_exp(exponent2)
        r2 = torch.where(cond1, 4 * self.kappa / torch.clamp(rep_eq, min=eps),
                         torch.where(cond2, self.kappa / torch.clamp(rep_eq, min=eps), r2_neg))
        # reward_3
        U_attr = -self.mu * (R - L) ** 2 + self.delta_const
        r3 = 2.0 * U_attr

        return r1, r2, r3, rep_eq

    def reset(self):
        self.reset_buf[:] = True
        self.episode_step[:] = 0
        self.episode_length_buf[:] = 0
        free_cells = np.argwhere(self.maze.grid == 0)
        ci, cj = free_cells[self.rng.randrange(len(free_cells))]
        cell = self.maze.corridor + self.maze.wall_t
        x = (cj - self.grid_size[1] / 2.0 + 0.5) * cell
        y = (ci - self.grid_size[0] / 2.0 + 0.5) * cell
        self.drone.set_pos(_gs_tensor([x, y, 0.35], like_shape=(self.num_envs,)))
        self.drone.set_quat(_gs_tensor([1.0, 0.0, 0.0, 0.0], like_shape=(self.num_envs,)))
        self.prev_rep_eq[:] = 1.0
        self.last_actions[:] = 0.0
        self.scene.step()
        return self._observe(), None

    def _observe(self):
        dists = self._read_lidar()
        F, R, L = self._compute_frontal_sectors(dists)
        self.obs_buf[:] = torch.tensor([F, R, L], device=gs.device, dtype=gs.tc_float)
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf

    def get_observations(self):
        self._observe()
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def step(self, actions: torch.Tensor):
        actions = actions.reshape(self.num_envs, self.num_actions)
        rpms = self._map_actions_to_rpms(actions)
        self.drone.set_propellels_rpm(rpms)
        self.scene.step()

        dists = self._read_lidar()
        F, R, L = self._compute_frontal_sectors(dists)
        r1, r2, r3, rep_eq = self._reward_hybrid(F, R, L, self.prev_rep_eq)
        rew = (r1 + r2 + r3) * self.step_reward_scale
        rew = torch.clamp(rew, -1000.0, 1000.0)
        self.rew_buf[:] = rew
        self.prev_rep_eq[:] = rep_eq

        self.episode_step += 1
        self.episode_length_buf += 1
        timeout = self.episode_length_buf > self.max_episode_length
        crash = (F < 0.2) or (R < 0.15) or (L < 0.15)
        done = crash or bool(timeout[0].item()) if torch.is_tensor(timeout) else (crash or timeout)
        self.reset_buf[:] = done

        time_outs = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        if torch.is_tensor(timeout):
            time_outs[timeout.nonzero(as_tuple=False).reshape((-1,))] = 1.0
        self.extras["time_outs"] = time_outs

        self.obs_buf[:] = torch.tensor([F, R, L], device=gs.device, dtype=gs.tc_float)
        self.extras["observations"]["critic"] = self.obs_buf
        self.last_actions[:] = actions

        _components = torch.cat([r1.reshape(-1), r2.reshape(-1), r3.reshape(-1)], dim=0)
        self.extras["reward_components"] = _components

        if done:
            self.reset()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
