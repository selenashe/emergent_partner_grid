"""Standalone verification script for CoordinationGrid + scripted partner.

Run: python dev/test_coordination_grid.py
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxmarl
from jaxmarl.environments.coordination_grid import (
    CoordinationGrid,
    State,
    Actions,
    Messages,
    encode_ego,
    decode_ego,
    N_EGO_ACTIONS,
    GOAL_UNSET,
    GOAL_RED,
    GOAL_BLUE,
)
from jaxmarl.environments.coordination_grid.coordination_grid import (
    _sym_position,
    _sym_wall,
    N_SYMMETRIES,
    SYMMETRY_NAMES,
    LEGAL_ACTION_IDS_T0,
    LEGAL_ACTION_IDS_TGEQ1,
    ACTION_MASK_T0,
    ACTION_MASK_TGEQ1,
)


UP, DOWN, RIGHT, LEFT, STAY = 0, 1, 2, 3, 4
NONE, M0, M1 = 0, 1, 2


def make_env(**kw):
    return jaxmarl.make("coordination_grid", **kw)


def _set_state(env, ego_xy, partner_xy, *, time=0, partner_goal=GOAL_UNSET,
               pending_message=NONE, z=None, terminal=False, layout_idx=0):
    """Manually construct a State for targeted transition tests."""
    z = env.partner_z if z is None else z
    return State(
        agent_pos=jnp.array([ego_xy, partner_xy], dtype=jnp.int32),
        wall_map=env.wall_map,
        red_goal=env.red_goal,
        blue_goal=env.blue_goal,
        time=jnp.int32(time),
        terminal=jnp.bool_(terminal),
        z=jnp.float32(z),
        partner_goal=jnp.int32(partner_goal),
        pending_message=jnp.int32(pending_message),
        layout_idx=jnp.int32(layout_idx),
    )


def check(name, cond):
    status = "OK   " if bool(cond) else "FAIL "
    print(f"  [{status}] {name}")
    assert bool(cond), f"assertion failed: {name}"


def main():
    env = make_env()
    print(
        f"env: {env.name} H={env.height} W={env.width} "
        f"grid_shape={env.grid_shape} msg_shape={env.msg_shape} "
        f"n_ego_actions={env.n_ego_actions} max_steps={env.max_steps} "
        f"z={env.partner_z}"
    )
    print(f"ego_start={tuple(np.array(env.ego_start))} "
          f"partner_start={tuple(np.array(env.partner_start))} "
          f"red={tuple(np.array(env.red_goal))} "
          f"blue={tuple(np.array(env.blue_goal))}")

    key = jax.random.PRNGKey(0)

    # ------------------ encode/decode round-trip ------------------
    print("\n[encode/decode]")
    for mv in range(5):
        for msg in range(3):
            a = encode_ego(mv, msg)
            mv2, msg2 = decode_ego(a)
            check(f"encode/decode ({mv},{msg}) -> {a}",
                  int(mv2) == mv and int(msg2) == msg)
    check("N_EGO_ACTIONS == 15", N_EGO_ACTIONS == 15)

    # ------------------ reset ------------------
    print("\n[reset]")
    obs, state = env.reset(key)
    check("agent_0 at ego_start", jnp.all(state.agent_pos[0] == env.ego_start))
    check("agent_1 at partner_start",
          jnp.all(state.agent_pos[1] == env.partner_start))
    check("time == 0", int(state.time) == 0)
    check("partner_goal UNSET at reset", int(state.partner_goal) == GOAL_UNSET)
    check("pending_message NONE at reset", int(state.pending_message) == NONE)
    check("z copied from constructor", float(state.z) == float(env.partner_z))
    check("layout_idx == 0", int(state.layout_idx) == 0)

    # ------------------ obs structure ------------------
    print("\n[obs dict]")
    check("obs is dict per agent", isinstance(obs, dict)
          and "agent_0" in obs and "agent_1" in obs)
    for aid in ("agent_0", "agent_1"):
        check(f"{aid} obs is dict with grid+last_message+is_t0",
              isinstance(obs[aid], dict)
              and "grid" in obs[aid] and "last_message" in obs[aid]
              and "is_t0" in obs[aid])
        check(f"{aid} grid shape", obs[aid]["grid"].shape == (env.height, env.width, 5))
        check(f"{aid} last_message shape", obs[aid]["last_message"].shape == (3,))
        check(f"{aid} is_t0 shape", obs[aid]["is_t0"].shape == ())
    check("last_message one-hot on NONE at reset",
          jnp.allclose(obs["agent_0"]["last_message"], jnp.array([1.0, 0.0, 0.0])))
    check("is_t0 == 1.0 at reset", float(obs["agent_0"]["is_t0"]) == 1.0)
    ex, ey = int(env.ego_start[0]), int(env.ego_start[1])
    check("ego channel one-hot on grid",
          float(obs["agent_0"]["grid"][ey, ex, 3]) == 1.0)

    # ------------------ t=0 both agents stationary; msg recorded ------------
    print("\n[t=0 stationary + msg recorded]")
    # Ego tries to move UP at t=0; must NOT move. Msg=M0 must be recorded.
    _, s1, r1, d1, info1 = env.step_env(
        key, state, {"agent_0": encode_ego(UP, M0)}
    )
    check("t=0 ego did not move",
          jnp.all(s1.agent_pos[0] == env.ego_start))
    check("t=0 partner did not move",
          jnp.all(s1.agent_pos[1] == env.partner_start))
    check("t=0 pending_message == M0 after step",
          int(s1.pending_message) == M0)
    check("t=0 partner_goal still UNSET",
          int(s1.partner_goal) == GOAL_UNSET)
    check("t=0 obs last_message reflects M0",
          jnp.allclose(env.get_obs(s1)["agent_0"]["last_message"],
                       jnp.array([0.0, 1.0, 0.0])))
    check("is_t0 == 0.0 after first step (state.time=1)",
          float(env.get_obs(s1)["agent_0"]["is_t0"]) == 0.0)
    check("t=0 no success", not bool(info1["success"]))
    check("t=0 done False", not bool(d1["__all__"]))
    check("t=0 time == 1", int(s1.time) == 1)

    # ------------------ z=1: M0 -> RED, M1 -> BLUE deterministically ---------
    print("\n[deterministic goal commit at z=1 and z=0]")
    env_z1 = make_env(partner_z=1.0)
    env_z0 = make_env(partner_z=0.0)

    # z=1, msg=M0 at t=0 -> commit RED at t=1 with prob 1.
    _, s0 = env_z1.reset(key)
    _, s1z1_m0, _, _, _ = env_z1.step_env(
        key, s0, {"agent_0": encode_ego(STAY, M0)}
    )
    _, s2z1_m0, _, _, info2 = env_z1.step_env(
        key, s1z1_m0, {"agent_0": encode_ego(STAY, NONE)}
    )
    check("z=1 + M0 commits RED", int(s2z1_m0.partner_goal) == GOAL_RED)

    # z=1, msg=M1 at t=0 -> commit BLUE.
    _, s0 = env_z1.reset(key)
    _, s1z1_m1, _, _, _ = env_z1.step_env(
        key, s0, {"agent_0": encode_ego(STAY, M1)}
    )
    _, s2z1_m1, _, _, _ = env_z1.step_env(
        key, s1z1_m1, {"agent_0": encode_ego(STAY, NONE)}
    )
    check("z=1 + M1 commits BLUE", int(s2z1_m1.partner_goal) == GOAL_BLUE)

    # z=0, msg=M0 -> commit BLUE.
    _, s0 = env_z0.reset(key)
    _, s1z0_m0, _, _, _ = env_z0.step_env(
        key, s0, {"agent_0": encode_ego(STAY, M0)}
    )
    _, s2z0_m0, _, _, _ = env_z0.step_env(
        key, s1z0_m0, {"agent_0": encode_ego(STAY, NONE)}
    )
    check("z=0 + M0 commits BLUE", int(s2z0_m0.partner_goal) == GOAL_BLUE)

    # z=0, msg=M1 -> commit RED.
    _, s0 = env_z0.reset(key)
    _, s1z0_m1, _, _, _ = env_z0.step_env(
        key, s0, {"agent_0": encode_ego(STAY, M1)}
    )
    _, s2z0_m1, _, _, _ = env_z0.step_env(
        key, s1z0_m1, {"agent_0": encode_ego(STAY, NONE)}
    )
    check("z=0 + M1 commits RED", int(s2z0_m1.partner_goal) == GOAL_RED)

    # ------------------ NONE at t=0: ~50/50 across seeds ---------------------
    print("\n[NONE at t=0: 50/50 prior]")
    n_trials = 400
    reds = 0
    env_neutral = make_env(partner_z=0.7)
    keys = jax.random.split(key, n_trials)
    for k in keys:
        _, s0 = env_neutral.reset(k)
        _, s1, _, _, _ = env_neutral.step_env(k, s0, {"agent_0": encode_ego(STAY, NONE)})
        _, s2, _, _, _ = env_neutral.step_env(k, s1, {"agent_0": encode_ego(STAY, NONE)})
        reds += int(s2.partner_goal) == GOAL_RED
    frac_red = reds / n_trials
    print(f"    P(RED | NONE) empirical = {frac_red:.3f}  (target 0.5)")
    check("NONE gives ~50/50 partner goal", 0.40 <= frac_red <= 0.60)

    # ------------------ commit-once: later msg does not change goal ----------
    print("\n[commit-once semantics]")
    env_z1 = make_env(partner_z=1.0)
    _, s0 = env_z1.reset(key)
    # t=0: M0 -> partner should commit RED at t=1.
    _, s1, _, _, _ = env_z1.step_env(key, s0, {"agent_0": encode_ego(STAY, M0)})
    # t=1: (partner commits) and ego now sends M1 (which would flip to BLUE if it could).
    _, s2, _, _, _ = env_z1.step_env(key, s1, {"agent_0": encode_ego(STAY, M1)})
    check("goal committed to RED at t=1", int(s2.partner_goal) == GOAL_RED)
    # t=2: ego keeps sending M1; goal must remain RED.
    _, s3, _, _, _ = env_z1.step_env(key, s2, {"agent_0": encode_ego(STAY, M1)})
    check("goal STAYS RED after later M1", int(s3.partner_goal) == GOAL_RED)

    # ------------------ partner navigation goes to committed goal ------------
    print("\n[partner walks toward committed goal]")
    env_z1 = make_env(partner_z=1.0, max_steps=25)
    _, s = env_z1.reset(key)
    # tell partner RED with M0 at t=0; ego STAYs afterward so only partner moves.
    _, s, _, _, _ = env_z1.step_env(key, s, {"agent_0": encode_ego(STAY, M0)})
    partner_positions = [tuple(int(v) for v in s.agent_pos[1])]
    for _ in range(env_z1.max_steps - 1):
        _, s, _, d, _ = env_z1.step_env(key, s, {"agent_0": encode_ego(STAY, NONE)})
        partner_positions.append(tuple(int(v) for v in s.agent_pos[1]))
        if bool(d["__all__"]):
            break
    red_xy = tuple(int(v) for v in env_z1.red_goal)
    check("partner reached RED", red_xy in partner_positions)

    # ------------------ full comm-guided rollout succeeds --------------------
    print("\n[comm-guided rollout end-to-end]")
    # z=1, ego sends M0 at t=0 -> partner commits RED. Ego walks to BLUE.
    env_z1 = make_env(partner_z=1.0, max_steps=25)
    _, s = env_z1.reset(key)
    blue_xy = tuple(int(v) for v in env_z1.blue_goal)

    # BFS from ego_start to BLUE for ego's move plan.
    from collections import deque
    def bfs(w, sx, sy, gx, gy):
        H, W = w.shape
        visited = {(sx, sy): None}
        q = deque([(sx, sy)])
        while q:
            cx, cy = q.popleft()
            if (cx, cy) == (gx, gy):
                break
            for a, dx, dy in [(UP, 0, -1), (DOWN, 0, 1), (RIGHT, 1, 0), (LEFT, -1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H and not w[ny, nx] and (nx, ny) not in visited:
                    visited[(nx, ny)] = ((cx, cy), a)
                    q.append((nx, ny))
        if (gx, gy) not in visited:
            return []
        acts = []
        cur = (gx, gy)
        while visited[cur] is not None:
            prev, a = visited[cur]
            acts.append(a); cur = prev
        return list(reversed(acts))

    wall_np = np.array(env_z1.wall_map)
    ex, ey = int(env_z1.ego_start[0]), int(env_z1.ego_start[1])
    ego_plan = bfs(wall_np, ex, ey, blue_xy[0], blue_xy[1])

    # Step 1 (t=0): STAY + M0.
    _, s, _, d, info = env_z1.step_env(key, s, {"agent_0": encode_ego(STAY, M0)})
    trace_done = False
    for i in range(env_z1.max_steps - 1):
        mv = ego_plan[i] if i < len(ego_plan) else STAY
        _, s, r, d, info = env_z1.step_env(key, s, {"agent_0": encode_ego(mv, NONE)})
        if bool(d["__all__"]):
            trace_done = True
            break
    check("comm-guided rollout terminated", trace_done)
    check("comm-guided rollout success", bool(info["success"]))
    check("comm-guided rollout final reward > 0", float(r["agent_0"]) > 0)

    # ------------------ walls / boundaries / collision still work -----------
    print("\n[movement primitives after t=0]")
    # Bypass t=0 gating by hand-constructing state with time>=1 and a goal committed.
    s = _set_state(env, ego_xy=(1, 6), partner_xy=(3, 3),
                   time=2, partner_goal=GOAL_RED)
    # LEFT would hit wall at (0, 6).
    _, s2, _, _, _ = env.step_env(key, s, {"agent_0": encode_ego(LEFT, NONE)})
    check("wall blocks ego LEFT at (1,6)",
          jnp.all(s2.agent_pos[0] == jnp.array([1, 6])))
    # DOWN would leave the grid at (1, 7).
    _, s2, _, _, _ = env.step_env(key, s, {"agent_0": encode_ego(DOWN, NONE)})
    check("boundary blocks ego DOWN at (1,6)",
          jnp.all(s2.agent_pos[0] == jnp.array([1, 6])))

    # ------------------ timeout ------------------
    print("\n[timeout]")
    # z chosen so the partner samples something (either goal); to force timeout,
    # place ego far from any goal and STAY every step.
    env_slow = make_env(partner_z=0.5, max_steps=6)
    _, s = env_slow.reset(key)
    done_all = False
    for t in range(env_slow.max_steps):
        _, s, r, d, _ = env_slow.step_env(key, s, {"agent_0": encode_ego(STAY, NONE)})
        done_all = bool(d["__all__"])
        if done_all and t < env_slow.max_steps - 1:
            # Partner might succeed by pure luck given close geometry; skip fail here.
            print(f"    (partner coincidentally arrived by t={t+1}; ignore for timeout test)")
            break
    else:
        check("timeout fires at max_steps", done_all)
        check("time == max_steps", int(s.time) == env_slow.max_steps)

    # ------------------ jit + vmap ------------------
    print("\n[jit + vmap]")
    jstep = jax.jit(env.step_env)
    _, s0 = env.reset(key)
    _, s_jit, r_jit, d_jit, _ = jstep(key, s0, {"agent_0": encode_ego(STAY, M0)})
    _, s_ref, r_ref, d_ref, _ = env.step_env(key, s0, {"agent_0": encode_ego(STAY, M0)})
    check("jit agent_pos matches", jnp.all(s_jit.agent_pos == s_ref.agent_pos))
    check("jit pending_message matches",
          int(s_jit.pending_message) == int(s_ref.pending_message))
    check("jit partner_goal matches",
          int(s_jit.partner_goal) == int(s_ref.partner_goal))
    check("jit reward matches", float(r_jit["agent_0"]) == float(r_ref["agent_0"]))
    check("jit done matches", bool(d_jit["__all__"]) == bool(d_ref["__all__"]))

    N = 4
    keys = jax.random.split(key, N)
    obs_v, state_v = jax.vmap(env.reset)(keys)
    check("vmap grid batch dim",
          obs_v["agent_0"]["grid"].shape == (N, env.height, env.width, 5))
    check("vmap last_message batch dim",
          obs_v["agent_0"]["last_message"].shape == (N, 3))
    check("vmap agent_pos batch dim", state_v.agent_pos.shape == (N, 2, 2))
    check("vmap partner_goal batch dim", state_v.partner_goal.shape == (N,))

    actions_v = {
        "agent_0": jnp.array(
            [encode_ego(STAY, M0), encode_ego(STAY, M1),
             encode_ego(STAY, NONE), encode_ego(STAY, M0)],
            dtype=jnp.int32,
        )
    }
    obs_v2, state_v2, r_v, d_v, _ = jax.vmap(env.step_env)(keys, state_v, actions_v)
    check("vmap step reward shape", r_v["agent_0"].shape == (N,))
    check("vmap step done shape", d_v["__all__"].shape == (N,))
    check("vmap step pending_message shape", state_v2.pending_message.shape == (N,))

    @jax.jit
    def rollout(k):
        obs, state = env.reset(k)

        def body(carry, _):
            state = carry
            _, state2, r, d, _ = env.step_env(
                k, state, {"agent_0": jnp.int32(encode_ego(STAY, NONE))}
            )
            return state2, (r["agent_0"], d["__all__"])

        state_end, (rs, ds) = jax.lax.scan(body, state, xs=None, length=env.max_steps)
        return state_end, rs, ds

    _end, rs, ds = jax.vmap(rollout)(keys)
    check("scanned rollout reward shape", rs.shape == (N, env.max_steps))
    check("scanned rollout done at end for all", bool(jnp.all(ds[:, -1])))

    # ------------------ D4 symmetry helpers (unit tests) ------------------
    print("\n[D4 helpers]")
    check("N_SYMMETRIES == 8", N_SYMMETRIES == 8)

    # (a) Position map: identity is identity; each symmetry is idempotent
    # only for specific elements — instead check that the group closes under
    # composition (applying g then g^{-1} recovers the point) and that all 8
    # transformations produce 8 distinct points from a generic start on 7x7.
    n = 7
    p = np.array([2, 5], dtype=np.int32)  # generic point
    seen = set()
    for g in range(N_SYMMETRIES):
        q = tuple(int(v) for v in _sym_position(g, p, n))
        seen.add(q)
        # in bounds
        check(f"_sym_position({SYMMETRY_NAMES[g]}) in bounds",
              0 <= q[0] < n and 0 <= q[1] < n)
    check("all 8 D4 images of a generic point are distinct on 7x7",
          len(seen) == 8)

    # (b) Each symmetry's inverse gives back the origin.
    # For D4, every element except rot90/rot270 is its own inverse.
    inverse = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
    p_start = np.array([1, 4], dtype=np.int32)
    for g, g_inv in inverse.items():
        q = _sym_position(g, p_start, n)
        pp = _sym_position(g_inv, q, n)
        check(f"{SYMMETRY_NAMES[g]} ∘ {SYMMETRY_NAMES[g_inv]} == identity",
              tuple(int(v) for v in pp) == tuple(int(v) for v in p_start))

    # (c) Wall array: transformation must preserve number of wall cells and
    # the position map must be consistent — walls at old positions become
    # walls at transformed positions.
    walls = env.wall_map.astype(bool)  # single-layout env
    walls_np = np.asarray(walls)
    n_walls = int(walls_np.sum())
    for g in range(N_SYMMETRIES):
        walls_new = _sym_wall(g, walls_np)
        check(f"{SYMMETRY_NAMES[g]}: wall count preserved",
              int(walls_new.sum()) == n_walls)
        # For every original wall (x, y), the transformed position must also
        # be a wall in walls_new.
        for yy in range(n):
            for xx in range(n):
                if walls_np[yy, xx]:
                    nx, ny = _sym_position(g, np.array([xx, yy]), n)
                    if not bool(walls_new[int(ny), int(nx)]):
                        check(
                            f"{SYMMETRY_NAMES[g]}: wall at ({xx},{yy}) maps to ({int(nx)},{int(ny)})",
                            False,
                        )

    # ------------------ D4 augmentation in the env ------------------
    print("\n[env with augment_symmetries=True]")
    import os as _os
    base_paths = sorted(
        [_os.path.join(_os.path.dirname(__file__), "grids", "layouts", "train",
                       f"train_{i:04d}.json")
         for i in range(0, 5)]  # 5 base layouts
    )
    env_aug = CoordinationGrid(
        layout_paths=base_paths, partner_z=1.0, max_steps=15,
        augment_symmetries=True,
    )
    check("n_base_layouts == 5", env_aug.n_base_layouts == 5)
    check("symmetries_per_layout == 8", env_aug.symmetries_per_layout == 8)
    check("n_layouts == 5 * 8 == 40", env_aug.n_layouts == 40)
    check("stacked wall_maps (40, 7, 7)", env_aug.wall_maps.shape == (40, 7, 7))

    # The identity variant (idx 0 of each base) must reproduce the original
    # geometry byte-for-byte.
    for b in range(env_aug.n_base_layouts):
        idx = b * 8  # identity is g=0 per _augment_layout order
        base = CoordinationGrid(layout_path=base_paths[b])
        check(f"base {b}: identity wall_map matches original",
              jnp.all(env_aug.wall_maps[idx] == base.wall_map))
        check(f"base {b}: identity ego_start matches original",
              jnp.all(env_aug.ego_starts[idx] == base.ego_start))
        check(f"base {b}: identity red_goal matches original",
              jnp.all(env_aug.red_goals[idx] == base.red_goal))

    # BFS distance from partner_start to red_goal should be invariant under
    # every symmetry of the same base layout — the maze got flipped/rotated
    # but the metric structure didn't change.
    def _bfs_dist(wall_np, sx, sy, gx, gy):
        H, W = wall_np.shape
        dist = np.full((H, W), -1, dtype=np.int32)
        if wall_np[gy, gx]:
            return -1
        dist[gy, gx] = 0
        from collections import deque as _deque
        q = _deque([(gx, gy)])
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < W and 0 <= ny < H and not wall_np[ny, nx]
                        and dist[ny, nx] < 0):
                    dist[ny, nx] = dist[cy, cx] + 1
                    q.append((nx, ny))
        return int(dist[sy, sx])

    for b in range(env_aug.n_base_layouts):
        # Reference distance from the identity variant.
        walls0 = np.asarray(env_aug.wall_maps[b * 8])
        p0 = np.asarray(env_aug.partner_starts[b * 8])
        r0 = np.asarray(env_aug.red_goals[b * 8])
        d0 = _bfs_dist(walls0, int(p0[0]), int(p0[1]), int(r0[0]), int(r0[1]))
        for g in range(1, 8):
            walls_g = np.asarray(env_aug.wall_maps[b * 8 + g])
            pg = np.asarray(env_aug.partner_starts[b * 8 + g])
            rg = np.asarray(env_aug.red_goals[b * 8 + g])
            dg = _bfs_dist(walls_g, int(pg[0]), int(pg[1]),
                           int(rg[0]), int(rg[1]))
            check(
                f"base {b} sym {SYMMETRY_NAMES[g]}: BFS(partner→RED) preserved "
                f"({d0} == {dg})",
                d0 == dg,
            )

    # Positions in the augmented layouts remain distinct and off walls.
    for i in range(env_aug.n_layouts):
        w = np.asarray(env_aug.wall_maps[i])
        e = np.asarray(env_aug.ego_starts[i])
        p = np.asarray(env_aug.partner_starts[i])
        r = np.asarray(env_aug.red_goals[i])
        b_ = np.asarray(env_aug.blue_goals[i])
        for name, xy in [("ego", e), ("partner", p),
                         ("red", r), ("blue", b_)]:
            check(f"{i}:{name} not on wall",
                  not bool(w[int(xy[1]), int(xy[0])]))
        check(f"{i}: 4 positions all distinct",
              len({tuple(e.tolist()), tuple(p.tolist()),
                   tuple(r.tolist()), tuple(b_.tolist())}) == 4)

    # env.reset actually samples across the full augmented pool.
    seen_idxs = set()
    for seed in range(256):
        _, s = env_aug.reset(jax.random.PRNGKey(seed))
        seen_idxs.add(int(s.layout_idx))
    check("reset samples ≥ 20 distinct augmented layouts over 256 seeds",
          len(seen_idxs) >= 20)
    check("reset never samples out-of-range layout_idx",
          all(0 <= i < env_aug.n_layouts for i in seen_idxs))

    # Under z=1 + M0, partner commits RED on every augmented variant too.
    for i in range(env_aug.n_layouts):
        _, s = env_aug.reset_to_layout(jax.random.PRNGKey(i), i)
        _, s, _, _, _ = env_aug.step_env(
            jax.random.PRNGKey(0), s, {"agent_0": encode_ego(STAY, M0)}
        )
        _, s, _, _, _ = env_aug.step_env(
            jax.random.PRNGKey(0), s, {"agent_0": encode_ego(STAY, NONE)}
        )
        check(f"augmented layout {i}: z=1+M0 commits RED",
              int(s.partner_goal) == GOAL_RED)

    # augment_symmetries=False stays untouched.
    env_noaug = CoordinationGrid(
        layout_paths=base_paths, partner_z=1.0, max_steps=15,
        augment_symmetries=False,
    )
    check("augment_symmetries=False: n_layouts == 5 (no expansion)",
          env_noaug.n_layouts == 5)

    # ------------------ multi-layout ------------------
    print("\n[multi-layout: layout pool of 5]")
    import os
    layouts_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "grids", "layouts", "train"
    ))
    paths = sorted(
        [os.path.join(layouts_dir, f"train_{i:04d}.json") for i in range(2, 7)]
    )
    env_multi = CoordinationGrid(layout_paths=paths, partner_z=1.0, max_steps=15)
    check("n_layouts == 5", env_multi.n_layouts == 5)
    check("wall_maps stacked (5,H,W)", env_multi.wall_maps.shape == (5, 7, 7))
    check("ego_starts stacked (5,2)", env_multi.ego_starts.shape == (5, 2))
    check(
        "next_action_toward_red stacked (5,H,W)",
        env_multi.next_action_toward_red.shape == (5, 7, 7),
    )

    # reset samples a valid layout_idx
    for seed in range(6):
        _, s = env_multi.reset(jax.random.PRNGKey(seed))
        idx = int(s.layout_idx)
        check(f"seed={seed} layout_idx in [0,5)", 0 <= idx < 5)
        # ego/partner start match the sampled slice
        check(
            f"seed={seed} ego@sampled ego_start",
            jnp.all(s.agent_pos[0] == env_multi.ego_starts[idx]),
        )
        check(
            f"seed={seed} partner@sampled partner_start",
            jnp.all(s.agent_pos[1] == env_multi.partner_starts[idx]),
        )
        check(
            f"seed={seed} state.wall_map matches sampled slice",
            jnp.all(s.wall_map == env_multi.wall_maps[idx]),
        )

    # Distinct layouts must actually get sampled across many seeds.
    seen = set()
    for seed in range(64):
        _, s = env_multi.reset(jax.random.PRNGKey(seed))
        seen.add(int(s.layout_idx))
    check("sees multiple distinct layouts across 64 seeds", len(seen) >= 3)

    # reset_to_layout is deterministic
    for target_idx in range(5):
        _, s = env_multi.reset_to_layout(jax.random.PRNGKey(0), target_idx)
        check(f"reset_to_layout({target_idx}) sets layout_idx",
              int(s.layout_idx) == target_idx)

    # Per-layout goal commit: z=1 + M0 at t=0 must commit partner_goal to
    # RED on every layout, and state.red_goal must be the sampled layout's
    # RED coords (i.e. the state uses the right slice, not layout 0's).
    #
    # We don't require the partner to actually REACH RED because on some
    # layouts the greedy BFS path is blocked by a stationary ego (that's an
    # env-collision property, not a multi-layout property, and is tested
    # separately by the collision suite).
    for target_idx in range(5):
        env_z1 = env_multi
        _, s = env_z1.reset_to_layout(jax.random.PRNGKey(target_idx), target_idx)
        # After the t=0 step, partner_goal commits at the NEXT step (t=1);
        # advance one more env call so we can observe the commit.
        _, s, _, _, _ = env_z1.step_env(
            jax.random.PRNGKey(0), s, {"agent_0": encode_ego(STAY, M0)}
        )
        _, s, _, _, _ = env_z1.step_env(
            jax.random.PRNGKey(0), s, {"agent_0": encode_ego(STAY, NONE)}
        )
        check(
            f"layout {target_idx}: partner committed RED",
            int(s.partner_goal) == GOAL_RED,
        )
        check(
            f"layout {target_idx}: state.red_goal matches layout slice",
            jnp.all(s.red_goal == env_multi.red_goals[target_idx]),
        )
        check(
            f"layout {target_idx}: state.blue_goal matches layout slice",
            jnp.all(s.blue_goal == env_multi.blue_goals[target_idx]),
        )

    # jit + vmap over layouts still work with the new stacked arrays.
    keys_ml = jax.random.split(jax.random.PRNGKey(0), 8)
    obs_v, state_v = jax.vmap(env_multi.reset)(keys_ml)
    check("vmap reset picks per-batch layout_idx",
          state_v.layout_idx.shape == (8,))
    check("vmap reset wall_map slice per batch",
          state_v.wall_map.shape == (8, 7, 7))
    step_v = jax.jit(jax.vmap(env_multi.step_env,
                              in_axes=(0, 0, {"agent_0": 0})))
    act_v = {"agent_0": jnp.array([encode_ego(STAY, M0)] * 8, dtype=jnp.int32)}
    obs_v2, state_v2, r_v, d_v, _ = step_v(keys_ml, state_v, act_v)
    check("vmap step preserves per-batch layout_idx",
          jnp.all(state_v2.layout_idx == state_v.layout_idx))

    # ------------------ action-legality masks + masked policy ---------------
    print("\n[action legality masks]")
    check("LEGAL_ACTION_IDS_T0 == (12, 13, 14)",
          tuple(LEGAL_ACTION_IDS_T0) == (12, 13, 14))
    check("LEGAL_ACTION_IDS_TGEQ1 == (0, 3, 6, 9, 12)",
          tuple(LEGAL_ACTION_IDS_TGEQ1) == (0, 3, 6, 9, 12))
    check("ACTION_MASK_T0 has 3 True entries", int(ACTION_MASK_T0.sum()) == 3)
    check("ACTION_MASK_TGEQ1 has 5 True entries",
          int(ACTION_MASK_TGEQ1.sum()) == 5)
    check("intersection of masks == {12} (STAY+NONE)",
          set(LEGAL_ACTION_IDS_T0) & set(LEGAL_ACTION_IDS_TGEQ1) == {12})

    print("\n[masked ActorCriticCommRNN samples only legal actions]")
    # Import the network here so plain env tests can still run without the
    # trainer file being importable (e.g. no PPO deps installed).
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]
                             / "baselines" / "IPPO"))
    from ippo_rnn_coordination_grid import ActorCriticCommRNN, ScannedRNN

    single_env = CoordinationGrid(partner_z=1.0)
    N = 128
    cfg = {"GRU_HIDDEN_DIM": 32, "FC_DIM_SIZE": 32,
           "GRID_EMB_DIM": 16, "MSG_EMB_DIM": 4, "ACTIVATION": "relu"}
    net = ActorCriticCommRNN(action_dim=single_env.n_ego_actions, config=cfg)
    hstate = ScannedRNN.initialize_carry(N, cfg["GRU_HIDDEN_DIM"])
    def _mk_obs(is_t0_val):
        return {
            "grid": jnp.zeros((1, N, single_env.height, single_env.width, 5),
                              dtype=jnp.float32),
            "last_message": jnp.zeros((1, N, 3), dtype=jnp.float32),
            "is_t0": jnp.full((1, N), float(is_t0_val), dtype=jnp.float32),
        }
    dones = jnp.zeros((1, N), dtype=bool)
    params = net.init(jax.random.PRNGKey(0), hstate,
                      (_mk_obs(1.0), dones))

    # Sample at t=0: all samples must be in the t=0 legal set.
    _, pi_t0, _ = net.apply(params, hstate, (_mk_obs(1.0), dones))
    a_t0 = np.asarray(pi_t0.sample(seed=jax.random.PRNGKey(1)))   # (1, N)
    check("all t=0 samples in {12,13,14}",
          set(int(x) for x in a_t0.reshape(-1).tolist())
          <= set(LEGAL_ACTION_IDS_T0))
    # log_prob is finite for legal actions
    lp_t0 = np.asarray(pi_t0.log_prob(jnp.asarray(a_t0)))
    check("t=0 log_prob finite for sampled actions",
          bool(np.isfinite(lp_t0).all()))
    # Distribution over the 15 slots must place 0 mass on illegal ids.
    probs_t0 = np.asarray(pi_t0.probs)                           # (1, N, 15)
    ill_t0 = np.array([i for i in range(15)
                       if i not in LEGAL_ACTION_IDS_T0])
    check("t=0 probs sum to ~1", np.allclose(probs_t0.sum(-1), 1.0, atol=1e-5))
    check("t=0 probs on illegal ids == 0",
          float(probs_t0[..., ill_t0].max()) < 1e-6)

    # Sample at t>=1: all samples must be in the t>=1 legal set.
    _, pi_tg1, _ = net.apply(params, hstate, (_mk_obs(0.0), dones))
    a_tg1 = np.asarray(pi_tg1.sample(seed=jax.random.PRNGKey(2)))
    check("all t>=1 samples in {0,3,6,9,12}",
          set(int(x) for x in a_tg1.reshape(-1).tolist())
          <= set(LEGAL_ACTION_IDS_TGEQ1))
    probs_tg1 = np.asarray(pi_tg1.probs)
    ill_tg1 = np.array([i for i in range(15)
                        if i not in LEGAL_ACTION_IDS_TGEQ1])
    check("t>=1 probs sum to ~1",
          np.allclose(probs_tg1.sum(-1), 1.0, atol=1e-5))
    check("t>=1 probs on illegal ids == 0",
          float(probs_tg1[..., ill_tg1].max()) < 1e-6)

    # Entropy caps: t=0 entropy ≤ ln 3, t>=1 entropy ≤ ln 5.
    ent_t0 = float(np.asarray(pi_t0.entropy()).mean())
    ent_tg1 = float(np.asarray(pi_tg1.entropy()).mean())
    check(f"t=0 entropy ≤ ln 3 ({ent_t0:.3f} ≤ {np.log(3):.3f})",
          ent_t0 <= np.log(3) + 1e-4)
    check(f"t>=1 entropy ≤ ln 5 ({ent_tg1:.3f} ≤ {np.log(5):.3f})",
          ent_tg1 <= np.log(5) + 1e-4)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
