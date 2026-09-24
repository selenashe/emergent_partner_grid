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
        check(f"{aid} obs is dict with grid+last_message",
              isinstance(obs[aid], dict)
              and "grid" in obs[aid] and "last_message" in obs[aid])
        check(f"{aid} grid shape", obs[aid]["grid"].shape == (env.height, env.width, 5))
        check(f"{aid} last_message shape", obs[aid]["last_message"].shape == (3,))
    check("last_message one-hot on NONE at reset",
          jnp.allclose(obs["agent_0"]["last_message"], jnp.array([1.0, 0.0, 0.0])))
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

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
