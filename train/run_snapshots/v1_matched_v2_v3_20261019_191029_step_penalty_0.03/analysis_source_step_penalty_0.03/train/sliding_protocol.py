"""Minimal source transforms for sliding initial allocation on frozen sliding irreversible/v3.

The frozen baselines remain authoritative. At t=0 every transform is the
identity. At t>0 the ego observes t ordinary transitions while forced to
STAY/NONE; the partner follows a provisional goal from transition zero.
Influence-enabled variants use the same round-parity provisional assignment.
No-influence retains sliding_irreversible_baseline's parity rule and v3's original random draw. At the
initial decision both agents STAY and partner cooldown becomes zero, as in
both baseline initial decision transitions. irreversible baseline then locks allocation; v3
retains its existing reassignment and destination-delay mechanics.

The observation/action dimensions, reward, horizon, reset key splitting,
layouts, capability selection, network layers, and PPO remain unchanged.
``is_t0`` is reused as a phase flag: -1 during observation, 1 at commitment,
0 during navigation. The one necessary sliding_irreversible_baseline trainer change delegates legality
masking to the environment; its previous binary mask cannot represent the
forced observation phase. The v3 trainer and both evaluators remain unchanged.
"""


def _validate(t, reversible):
    if isinstance(t, bool) or not isinstance(t, int) or t < 0:
        raise ValueError("precommit observation steps must be a nonnegative integer")
    if not isinstance(reversible, bool):
        raise ValueError("reversible must be a bool")


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Expected exactly one frozen source anchor: " + old)
    return source.replace(old, new)


def _replace_section(source, start, end, new):
    if source.count(start) != 1 or end not in source:
        raise ValueError("Expected unique frozen section anchors: " + start + ", " + end)
    left = source.index(start)
    right = source.index(end, left)
    return source[:left] + new + source[right:]


_MASK = '''def ego_action_mask(obs):
    """Legality over existing input shape; phase -1 observes, 1 commits."""
    phase = obs["is_t0"]
    observing_mask = jnp.arange(N_EGO_ACTIONS) == 3 * int(Actions.stay)
    initial_mask = jnp.asarray(ACTION_MASK_T0)
{forced}    return jnp.where((phase < -0.5)[..., None], observing_mask,
                     jnp.where((phase > 0.5)[..., None], initial_mask,
                               jnp.asarray(ACTION_MASK_TGEQ1)))


'''

_FORCED_V3 = '''    observed_alloc = jnp.argmax(obs["last_allocation"], axis=-1)
    forced_action = 3 * int(Actions.stay) + jnp.maximum(1, observed_alloc)
    forced_mask = jnp.arange(N_EGO_ACTIONS) == forced_action[..., None]
    initial_mask = jnp.where((observed_alloc == int(Allocations.none))[..., None],
                             initial_mask, forced_mask)
'''


def transform_environment(source, t, reversible):
    """Transform the corresponding frozen baseline, failing on anchor drift."""
    _validate(t, reversible)
    if t == 0:
        return source
    actual_reversible = 'ALLOCATION_PROTOCOL = "online_v3"' in source
    if actual_reversible != reversible:
        raise ValueError("Reversibility must select the corresponding frozen sliding irreversible/v3 baseline")
    constants = (f"PRECOMMIT_OBSERVATION_STEPS = {t}\n"
                 f"ALLOCATION_REVERSIBLE = {reversible}\n")
    source = _replace(source, "LEGAL_ACTION_IDS_T0 = tuple(",
                      constants + "LEGAL_ACTION_IDS_T0 = tuple(")
    if reversible:
        source = _replace_section(source, "def ego_action_mask(obs):", "# ---------------------------------------------------------------------------\n# Communication condition",
                                  _MASK.format(forced=_FORCED_V3))
    else:
        source = _replace(source, "# ---------------------------------------------------------------------------\n# Communication condition",
                          _MASK.format(forced="") + "# ---------------------------------------------------------------------------\n# Communication condition")
    source = _replace(source, "        influence: bool = True,",
                      "        influence: bool = True,\n"
                      "        precommit_observation_steps: int = PRECOMMIT_OBSERVATION_STEPS,\n"
                      "        allocation_reversible: bool = ALLOCATION_REVERSIBLE,")
    source = _replace(source, "        self.influence: bool = bool(influence)",
                      "        self.influence: bool = bool(influence)\n"
                      "        if (isinstance(precommit_observation_steps, bool)\n"
                      "                or precommit_observation_steps != PRECOMMIT_OBSERVATION_STEPS):\n"
                      "            raise ValueError(\"Observation time must match this frozen sliding variant\")\n"
                      "        if not isinstance(allocation_reversible, bool) or allocation_reversible != ALLOCATION_REVERSIBLE:\n"
                      "            raise ValueError(\"Reversibility must match this frozen sliding variant\")\n"
                      "        if precommit_observation_steps >= self.max_steps:\n"
                      "            raise ValueError(\"Observation time must leave an allocation transition in the round\")\n"
                      "        self.precommit_observation_steps = int(precommit_observation_steps)\n"
                      "        self.allocation_reversible = allocation_reversible")
    if reversible:
        source = _replace(source,
                          '        initial_alloc = jax.random.randint(key, (), 1, 3, dtype=jnp.int32)\n'
                          '        # Retain the v2 random draw/key stream; influence-enabled v3 starts unset.\n'
                          '        if self.choose_initial_allocation:\n'
                          '            initial_alloc = jnp.int32(Allocations.none)\n'
                          '        initial_goal = jnp.where(initial_alloc == int(Allocations.red),\n'
                          '                                 jnp.int32(GOAL_BLUE), jnp.int32(GOAL_RED))\n'
                          '        if self.choose_initial_allocation:\n'
                          '            initial_goal = jnp.int32(GOAL_UNSET)',
                          '        # Keep the exact baseline draw/key stream for no-influence.\n'
                          '        initial_alloc = jax.random.randint(key, (), 1, 3, dtype=jnp.int32)\n'
                          '        if self.influence:\n'
                          '            initial_alloc = jnp.where(round_val % 2 == 0,\n'
                          '                                      jnp.int32(Allocations.red), jnp.int32(Allocations.blue))\n'
                          '        initial_goal = jnp.where(initial_alloc == int(Allocations.red),\n'
                          '                                 jnp.int32(GOAL_BLUE), jnp.int32(GOAL_RED))')
        source = _replace(source, "            last_ego_allocation=initial_alloc,",
                          "            last_ego_allocation=jnp.int32(Allocations.none),")
    else:
        source = _replace(source, "        return State(\n            agent_pos=agent_pos,",
                          "        # Provisional parity goal exposes real partner movement without a policy commitment.\n"
                          "        initial_alloc = jnp.where(round_val % 2 == 0,\n"
                          "                                  jnp.int32(Allocations.red), jnp.int32(Allocations.blue))\n"
                          "        initial_goal = jnp.where(initial_alloc == int(Allocations.red),\n"
                          "                                 jnp.int32(GOAL_BLUE), jnp.int32(GOAL_RED))\n"
                          "        return State(\n            agent_pos=agent_pos,")
        source = _replace(source, "            partner_goal=jnp.int32(GOAL_UNSET),", "            partner_goal=initial_goal,")
        source = _replace(source, "            partner_assignment=jnp.int32(Allocations.none),", "            partner_assignment=initial_alloc,")
    source = _replace(source, "        is_time0 = state.time == 0",
                      "        is_time0 = state.time == 0\n"
                      "        is_observing = state.time < self.precommit_observation_steps\n"
                      "        is_commit = state.time == self.precommit_observation_steps")
    allocation_core = '''        # Allocation is unset throughout observation; commit consumes an ordinary transition.
        valid_alloc = (ego_alloc == int(Allocations.red)) | (ego_alloc == int(Allocations.blue))
        fallback_alloc = jnp.maximum(jnp.int32(Allocations.red), state.last_ego_allocation)
        ego_alloc = jnp.where(valid_alloc, ego_alloc, fallback_alloc)
{request}        partner_assignment = jnp.where(
            {write} & jnp.bool_(self.influence), ego_alloc, state.partner_assignment,
        )
        new_partner_goal = jnp.where(partner_assignment == int(Allocations.red),
                                    jnp.int32(GOAL_BLUE), jnp.int32(GOAL_RED))
{changed}
'''
    if reversible:
        core = allocation_core.format(
            request='        ego_alloc = jnp.where(is_commit & jnp.bool_(not self.choose_initial_allocation),\n'
                    '                                  state.partner_assignment, ego_alloc)\n'
                    '        ego_alloc = jnp.where(is_observing, jnp.int32(Allocations.none), ego_alloc)\n',
            write='(~is_observing)',
            changed='        # The initial choice has zero cooldown; only subsequent switches incur v3 costs.\n'
                    '        assignment_changed = (state.time > self.precommit_observation_steps) & (new_partner_goal != state.partner_goal)\n')
        source = _replace_section(source, "        # V3 influence chooses at t=0;", "        # ------- Delay mechanic", core)
        source = _replace(source, "        move_ctr = jnp.where(assignment_changed, d_for_goal, state.partner_move_ctr)",
                          "        initial_move_ctr = jnp.where(is_commit, jnp.int32(0), state.partner_move_ctr)\n"
                          "        move_ctr = jnp.where(assignment_changed, d_for_goal, initial_move_ctr)")
        source = _replace(source, "        can_move_now = (move_ctr == 0) & partner_has_goal & (~is_time0)",
                          "        can_move_now = (move_ctr == 0) & partner_has_goal & (~is_commit)")
        source = _replace(source, '            "allocation_decision": ~is_time0,',
                          '            "allocation_decision": state.time > self.precommit_observation_steps,')
        source = _replace(source, '            "ego_request_changed": (~is_time0) & (ego_alloc != state.last_ego_allocation),',
                          '            "ego_request_changed": (state.time > self.precommit_observation_steps) & (ego_alloc != state.last_ego_allocation),')
        source = _replace(source, "            state.last_ego_allocation, N_ALLOCATIONS",
                          "            jnp.where((state.time == self.precommit_observation_steps) & jnp.bool_(not self.influence),\n"
                          "                      state.partner_assignment, state.last_ego_allocation), N_ALLOCATIONS")
    else:
        core = allocation_core.format(
            request='        ego_alloc = jnp.where(is_commit, ego_alloc, jnp.int32(Allocations.none))\n',
            write='is_commit', changed='')
        source = _replace_section(source, "        # ------- Partner goal commitment", "        # ------- Delay mechanic", core)
        source = _replace(source, "        can_move_now = (state.partner_move_ctr == 0) & partner_has_goal & (~is_time0)",
                          "        move_ctr = jnp.where(is_commit, jnp.int32(0), state.partner_move_ctr)\n"
                          "        can_move_now = (move_ctr == 0) & partner_has_goal & (~is_commit)")
        source = _replace(source, "            jnp.maximum(state.partner_move_ctr - jnp.int32(1), jnp.int32(0)),",
                          "            jnp.maximum(move_ctr - jnp.int32(1), jnp.int32(0)),")
        source = _replace_section(source, "        # ------- Allocation write -------", "        step_state = state.replace(",
                                  "        # The initial allocation is locked after this round's commitment.\n"
                                  "        partner_assignment_written = partner_assignment\n"
                                  "        last_ego_allocation_written = ego_alloc\n\n")
    source = _replace(source, "            is_time0, jnp.int32(Actions.stay), ego_move",
                      "            is_observing | is_commit, jnp.int32(Actions.stay), ego_move")
    source = _replace(source, "        is_t0 = (state.time == 0).astype(jnp.float32)",
                      "        # Reuse the scalar flag without adding any input dimensions.\n"
                      "        is_t0 = jnp.where(state.time < self.precommit_observation_steps,\n"
                      "                          jnp.float32(-1),\n"
                      "                          (state.time == self.precommit_observation_steps).astype(jnp.float32))")
    return source


def transform_trainer(source, t, reversible):
    """Replace only sliding_irreversible_baseline RNN/MLP legality computations; t=0/v3 are identities."""
    _validate(t, reversible)
    if t == 0 or reversible:
        return source
    source = _replace(source, "    ACTION_MASK_TGEQ1,", "    ACTION_MASK_TGEQ1,\n    ego_action_mask,")
    source = _replace_section(source, "        # Legality mask. t=0 ->", "        logits = jnp.where(legal, logits,",
                              "        # Sliding observation uses the environment's phase-specific legality.\n"
                              "        legal = ego_action_mask(obs)\n")
    source = _replace_section(source, '        is_t0 = obs["is_t0"]\n', "        logits = jnp.where(legal, logits,",
                              "        legal = ego_action_mask(obs)\n")
    return source


def transform_exports(source, t, reversible):
    """Expose the sliding_irreversible_baseline mask to its unchanged-shape policy; other cases are identities."""
    _validate(t, reversible)
    if t == 0 or reversible:
        return source
    return _replace(source, "    ACTION_MASK_TGEQ1,", "    ACTION_MASK_TGEQ1,\n    ego_action_mask,")


upgrade_environment = transform_environment
