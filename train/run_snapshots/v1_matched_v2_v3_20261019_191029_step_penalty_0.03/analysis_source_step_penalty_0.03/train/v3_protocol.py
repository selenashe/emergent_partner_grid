"""Apply v3 initialization to the executed v2 environment, preserving its dynamics.

The same transformation is used for the active environment and frozen runs.
Frozen v3 changes only this environment file; its v2 trainer and evaluator are
copied byte for byte. V2 remains selectable in the active environment.
"""


def upgrade_environment(source, default_protocol="online_v3"):
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError("Expected exactly one v2 source anchor: " + old)
        source = source.replace(old, new)

    replace('ALLOCATION_PROTOCOL = "online_v2"',
            f'ALLOCATION_PROTOCOL = "{default_protocol}"')
    replace('        influence: bool = True,',
            '        influence: bool = True,\n'
            '        allocation_protocol: str = ALLOCATION_PROTOCOL,')
    replace('        self.influence: bool = bool(influence)',
            '        self.influence: bool = bool(influence)\n'
            '        if allocation_protocol not in ("online_v2", "online_v3"):\n'
            '            raise ValueError(f"Unsupported allocation protocol: {allocation_protocol}")\n'
            '        self.allocation_protocol = allocation_protocol\n'
            '        self.choose_initial_allocation = allocation_protocol == "online_v3" and self.influence')
    replace('    initial_alloc = jnp.maximum(1, jnp.argmax(obs["last_allocation"], axis=-1))',
            '    observed_alloc = jnp.argmax(obs["last_allocation"], axis=-1)\n'
            '    initial_alloc = jnp.maximum(1, observed_alloc)')
    replace('    return jnp.where((obs["is_t0"] > 0.5)[..., None], forced_mask,',
            '    # NONE at initialization denotes v3 with influence: both choices are legal.\n'
            '    # A supplied RED/BLUE assignment keeps v2 and no-influence actions forced.\n'
            '    initial_mask = jnp.where((observed_alloc == int(Allocations.none))[..., None],\n'
            '                             jnp.asarray(ACTION_MASK_T0), forced_mask)\n'
            '    return jnp.where((obs["is_t0"] > 0.5)[..., None], initial_mask,')
    replace('        initial_alloc = jax.random.randint(key, (), 1, 3, dtype=jnp.int32)',
            '        initial_alloc = jax.random.randint(key, (), 1, 3, dtype=jnp.int32)\n'
            '        # Retain the v2 random draw/key stream; influence-enabled v3 starts unset.\n'
            '        if self.choose_initial_allocation:\n'
            '            initial_alloc = jnp.int32(Allocations.none)')
    replace('        return State(\n            agent_pos=agent_pos,',
            '        if self.choose_initial_allocation:\n'
            '            initial_goal = jnp.int32(GOAL_UNSET)\n'
            '        return State(\n            agent_pos=agent_pos,')
    replace('        ego_alloc = jnp.where(is_time0 | (~valid_alloc), state.last_ego_allocation, ego_alloc)',
            '        forced_initial = is_time0 & jnp.bool_(not self.choose_initial_allocation)\n'
            '        # A direct invalid NONE request falls back to RED if no assignment exists.\n'
            '        fallback_alloc = jnp.maximum(jnp.int32(Allocations.red), state.last_ego_allocation)\n'
            '        ego_alloc = jnp.where(forced_initial | (~valid_alloc), fallback_alloc, ego_alloc)')
    replace('            (~is_time0) & jnp.bool_(self.influence), ego_alloc, state.partner_assignment,',
            '            ((~is_time0) | jnp.bool_(self.choose_initial_allocation)) & jnp.bool_(self.influence),\n'
            '            ego_alloc, state.partner_assignment,')
    replace('At reset, a uniform random RED/BLUE allocation is supplied to the ego.\n'
            'At t=0 both agents STAY; only STAY + that supplied allocation is legal.',
            'At t=0 both agents STAY. V2 supplies a uniform random RED/BLUE assignment.\n'
            'V3 with influence starts unset and lets the ego choose STAY + RED or BLUE.\n'
            'Without influence, both versions force the supplied random assignment.')
    replace('    """JAX-compatible legality mask, including the forced t=0 default.\n\n'
            '    Accepts any leading batch/time dimensions. At t=0 the policy has\n'
            '    probability one on the supplied assignment, so it cannot choose it.',
            '    """JAX-compatible mask for v2/v3 with arbitrary batch/time dimensions.\n\n'
            '    An unset v3 initialization permits RED/BLUE; supplied assignments\n'
            '    remain forced. After initialization the v2 movement mask is unchanged.')
    replace('        # t=0 cannot override the supplied default, even for direct API calls.',
            '        # V3 influence chooses at t=0; v2/no-influence keep the supplied default.')
    return source
