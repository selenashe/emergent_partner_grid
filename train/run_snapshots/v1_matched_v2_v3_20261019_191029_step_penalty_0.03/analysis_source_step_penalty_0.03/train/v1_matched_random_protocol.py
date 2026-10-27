"""Randomize only the no-influence allocation in frozen matched V1_matched."""

ENVIRONMENT_FILE = "jaxmarl/environments/coordination_grid/coordination_grid.py"


def randomize_noinfluence(source):
    """Draw a fair assignment at t=0, retaining V1_matched's commit and observation rules.

    The step already splits off an unused key. Using it leaves every other
    random stream and the influence-enabled transition unchanged. The sampled
    allocation is written only at t=0 and retained for the entire round.
    """
    old = """        forced_alloc = jnp.where(
            (state.round_idx % jnp.int32(2)) == jnp.int32(0),
            jnp.int32(Allocations.red),
            jnp.int32(Allocations.blue),
        )"""
    new = """        # Sample independently of capability, round parity, and ego action.
        # Only the t=0 write below consumes this assignment for the round.
        forced_alloc = jax.random.randint(_k_unused, (), 1, 3, dtype=jnp.int32)"""
    if source.count(old) != 1 or source.count("key, _k_unused = jax.random.split(key, 2)") != 1:
        raise ValueError("Expected the unmodified frozen V1_matched no-influence assignment")
    source = source.replace(old, new, 1)
    source = source.replace("There is no other stochastic goal-commit rule.",
                            "Without influence, a fresh uniform random assignment is used each round.")
    source = source.replace("partner-independent balanced\nrule (round_idx parity)",
                            "partner-independent uniform random\nrule (a fresh RED/BLUE draw per round)")
    source = source.replace("round-parity forced alloc", "per-round uniform random alloc")
    return source
