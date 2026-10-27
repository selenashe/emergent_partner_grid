"""Recover early positions by replaying the exact frozen evaluation batch.

Replay is accepted episode by episode only after checking all originally saved
behavior and hidden states. No policy fitting or replacement evaluation data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

import full_capability_evaluation as full

N_STEPS = 81


def replay_slice(evaluator, record, source, evaluation_seed, anchor_hidden=False):
    jax=evaluator.jax;jnp=evaluator.jnp
    config=full.read(record["config"])
    with h5py.File(source,"r") as handle:
        pool=handle["capability_pool"][:]
        n_repetitions=handle["hidden_state"].shape[0]//len(pool)
        reference_hidden=jnp.asarray(handle["hidden_state"][:,:N_STEPS].swapaxes(0,1)) if anchor_hidden else None
    kwargs=dict(config["ENV_KWARGS"],layouts_dir=record["layouts_dir"],augment_symmetries=False,
                partner_capability_pairs=pool.tolist(),influence=bool(config["INFLUENCE"]))
    kwargs.pop("layout_path",None);kwargs.pop("layout_paths",None)
    env=evaluator.CoordinationGrid(**kwargs)
    network,init_hidden,_=evaluator.trainer_mod.build_network(config,env.n_ego_actions)
    params=evaluator.load_params(record["checkpoint"])
    B=len(pool)*n_repetitions
    cap_indices=jnp.repeat(jnp.arange(len(pool),dtype=jnp.int32),n_repetitions)
    key_reset,key_step=jax.random.split(jax.random.PRNGKey(evaluation_seed))
    obs,states=jax.vmap(env.reset)(jax.random.split(key_reset,B))
    states=states.replace(capability=env.partner_capability_pairs[cap_indices])
    obs=jax.vmap(env.get_obs)(states)

    def body(carry,t):
        obs,states,hidden,done_prev,key=carry
        obs_a=obs["agent_0"]
        network_obs=jax.tree_util.tree_map(lambda x:x[jnp.newaxis,:],obs_a)
        new_hidden,pi,_=network.apply(params,hidden,(network_obs,done_prev[jnp.newaxis,:]))
        if anchor_hidden:
            # The saved post-observation state anchors the actual actor input.
            # Separately computed one-step GRU states validate reconstructed
            # observations without accumulating CPU/GPU recurrent roundoff.
            reference=reference_hidden[t]
            def intercept(next_function,args,kwargs,context):
                if isinstance(context.module,evaluator.trainer_mod.ScannedRNN) and context.method_name=="__call__":
                    return reference,reference
                return next_function(*args,**kwargs)
            with evaluator.trainer_mod.nn.intercept_methods(intercept):
                _,pi,_=network.apply(params,hidden,(network_obs,done_prev[jnp.newaxis,:]))
        key,ka,ks=jax.random.split(key,3)
        action=pi.sample(seed=ka).squeeze(0)
        obs2,states2,reward,done,info=jax.vmap(env.step_env,in_axes=(0,0,{"agent_0":0}))(
            jax.random.split(ks,B),states,{"agent_0":action})
        saved=dict(rewards=reward["agent_0"],dones=done["__all__"],success=info["success"],
            round_done=info["round_done"],round_idx=info["round_idx"],layout_idx=info["layout_idx"],
            capability=info["capability"],partner_assignment=info["partner_assignment"],
            ego_alloc_action=info["ego_alloc_action"],partner_goal=info["partner_goal"],round_time=info["round_time"],
            is_t0=obs_a["is_t0"]>.5,hidden_state=new_hidden)
        extra=dict(pre_positions=states.agent_pos,post_positions=states2.agent_pos,
            pre_clock=states.partner_move_ctr,post_clock=states2.partner_move_ctr,
            pre_time=states.time,pre_round=states.round_idx,pre_layout=states.layout_idx,
            red_goal=states.red_goal,blue_goal=states.blue_goal,ego_action=action)
        carry_hidden=reference_hidden[t] if anchor_hidden else new_hidden
        return (obs2,states2,carry_hidden,done["__all__"],key),(saved,extra)

    (_,_,_,_,_),(saved,extra)=jax.lax.scan(body,(obs,states,init_hidden(B),jnp.zeros(B,bool),key_step),jnp.arange(N_STEPS))
    saved={k:np.asarray(v).swapaxes(0,1) for k,v in saved.items()}
    extra={k:np.asarray(v).swapaxes(0,1) for k,v in extra.items()}
    valid=np.ones((B,N_STEPS),bool);max_error=0.;per_field={};debug={}
    with h5py.File(source,"r") as handle:
        for key,value in saved.items():
            prior=handle[key][:,:N_STEPS]
            axes=tuple(range(2,value.ndim))
            if key=="hidden_state":
                error=np.max(np.abs(value-prior),axis=-1)
                match=np.all(np.isclose(value,prior,atol=2e-3 if anchor_hidden else 2e-5,rtol=1e-4),axis=-1)
                max_error=max(max_error,float(error.max()))
                debug=dict(hidden_max_error_by_step=error.max(axis=0),hidden_median_error_by_step=np.median(error,axis=0),
                    debug_replayed_hidden=value[:4,:12],debug_saved_hidden=prior[:4,:12])
            elif np.issubdtype(value.dtype,np.floating):
                match=np.isclose(value,prior,atol=1e-6,rtol=1e-5)
                if axes:match=match.all(axis=axes)
            else:
                match=value==prior
                if axes:match=match.all(axis=axes)
            valid &= match
            per_field[key]=int((~match).sum())
    valid=np.logical_and.accumulate(valid,axis=1)
    repetitions=np.tile(np.arange(n_repetitions),len(pool))
    selected=np.isin(repetitions,[1,9,14,15])
    # Only held-out probe episodes enter all subsequent behavior summaries.
    output={k:v[selected] for k,v in extra.items()}
    output.update({k:v[selected] for k,v in saved.items() if k!="hidden_state"})
    output.update(pairs=np.repeat(pool,n_repetitions,axis=0)[selected],repetitions=repetitions[selected],
        verified_prefix=valid[selected],wall_maps=np.asarray(env.wall_maps))
    output.update(debug)
    audit=dict(source=str(source),batch_size=B,steps=N_STEPS,field_mismatches=per_field,
        anchored_on_saved_hidden_states=anchor_hidden,hidden_absolute_tolerance=.002 if anchor_hidden else .00002,
        max_hidden_error=max_error,test_episodes=int(selected.sum()),
        verified_test_episodes_all_steps=int(valid[selected].all(axis=1).sum()),
        verified_test_episodes_first30=int(valid[selected,:31].all(axis=1).sum()))
    return output,audit


def worker(path,index,out,only=None,anchor_hidden=False):
    manifest=full.read(path);record=manifest["policies"][index]
    full.verify(full.read(manifest["input_manifest"]),record)
    evaluator=full.load_evaluator(record)
    output=Path(out)/record["version"]/f"{record['condition']}_seed{record['seed']}"
    output.mkdir(parents=True,exist_ok=True)
    audits=[]
    paths=list(record["original_rollouts"].items())+[("additional",str(Path(record["directory"])/"additional_profiles.h5"))]
    for label,source in paths:
        if only is not None and label!=only:continue
        target=output/(label+"_replay.npz")
        if target.exists() and target.with_suffix(".json").exists():
            audits.append(full.read(target.with_suffix(".json")));continue
        print("[replay]",full.identity(record),label,flush=True)
        values,audit=replay_slice(evaluator,record,source,full.read(manifest["input_manifest"])["evaluation_seed"],anchor_hidden)
        np.savez_compressed(target,**values)
        audit.update(archive_sha256=full.sha(target),**full.identity(record))
        full.write(target.with_suffix(".json"),audit);audits.append(audit)
        print("[verified]",label,audit["verified_test_episodes_all_steps"],"/",audit["test_episodes"],"hidden error",audit["max_hidden_error"],flush=True)
    full.write(output/"completion.json",dict(status="completed",**full.identity(record),slices=audits,
        verified_test_episodes_all_steps=sum(a["verified_test_episodes_all_steps"] for a in audits),test_episodes=360,
        policy_retraining=False,probe_retraining=False,replay_only=True))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("path",type=Path);p.add_argument("--index",type=int,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--slice",choices=("train","test","additional"));p.add_argument("--anchor-hidden",action="store_true")
    args=p.parse_args();worker(args.path,args.index,args.out,args.slice,args.anchor_hidden)
