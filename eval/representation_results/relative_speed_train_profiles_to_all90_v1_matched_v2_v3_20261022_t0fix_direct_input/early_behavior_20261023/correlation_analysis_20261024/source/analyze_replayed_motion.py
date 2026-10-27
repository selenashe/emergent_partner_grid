"""Summarize validated saved-state-anchored motion reconstructions."""
from pathlib import Path
import argparse
import shutil

import h5py
import numpy as np
import pandas as pd

import full_capability_evaluation as full
from analyze_decoding_behavior import CATEGORIES,history,equal_profile_mean
from plot_early_decoding import summarize


def motion_events(data):
    E,T=data["partner_goal"].shape
    moved=(data["pre_positions"][:,:,1]!=data["post_positions"][:,:,1]).any(axis=-1)&~data["round_done"]
    goal=data["partner_goal"]
    target=np.where((goal==1)[...,None],data["red_goal"],data["blue_goal"])
    off_goal=(data["pre_positions"][:,:,1]!=target).any(axis=-1)
    four_waits=np.zeros((E,T),bool);quick_repeat=np.zeros((E,T),bool)
    for e in range(E):
        last=-1;run=0;previous_goal=-1;previous_round=-1
        for t in range(T):
            current_goal=int(goal[e,t]);current_round=int(data["pre_round"][e,t])
            if current_goal!=previous_goal or current_round!=previous_round or data["is_t0"][e,t]:last=-1;run=0
            active=current_goal in (1,2) and not data["is_t0"][e,t]
            if active and moved[e,t]:
                quick_repeat[e,t]=last>=0 and t-last<=4
                last=t;run=0
            elif active and last>=0 and off_goal[e,t]:
                run+=1;four_waits[e,t]=run>=4
            else:run=0
            previous_goal=current_goal;previous_round=current_round
    return dict(first_partner_move_seen=history(moved)>0,second_partner_move_seen=history(moved)>=2,
        four_waits_after_move_seen=history(four_waits)>0,fast_repeat_seen=history(quick_repeat)>0,
        cadence_pattern_seen=history(four_waits|quick_repeat)>0,
        cooldown_wait_seen=history((data["pre_clock"]>0)&off_goal&~data["is_t0"]&~data["round_done"])>0,
        round_feedback_seen=history(data["round_done"])>0,moved=moved)


def load_motion(record,out):
    directory=Path(out)/"anchored_replay"/record["version"]/f"{record['condition']}_seed{record['seed']}"
    done=full.read(directory/"completion.json")
    parts=[]
    for name in ("train","test","additional"):
        path=directory/(name+"_replay.npz");audit=full.read(path.with_suffix(".json"))
        full.require(audit["anchored_on_saved_hidden_states"] and full.sha(path)==audit["archive_sha256"],"Unverified replay artifact")
        with np.load(path) as z:
            parts.append({k:z[k].copy() for k in ("pairs","repetitions","verified_prefix","pre_positions","post_positions","partner_goal","round_done","is_t0","pre_round","pre_clock","red_goal","blue_goal")})
    result={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order=np.lexsort((result["repetitions"],result["pairs"][:,1],result["pairs"][:,0]))
    return {k:v[order] for k,v in result.items()},done


def example_figure(root,out,record):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    d=out/"anchored_replay/v1_matched/rnn_diverse_influence_seed1/train_replay.npz"
    with np.load(d) as z:
        e=np.flatnonzero((z["pairs"]==[1,7]).all(axis=1)&(z["repetitions"]==1))[0]
        full.require(z["verified_prefix"][e,:31].all(),"Example is outside validated prefix")
        positions=z["pre_positions"][e,:31].copy();goals=z["red_goal"][e,0].copy(),z["blue_goal"][e,0].copy()
        moved=(z["pre_positions"][e,:,1]!=z["post_positions"][e,:,1]).any(axis=-1)&~z["round_done"][e]
        first_moves=np.flatnonzero(moved[:30])+1
        layout=int(z["pre_layout"][e,0]);walls=z["wall_maps"][layout].copy()
        clock=z["pre_clock"][e,:31].copy()
    with np.load(Path(record["probe_directory"])/"training_capability_probes.npz") as p:
        selected=(p["pairs"]==[1,7]).all(axis=1)&(p["repetitions"]==1)
        probability=p["probabilities"][:31,selected,0].ravel();W=p["weight"][20];b=p["bias"][20]
    with h5py.File(record["original_rollouts"]["train"],"r") as h:
        row=np.flatnonzero((h["capability"][:,0,:]==[1,7]).all(axis=1))[1]
        logits=h["hidden_state"][row,:31]@W+b;logits-=logits.max(axis=-1,keepdims=True)
        fixed=np.exp(logits);fixed=fixed[:,0]/fixed.sum(axis=-1)
    pd.DataFrame(dict(env_step=np.arange(31),red_faster_probability=probability,fixed20_red_probability=fixed,
        partner_x=positions[:,1,0],partner_y=positions[:,1,1],partner_cooldown_clock=clock,
        partner_move_just_observed=np.isin(np.arange(31),first_moves))).to_csv(out/"example_1_7_seed1_rep1.csv",index=False)
    fig=plt.figure(figsize=(15,8));grid=fig.add_gridspec(3,6,height_ratios=[1.2,1,.35],hspace=.55)
    for c,t in enumerate((0,2,4,6,8,10)):
        ax=fig.add_subplot(grid[0,c]);ax.imshow(walls,cmap="Greys",vmin=0,vmax=1,origin="upper")
        for goal,color in zip(goals,("#d94b41","#3779bf")):ax.add_patch(Rectangle((goal[0]-.4,goal[1]-.4),.8,.8,color=color,alpha=.5))
        ax.plot(positions[t,0,0],positions[t,0,1],"o",color="#e67e00",ms=10,label="Ego")
        ax.plot(positions[t,1,0],positions[t,1,1],"s",color="#222222",ms=9,label="Partner")
        ax.set(title=f"Observation t={t}",xticks=[],yticks=[])
    ax=fig.add_subplot(grid[1,:]);ax.plot(np.arange(31),probability,color="#9380ff",lw=2,label="Decoder fitted separately at each timestep")
    ax.plot(np.arange(31),fixed,color="#5d3b9c",lw=1.5,ls="--",label="One fixed step-20 decoder")
    for t in first_moves:ax.axvline(t,color="#333333",alpha=.3,lw=1)
    ax.axvspan(4,9,color="gray",alpha=.15,label="Green-category steepest window")
    ax.axhline(.5,color="gray",ls=":");ax.set(xlim=(0,25),ylim=(0,1.03),ylabel="Probability: red faster",xlabel="Environment step / observation index")
    ax.spines[["top","right"]].set_visible(False);ax.legend(loc="lower right",fontsize=9)
    bottom=fig.add_subplot(grid[2,:]);bottom.eventplot(first_moves,lineoffsets=.5,linelengths=.5,color="black")
    bottom.set(xlim=(0,25),ylim=(0,1),yticks=[.5],yticklabels=["Partner move observed"],xlabel="Blue pursued throughout; no round feedback in this window")
    bottom.spines[["top","right","left"]].set_visible(False)
    fig.suptitle("Held-out example: v1_matched multi-partner RNN, seed1, partner (1,7), repetition1\n"
        "Partner pursues blue; red is faster. Positions reconstructed using saved h_t and checked against recorded outcomes and one-step GRU states.",fontsize=12)
    for extension in ("png","pdf"):fig.savefig(out/f"example_partner_waiting.{extension}",dpi=180,bbox_inches="tight")
    plt.close(fig)


def run(path,out):
    path=Path(path).resolve();root=path.parent;out=Path(out).resolve();run=full.read(path)
    original=full.read(run["input_manifest"]);lookup={(p["red_delay"],p["blue_delay"]):p["category"] for p in original["profiles"]}
    rows=[];audits=[];nvalid=0;ntotal=0
    records=[r for r in run["policies"] if r["condition"]==full.CONDITIONS[0]]
    for record in records:
        print("[summarize reconstructed motion]",full.identity(record),flush=True)
        data,audit=load_motion(record,out);audits.append(audit)
        with np.load(Path(record["probe_directory"])/"training_capability_probes.npz") as p:
            selected=p["test"]
            np.testing.assert_array_equal(data["pairs"],p["pairs"][selected]);np.testing.assert_array_equal(data["repetitions"],p["repetitions"][selected])
            truth=p["truth"][selected];correct=p["predictions"][:31,selected]==truth[None,:]
        valid=data["verified_prefix"][:,30];nvalid+=int(valid.sum());ntotal+=len(valid)
        labels=np.array([lookup[tuple(p)] for p in data["pairs"]]);events=motion_events(data)
        for cat in CATEGORIES:
            chosen=valid&(labels==cat);full.require(chosen.any(),"No reconstructed episodes in category")
            for t in range(31):
                row=dict(**full.identity(record),category=cat,env_step=t,n_episodes=int(chosen.sum()),
                    accuracy=equal_profile_mean(correct[t,chosen],data["pairs"][chosen]))
                row.update({k:equal_profile_mean(v[chosen,t],data["pairs"][chosen]) for k,v in events.items() if k!="moved"})
                rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(out/"motion_per_seed.csv",index=False)
    metrics=[c for c in table if c not in ("version","condition","training_seed","category","env_step")]
    summary=summarize(table,["version","condition","category","env_step"],metrics);summary.to_csv(out/"motion_summary.csv",index=False)
    onsets=pd.read_csv(out/"onset_summary.csv");snapshots=[]
    for _,p in onsets[onsets.condition==full.CONDITIONS[0]].iterrows():
        for t in (int(p.peak5_start),int(p.peak5_end)):
            frame=summary[(summary.version==p.version)&(summary.category==p.category)&(summary.env_step==t)].iloc[0]
            snapshots.append(frame.to_dict())
    pd.DataFrame(snapshots).to_csv(out/"motion_at_rise_windows.csv",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(16,10.5),sharex=True,sharey=True)
    specs=(("accuracy","Decoding accuracy","#9380ff"),("first_partner_move_seen","First partner move observed","#555555"),
        ("second_partner_move_seen","At least two partner moves observed","#b36b00"),
        ("four_waits_after_move_seen","Four consecutive off-goal stays after a move","#168e88"),
        ("round_feedback_seen","First round feedback already observed","#111111"))
    for r,cat in enumerate(CATEGORIES):
        for c,version in enumerate(full.VERSIONS):
            ax=axes[r,c];g=summary[(summary.version==version)&(summary.category==cat)].sort_values("env_step")
            for key,label,color in specs:ax.plot(g.env_step,g[key+"_mean"],label=label,color=color,lw=1.6)
            p=onsets[(onsets.version==version)&(onsets.condition==full.CONDITIONS[0])&(onsets.category==cat)].iloc[0]
            ax.axvspan(p.peak5_start,p.peak5_end,color="gray",alpha=.15);ax.set_facecolor(full.CATEGORY_COLORS[cat])
            ax.set(title=f"{version} · {full.CATEGORY_LABELS[cat]}",xlim=(0,30),ylim=(0,1.02),xlabel="Environment step",ylabel="Accuracy / episode fraction")
            ax.title.set_fontsize(10);ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Early decodability and reconstructed partner movement\n"
        "Same held-out episodes at every timestep; only prefixes validated through step30; five seeds; events precede h_t",fontsize=12)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.94))
    for ext in ("png","pdf"):fig.savefig(out/f"decoding_and_partner_motion.{ext}",dpi=180)
    plt.close(fig)
    example_figure(root,out,records[0])
    source=out/"source_motion";target=source/Path(__file__).name
    if Path(__file__).resolve()!=target.resolve():shutil.copyfile(__file__,target)
    full.write(out/"motion_audit.json",dict(status="passed",completed_utc=full.now(),policies=len(records),test_episodes=ntotal,
        verified_episode_prefixes_through_step30=nvalid,coverage=nvalid/ntotal,invalid_prefixes_excluded=ntotal-nvalid,
        reconstruction="Replay original batch, layout reset seed and action keys. Anchor actor input to saved post-observation h_t and recurrent input to saved h_(t-1).",
        checks="Exact saved behavior fields and one-step GRU-state closeness at every prior transition; absolute hidden tolerance .002 accommodates measured CPU/GPU recurrent matmul roundoff. Any mismatch invalidates that episode from that step onward.",
        caveat="Reconstructed trajectories, not original position logs. Actor numerical differences can invalidate rare episodes. Summary uses a constant cohort validated through step30, reweights profiles equally within seeds, and excludes failed prefixes.",
        cue_definition="A repeated move within four transitions or four consecutive off-goal stays after a move, within the same pursued goal and round. This describes motion compatible with the fast/slow training families, not a causal test of the network's mechanism.",
        probe_refitting=False,policy_retraining=False,source_sha256={p.name:full.sha(p) for p in source.glob("*.py")}))
    print("[motion analysis completed]",nvalid,"/",ntotal,"verified prefixes",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("path",type=Path);p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();run(args.path,args.out)
