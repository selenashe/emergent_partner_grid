"""Reproducible phase dispatcher. Use python -m eval.partner_dynamics --help."""
import argparse
import json
import sys
import subprocess
from pathlib import Path
from .data import ROOT,inventory,write_json,stream_events
from .geometry import analyze

def main():
    # Audit guide:
    # Coordinate the dedicated analysis stages and route frozen-source collection
    # through subprocesses to isolate imports. Keep outputs scoped by batch and
    # allocation protocol. Reuse prior checkpoint evidence; this entry point does not
    # train new policies.
    #
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    parser.add_argument('--phase',choices=['audit','geometry','collect','replay','validate','controls','local','dynamics','report','all'],default='all')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--force',action='store_true')
    parser.add_argument('--protocol',choices=['v1','v2']);parser.add_argument('--condition');parser.add_argument('--seed',type=int)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    for key,value in [('protocols',args.protocol),('conditions',args.condition),('seeds',args.seed)]:
        if value is not None:config[key]=[value]
    config['smoke']=args.smoke
    if args.smoke:
        config['protocols']=config['protocols'][:1];config['conditions']=config['conditions'][:1];config['seeds']=config['seeds'][:1]
    root=ROOT/config['output_root']/config['batch']
    if args.smoke:root=root/'smoke'
    root.mkdir(parents=True,exist_ok=True)
    files=inventory(config)
    write_json(root/'data_audit.json',files);write_json(root/'configuration.json',config)
    write_json(root/'provenance.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),working_tree=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),command=sys.argv,python=sys.executable,
        batch_choice='Completed paired categorical counterbalanced batch',sources={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}))
    if args.dry_run or args.phase=='audit':
        print(f'Inventory: {len(files)} HDF5 files, {root}',flush=True);return
    for version in config['protocols']:
        chosen=[f for f in files if f['version']==version];protocol=chosen[0]['protocol'];directory=root/protocol
        if args.phase in ('geometry','all'):
            for condition in config['conditions']:
                for seed in config['seeds']:
                    out=directory/f'{condition}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
                    summary=out/'geometry_summary.json'
                    if summary.exists() and not args.force:
                        print(f'Cached {out.name}',flush=True);continue
                    print(f'Geometry {protocol} {condition} seed{seed}',flush=True)
                    hidden,index=stream_events([f for f in chosen if f['condition']==condition and f['seed']==seed],config,out)
                    analyze(hidden,index,out,config)
        if args.phase in ('collect','all'):
            # Subprocess isolates each frozen environment namespace and original trainer.
            subprocess.run([sys.executable,'-m','eval.partner_dynamics.collect','--version',version,'--config',str(root/'configuration.json'),'--output',str(directory),*(['--force'] if args.force else [])],cwd=ROOT,check=True)
        if args.phase in ('validate','all'):
            subprocess.run([sys.executable,'-m','eval.partner_dynamics.validate_capture','--version',version,'--config',str(root/'configuration.json'),'--output',str(directory)],cwd=ROOT,check=True)
        if args.phase in ('replay','all'):
            subprocess.run([sys.executable,'-m','eval.partner_dynamics.replay','--version',version,'--config',str(root/'configuration.json'),'--output',str(directory)],cwd=ROOT,check=True)
        if args.phase in ('controls','all'):
            subprocess.run([sys.executable,'-m','eval.partner_dynamics.rank_controls','--version',version,'--config',str(root/'configuration.json'),'--output',str(directory)],cwd=ROOT,check=True)
        if args.phase in ('local','all'):
            subprocess.run([sys.executable,'-m','eval.partner_dynamics.local','--version',version,'--config',str(root/'configuration.json'),'--output',str(directory)],cwd=ROOT,check=True)
        if args.phase in ('dynamics','all'):
            from .dynamics import run
            run(directory,config)
    if args.phase in ('report','all','geometry','dynamics'):
        from .report import build
        build(root,config)

if __name__=='__main__':main()
