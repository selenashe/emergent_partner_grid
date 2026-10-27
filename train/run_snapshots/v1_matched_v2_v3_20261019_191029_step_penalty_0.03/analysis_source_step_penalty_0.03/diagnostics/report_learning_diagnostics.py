"""Write a consolidated status table after the diagnostic array terminates."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagnostics.learning_diagnostics import targets,target_name,write_json
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args()
    report=[]
    for row in targets():
        out=a.run_dir/'runs'/target_name(row)
        r=dict(target=target_name(row),training='not_started',evaluation='not_started')
        for filename,key in [('status.json','training'),('evaluation_status.json','evaluation')]:
            if (out/filename).exists():
                data=json.loads((out/filename).read_text())
                r[key]=data.get('state')
                r[key+'_last_update']=data.get('last_update',data.get('last_checkpoint'))
                if data.get('error'): r[key+'_error']=data['error']
        if (out/'fidelity.json').exists():
            r['historical_fidelity']=json.loads((out/'fidelity.json').read_text())
        report.append(r)
    write_json(a.run_dir/'pipeline_status.json',dict(targets=report,complete=all(
        r['training']=='complete' and r['evaluation']=='complete' for r in report)))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
