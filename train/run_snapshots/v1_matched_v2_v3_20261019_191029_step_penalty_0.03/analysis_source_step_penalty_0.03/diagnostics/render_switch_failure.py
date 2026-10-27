"""Render a paired, recorded grid round; no policy or physics is invented."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from repo_paths import prepare_artifact_path,link_artifact


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib.font_manager as fm
    case=json.loads((a.analysis_dir/'illustrative_switch_failure.json').read_text())
    folder=Path(case['baseline_trace']).parents[1]
    prov=json.loads((folder/'provenance.json').read_text())
    layout=json.loads(Path(prov['layout_catalog'][case['layout_index']]['path']).read_text())
    walls=np.asarray(layout['grid'])==1
    red=np.asarray(layout['red_goal'])[::-1];blue=np.asarray(layout['blue_goal'])[::-1]
    font=ImageFont.truetype(fm.findfont('DejaVu Sans'),17)
    small=ImageFont.truetype(fm.findfont('DejaVu Sans'),13)
    datasets=[]
    for path in (case['baseline_trace'],case['hold_trace']):
        with np.load(path) as z:
            ep=case['episode'];mask=z['valid'][:,ep]&(z['round_idx'][:,ep]==case['round']-1)
            ix=np.flatnonzero(mask)
            assert np.all(z['layout_idx'][ix,ep]==case['layout_index'])
            d={key:z[key][ix,ep] for key in ('agent_pos','partner_assignment','effective_assignment','partner_move_ctr',
                'time','success','round_done','ego_move_effective','partner_move_effective')}
            assert d['round_done'][-1]
            datasets.append(d)
    def terminal(d):
        pos=d['agent_pos'][-1].copy()
        moves=np.array([d['ego_move_effective'][-1],d['partner_move_effective'][-1]])
        delta=np.array([[0,-1],[0,1],[1,0],[-1,0],[0,0]])[moves]
        proposed=pos+delta;h,w=walls.shape
        inside=(proposed[:,0]>=0)&(proposed[:,0]<w)&(proposed[:,1]>=0)&(proposed[:,1]<h)
        clipped=np.stack([np.clip(proposed[:,0],0,w-1),np.clip(proposed[:,1],0,h-1)],axis=1)
        bounced=~inside|walls[clipped[:,1],clipped[:,0]]
        result=np.where(bounced[:,None],pos,clipped)
        collision=np.array_equal(result[0],result[1]);swap=np.array_equal(result[0],pos[1]) and np.array_equal(result[1],pos[0])
        final=pos if collision or swap else result
        success=(np.array_equal(final[0],red) and np.array_equal(final[1],blue)) or (np.array_equal(final[0],blue) and np.array_equal(final[1],red))
        assert success==bool(d['success'][-1]),'Terminal reconstruction disagrees with saved success'
        return final
    frames=[];cell=44;margin=28;panel=walls.shape[1]*cell+2*margin;width=2*panel;height=475
    for tick in range(max(len(d['time']) for d in datasets)+12):
        img=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(img)
        draw.text((20,10),f"{case['target']} | {case['layout_id']} | delays (9, 0)",font=font,fill='black')
        draw.text((20,36),'E = ego, P = partner. Hold condition changes actions throughout the episode.',font=small,fill='#444444')
        for k,(name,d) in enumerate(zip(('Baseline','Hold assignment after tick 1'),datasets)):
            xoff=k*panel+margin;yoff=135;ended=tick>=len(d['time']);j=min(tick,len(d['time'])-1)
            pos=terminal(d) if ended else d['agent_pos'][j]
            assignment=int(d['effective_assignment'][-1] if ended else d['partner_assignment'][j])
            sw=((d['effective_assignment']!=d['partner_assignment'])&(d['time']>0))
            count=int(sw.sum() if ended else sw[:j].sum())
            draw.text((k*panel+20,65),name,font=font,fill='black')
            text=f"END: {'success' if d['success'][-1] else 'timeout'}" if ended else f"Round tick {int(d['time'][j])}; partner clock {int(d['partner_move_ctr'][j])}"
            draw.text((k*panel+20,90),text,font=small,fill='black')
            desc={0:'uncommitted',1:'ego red / partner blue',2:'ego blue / partner red'}[assignment]
            draw.text((k*panel+20,111),f'{desc}; switches {count}',font=small,fill='black')
            for y in range(walls.shape[0]):
                for x in range(walls.shape[1]):
                    draw.rectangle([xoff+x*cell,yoff+y*cell,xoff+(x+1)*cell,yoff+(y+1)*cell],fill='#777777' if walls[y,x] else '#f3f3f3',outline='#bbbbbb')
            for goal,color in ((red,'#d93838'),(blue,'#2977d1')):
                x,y=goal;draw.rectangle([xoff+x*cell+3,yoff+y*cell+3,xoff+(x+1)*cell-3,yoff+(y+1)*cell-3],fill=color)
            for xy,color,letter in ((pos[0],'#202020','E'),(pos[1],'#228b4c','P')):
                x,y=xy;cx=xoff+(x+.5)*cell;cy=yoff+(y+.5)*cell
                draw.ellipse([cx-14,cy-14,cx+14,cy+14],fill=color)
                draw.text((cx-6,cy-10),letter,font=font,fill='white')
        frames.append(img)
    logical=a.analysis_dir/'recorded_switch_failure.gif';physical=prepare_artifact_path(logical)
    frames[0].save(physical,save_all=True,append_images=frames[1:],duration=120,loop=0,optimize=True)
    link_artifact(logical)
    print(logical,physical.stat().st_size)

if __name__=='__main__':main()
