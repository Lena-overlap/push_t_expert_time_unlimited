"""Closed-loop vision policy evaluation on independent Push-T seeds."""
import argparse, csv, json, time, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/'src'))
from dataset import make_env, exact_yaw_reset, reseed, cases, Video
from collection import wide_cases, read_cases, band, typical_video_seeds
from vision_data import rgb, proprio
from act.data import normalize
from act.model import ACT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=40000); p.add_argument('--per-bin',type=int,default=4)
    p.add_argument('--cases-file'); p.add_argument('--max-steps',type=int,default=1000); p.add_argument('--exec-steps',type=int,default=10)
    p.add_argument('--video-count',type=int,default=0); p.add_argument('--video-camera',default='agentview'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if a.max_steps<10 or a.per_bin<2 or a.per_bin%2 or a.video_count<0: p.error('Invalid evaluation settings')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True); cfg=ck['data_config']; stats=ck['stats']
    if not 1<=a.exec_steps<=ck['model_config']['chunk']: p.error('exec-steps must be within the model action chunk')
    model=ACT(**ck['model_config']).to(a.device); model.load_state_dict(ck['model']); model.eval()
    selected=read_cases(a.cases_file) if a.cases_file else cases(a.seed,a.per_bin,271828)+wide_cases(a.seed+6*a.per_bin,a.per_bin,314159)
    if set(s for s,_ in selected)&set(ck['reserved_seeds']): p.error('Evaluation seeds overlap with the training/validation dataset')
    videos=typical_video_seeds(selected,a.video_count) if a.video_count else set(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.csv').exists(): p.error('Choose a new output directory')
    (out/'eval_config.json').write_text(json.dumps(dict(vars(a),cameras=cfg['cameras'],size=cfg['size'],stable=10,success_threshold=.85),indent=2))
    fields=['seed','angle_deg','angle_bin','success','steps','final_iou','best_iou','final_streak','max_streak','final_eef_z','stop_reason','wall_seconds']
    rows=[]
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for seed,angle in selected:
            env=make_env(True,a.max_steps); video=None; start=time.monotonic()
            try:
                obs=exact_yaw_reset(env,seed,angle,reseed); m=env.task_metrics(); best=float(m['iou']); streak=0; maxstreak=0; success=False; t=0; reason='step_cap'
                if seed in videos:
                    from types import SimpleNamespace
                    video=Video(out/'videos'/f'seed_{seed}.mp4',env,SimpleNamespace(camera=a.video_camera,width=640,height=480,max_steps=a.max_steps))
                    video.append(str(seed),t,m,'ACT','reset',False)
                while t<a.max_steps and m['upright'] and m['on_table']:
                    q=normalize(proprio(obs),stats,'qpos'); im=np.stack([rgb(env,c,cfg['size']) for c in cfg['cameras']]).transpose(0,3,1,2).copy()
                    with torch.inference_mode():
                        y,_,_=model(torch.from_numpy(q)[None].to(a.device),torch.from_numpy(im)[None].to(a.device))
                    acts=y[0].cpu().numpy()*np.asarray(stats['action_std'])+np.asarray(stats['action_mean'])
                    if not np.isfinite(acts).all(): raise RuntimeError('Nonfinite ACT prediction')
                    for u in acts[:a.exec_steps]:
                        if t>=a.max_steps: break
                        obs,_,_,_=env.step(np.clip(u,-1.,1.)); t+=1; m=env.task_metrics(); best=max(best,float(m['iou']))
                        valid=m['upright'] and m['on_table']; ok=valid and m['iou']>=env.overlap_threshold
                        streak=streak+1 if ok else 0; maxstreak=max(maxstreak,streak)
                        success=streak>=10 and float(obs['robot0_eef_pos'][2])>=.895
                        if video: video.append(str(seed),t,m,'ACT','policy',success)
                        if success or not valid: break
                    if success: reason='success'; break
                if not (m['upright'] and m['on_table']): reason='invalid_pose'
                row=dict(seed=seed,angle_deg=angle,angle_bin=band(angle),success=bool(success),steps=t,final_iou=float(m['iou']),best_iou=best,final_streak=streak,max_streak=maxstreak,final_eef_z=float(obs['robot0_eef_pos'][2]),stop_reason=reason,wall_seconds=round(time.monotonic()-start,2))
                w.writerow(row); f.flush(); rows.append(row); print(row,flush=True)
            finally:
                try:
                    if video: video.close()
                finally: env.close()
    groups={'ALL':rows}
    for row in rows: groups.setdefault(row['angle_bin'],[]).append(row)
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['group','n','successes','success_rate','mean_final_iou','mean_success_steps']); w.writeheader()
        for name,g in groups.items():
            wins=[r['steps'] for r in g if r['success']]; row=dict(group=name,n=len(g),successes=len(wins),success_rate=len(wins)/len(g),mean_final_iou=float(np.mean([r['final_iou'] for r in g])),mean_success_steps=float(np.mean(wins)) if wins else '')
            w.writerow(row); print(row)

if __name__=='__main__': main()
