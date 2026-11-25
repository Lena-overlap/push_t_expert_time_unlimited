"""Collect Push-T expert demonstrations using the forward-only controller."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, time
from pathlib import Path
import numpy as np
from dataset import cases, band as original_band
from contact_planner import load_model
from forward_expert import Runner, run_forward, wrapped, valid, ok

WIDE_BINS=tuple((lo,lo+15) for lo in range(90,180,15))

def band(angle):
    a=abs(float(angle))
    if a<=90: return original_band(angle)
    return next(f'{lo}-{hi}' for lo,hi in WIDE_BINS if lo<=a<hi or (hi==180 and a==180))


def wide_cases(seed,per_bin,angle_seed):
    """Balanced signed errors, strictly inside each 90–180 degree stratum."""
    if per_bin<2 or per_bin%2: raise ValueError('per-bin must be positive and even')
    rng=np.random.default_rng(angle_seed); result=[]
    for lo,hi in WIDE_BINS:
        signs=np.tile([1.,-1.],per_bin//2); rng.shuffle(signs)
        for sign in signs:
            result.append((seed+len(result),float(round(sign*rng.uniform(lo+.10,hi-.10),4))))
    return result


def typical_video_seeds(selected, count):
    """Preselect evenly spaced difficulty strata, alternating yaw signs if available."""
    ordered=sorted(selected,key=lambda item:(abs(item[1]),item[0]))
    if count is None or count>=len(ordered): return {seed for seed,_ in ordered}
    if count<1: raise ValueError('video count must be positive')
    chosen=set()
    for i,indices in enumerate(np.array_split(np.arange(len(ordered)),count)):
        group=[ordered[int(k)] for k in indices]
        target=(abs(group[0][1])+abs(group[-1][1]))/2
        sign=1 if i%2==0 else -1
        candidates=[item for item in group if item[1]*sign>0] or group
        chosen.add(min(candidates,key=lambda item:(abs(abs(item[1])-target),item[0]))[0])
    return chosen


def write_csv(path,rows):
    if not rows: return
    path=Path(path); keys=list(dict.fromkeys(k for row in rows for k in row))
    tmp=path.with_suffix('.tmp');
    with tmp.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def record_selected_video(r,node,path,args,label,video_cls=None):
    """Render only the selected action prefix; publish only after replay verification."""
    if video_cls is None:
        from dataset import Video as video_cls
    path=Path(path); temp=path.with_name(path.stem+'.partial.mp4')
    video=None
    try:
        video=video_cls(temp,r.env,args)
        r.replay_video(node,frame=lambda step,m,tag,success: video.append(label,step,m,*tag,success))
        frames=video.frames
        video.close(); video=None
        if frames!=node.steps+1: raise RuntimeError(f'Incomplete video: {frames} frames')
        if not temp.is_file() or temp.stat().st_size==0: raise RuntimeError('Empty video')
        temp.replace(path)
        return str(path),frames,''
    except Exception as exc:
        if video is not None:
            try: video.close()
            except Exception: pass
        temp.unlink(missing_ok=True)
        return '',0,str(exc)


def run_case(a,out,seed,angle,record_video):
    """Independent environment per seed; worker never writes shared CSVs."""
    from dataset import make_env, exact_yaw_reset
    from dataset import reseed
    model=load_model(a.response_model)
    # New environment per case, matching original physics and observation schema.
    video_error=''; video_path=''; video_frames=0; video_replay_steps=0
    max_steps=a.max_steps; strategy="loop"
    try: env=make_env(record_video,max_steps)
    except Exception as exc:
        if not record_video: raise
        video_error=str(exc); print(f'Offscreen unavailable: {exc}; continuing without video',flush=True); env=make_env(False,max_steps)
    start=time.monotonic(); trial_log=[]; error=''; result=None; rec=None
    try:
        r=Runner(env,lambda:exact_yaw_reset(env,seed,angle,reseed),a.stable,max_steps)
        init,final,best,win,rec,trial_log,forward_reason=run_forward(r,a,angle,model)
        if abs(np.rad2deg(wrapped(init.q[2]+np.deg2rad(angle))))>.05 or not valid(init.metrics):
            raise RuntimeError(f'Initial reset mismatch for seed {seed}')
        method='baseline' if win is not None and win.steps<=final.steps else 'forward_loop'
        if win is not None: result=win
        else: error=forward_reason
        saved=''; stats={}; demo_ok=False
        if result is not None and rec is not None:
            # Eligibility derives ONLY from recorded real environment states.
            if len(rec.actions)!=result.steps or not 1<=result.steps<=r.limit or len(rec.obs)!=result.steps+1 or not all(rec.success[-a.stable:]): raise AssertionError('Invalid expert trajectory')
            stats=rec.stats(); stats['demo_prefix_steps']=result.steps
            name=f'seed_{seed}_{angle:+.4f}.npz'
            # Never overwrite previous successful data silently; unique output per run recommended.
            target=out/'demos'/name
            saved=rec.save(target,dict(seed=seed,initial_yaw_deg=angle,controller=method,strategy=strategy,max_steps=r.limit,control_hz=20,action_dim=7,steps=result.steps,success_threshold=float(env.overlap_threshold),stable_steps=result.streak,final_iou=float(result.metrics['iou']),history=list(result.history),split='train_candidate' if a.collect else 'evaluation',calibration_sha256={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (a.rotation_model,a.response_model)})); demo_ok=True
        # Keep real baseline and selected/best failure traces for diagnosis.
        trace_dir=out/'traces'; trace_dir.mkdir(exist_ok=True)
        if final is not None: write_csv(trace_dir/f'{seed}_baseline.csv',final.trace)
        trace_node=result if demo_ok else best
        if trace_node.trace: write_csv(trace_dir/f'{seed}_selected_or_best.csv',trace_node.trace)
        row=dict(seed=seed,angle_deg=angle,angle_bin=band(angle),strategy=strategy,max_steps=r.limit,stop_reason=forward_reason,baseline_final_iou=final.metrics['iou'] if final is not None else '',baseline_final_success=ok(final.metrics,env) if final is not None else '',baseline_stable=(win is not None and win.steps<=final.steps) if final is not None else '',success=demo_ok,method=method if demo_ok else '',steps=result.steps if demo_ok else '',final_iou=result.metrics['iou'] if demo_ok else '',best_observed_iou=max(best.metrics['iou'],result.metrics['iou'] if demo_ok else 0),best_task_success=ok(trace_node.metrics,env),best_yaw_deg=trace_node.metrics['yaw_error_degrees'],best_position_m=trace_node.metrics['position_error'],forward_corrections=len(trial_log),physical_steps_including_replays=r.executed,demo_path=saved,error=error,wall_seconds=round(time.monotonic()-start,2),**stats)
        if record_video and not video_error:
            before=r.executed
            label=f'Seed {seed} | '+('selected' if demo_ok else 'best observed')
            video_path,video_frames,video_error=record_selected_video(r,trace_node,out/'videos'/f'seed_{seed}_{"success" if demo_ok else "best_failure"}.mp4',a,label)
            video_replay_steps=r.executed-before
            if video_error: print(f'Video error for {seed}: {video_error}',flush=True)
        row.update(baseline_evaluated_steps=final.steps if final is not None else '')
        row.update(total_wall_seconds=round(time.monotonic()-start,2),video_selected=record_video,video_path=video_path,video_frames=video_frames,video_error=video_error,video_replay_steps=video_replay_steps)
        return row, trial_log
    finally:
        env.close()


def read_cases(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    result=[(int(r['seed']),float(r['angle_deg'])) for r in rows]
    if not result or len(set(s for s,_ in result))!=len(result) or any(not math.isfinite(x) or abs(x)>180 or s<0 for s,x in result):
        raise ValueError('Case manifest must contain unique nonnegative seeds and finite angles in [-180,180]')
    return result


def iter_case_results(a,out,selected,video_seeds,worker=run_case):
    """Yield completed seeds. Only the parent process aggregates shared results."""
    if a.workers==1:
        for seed,angle in selected:
            yield worker(a,out,seed,angle,seed in video_seeds)
    else:
        import os, multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
        original={key:os.environ.get(key) for key in keys}
        try:
            for key in keys: os.environ[key]='1'
            with ProcessPoolExecutor(max_workers=min(a.workers,len(selected)),mp_context=mp.get_context('spawn')) as pool:
                futures=[pool.submit(worker,a,out,seed,angle,seed in video_seeds) for seed,angle in selected]
                for future in as_completed(futures): yield future.result()
        finally:
            for key,value in original.items():
                if value is None: os.environ.pop(key,None)
                else: os.environ[key]=value


def main(root):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(); group.add_argument('--fresh',action='store_true'); group.add_argument('--collect',action='store_true')
    group.add_argument('--cases-file',type=Path,help='Run exact seed/angle pairs from an existing cases.csv')
    parser.add_argument('--workers',type=int,default=1,help='Independent spawned simulator processes; start with 2–4')
    parser.add_argument('--max-steps',type=int,default=1000,help='Protective per-trajectory step cap (default 1000)')
    parser.add_argument('--wide-angles',action='store_true',help='Sample only absolute initial yaw errors from 90 to 180 degrees; requires --fresh or --collect')
    parser.add_argument('--per-bin',type=int,default=None); parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--flush-every',type=int,default=8,help='Write aggregated CSVs every N completed cases, and at the end')
    parser.add_argument('--case-seeds',type=int,nargs='+',help='Restrict to seed IDs in the selected case set')
    parser.add_argument('--stable',type=int,default=10); parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--plan-only',action='store_true')
    vg=parser.add_mutually_exclusive_group(); vg.add_argument('--video',action='store_true',help='Record selected or best failure trajectory'); vg.add_argument('--no-video',action='store_true')
    parser.add_argument('--video-count',type=int,default=None,help='Record only N representative cases, chosen before testing; requires --video')
    parser.add_argument('--camera',default='agentview'); parser.add_argument('--width',type=int,default=640); parser.add_argument('--height',type=int,default=480)
    a=parser.parse_args(); a.fresh = a.fresh or not (a.collect or a.cases_file); a.rotation_model=str(root/'config/contact_model.json'); a.response_model=str(root/'config/contact_response_model.json')
    a.strategy='loop'
    if not 100<=a.max_steps<=10000: parser.error("--max-steps must be between 100 and 10000")
    if min(a.width,a.height)<240 or a.width%16 or a.height%16: parser.error('Video dimensions must be multiples of 16, at least 240')
    if not 1<=a.stable<=40: parser.error('Invalid stable-step count')
    if a.video_count is not None and (not a.video or a.video_count<1): parser.error('--video-count requires --video and a positive count')
    if a.workers<1: parser.error('--workers must be positive')
    if a.flush_every<1: parser.error('Invalid flush interval')
    if a.workers>1 and a.video: parser.error('Use --no-video for parallel collection; record representative cases separately with --workers 1')
    if a.wide_angles and not (a.fresh or a.collect): parser.error('--wide-angles requires --fresh or --collect')
    if a.per_bin is not None and (a.per_bin<2 or a.per_bin%2): parser.error('--per-bin must be positive and even')
    if a.cases_file:
        try: selected=read_cases(a.cases_file)
        except (OSError,ValueError,KeyError) as exc: parser.error(str(exc))
        if a.case_seeds:
            selected=[(s,x) for s,x in selected if s in a.case_seeds]
            if len(selected)!=len(set(a.case_seeds)): parser.error('Unknown seed in case manifest')
    elif a.fresh or a.collect:
        sampler=wide_cases if a.wide_angles else cases
        default_seed=(22000 if a.collect else 9200) if a.wide_angles else (12000 if a.collect else 7200)
        selected=sampler(a.seed if a.seed is not None else default_seed,a.per_bin or (40 if a.collect else 4),20261141 if a.collect else 20261132)
    if (a.fresh or a.collect) and a.case_seeds:
        selected=[(s,x) for s,x in selected if s in a.case_seeds]
        if len(selected)!=len(set(a.case_seeds)): parser.error('Unknown seed in sampled cases')
    video_seeds=typical_video_seeds(selected,a.video_count) if a.video else set()
    out=a.output or root/'results'/('collect' if a.collect else 'fresh' if a.fresh else 'cases')
    if a.wide_angles and a.output is None: out=out.with_name(out.name+'_90_180')
    out=Path(out)
    if (out/'summary.csv').exists() or (out/'demos').exists():
        if a.output is not None: parser.error('Output directory already contains results; choose a new --output')
        out=out.with_name(out.name+'_'+time.strftime('%Y%m%d_%H%M%S'))
        i=1
        while out.exists():
            out=out.with_name(out.name+f'_{i}'); i+=1
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'cases.csv',[dict(seed=s,angle_deg=x,angle_bin=band(x),video_selected=s in video_seeds) for s,x in selected])
    if a.video: print('Video selection: '+', '.join(f'{s} ({x:+.2f}°)' for s,x in selected if s in video_seeds),flush=True)
    (out/'run_config.json').write_text(json.dumps(vars(a),default=str,indent=2),encoding='utf-8')
    if a.plan_only: print(f'{len(selected)} cases: {out}/cases.csv'); return
    rows=[]; logs=[]; order={s:i for i,(s,_) in enumerate(selected)}
    def flush():
        write_csv(out/'summary.csv',sorted(rows,key=lambda x:order[x['seed']]))
        write_csv(out/'corrections.csv',sorted(logs,key=lambda x:(order[x['seed']],x['trial'])))
    def accept(row,trial_log):
        seed=row['seed']; angle=row['angle_deg']
        rows.append(row); logs.extend(dict(seed=seed,**x) for x in trial_log)
        # One atomic per-case report protects completed work even before a batch flush.
        reports=out/'case_reports'; reports.mkdir(exist_ok=True)
        write_csv(reports/f'{seed}_summary.csv',[row]); write_csv(reports/f'{seed}_corrections.csv',trial_log)
        if len(rows)%a.flush_every==0: flush()
        base=row['baseline_final_iou']; label=f'{base:.3f}' if base!='' else '-'
        print(f'{seed} {angle:+.2f}° | staged baseline={label} | best={row["best_observed_iou"]:.3f} | success={row["success"]} | steps={row["steps"] or "-"} | {row["method"] or row["error"] or "no successful rollout"} | {row["total_wall_seconds"]:.1f}s | video={row["video_path"] or "-"}',flush=True)
    batch_start=time.monotonic()
    try:
        for row,trial_log in iter_case_results(a,out,selected,video_seeds): accept(row,trial_log)
    finally: flush()
    print(f'Batch wall time: {time.monotonic()-batch_start:.1f}s',flush=True)
    n=sum(x['success'] for x in rows); print(f'Expert demos: {n}/{len(rows)}. Results: {out}',flush=True)


