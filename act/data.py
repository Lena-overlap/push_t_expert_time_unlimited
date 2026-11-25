"""Episode-level split, train-only normalization and variable-length ACT chunks."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


def inspect(path):
    path = Path(path).resolve()
    with h5py.File(path, 'r') as f:
        if f.attrs.get('schema') != 'push_t_act_v1':
            raise ValueError(f'Not a Push-T ACT episode: {path}')
        meta = json.loads(f.attrs['metadata_json']); cams = json.loads(f.attrs['cameras_json'])
        t = len(f['action']); q = f['observations/qpos']
        if t < 1 or f['action'].shape != (t, 7) or len(q.shape) != 2 or len(q) != t+1:
            raise ValueError(f'Invalid episode shapes: {path}')
        stable = int(meta['stable_steps'])
        if len(f['task/success']) != t+1 or not cams or len(set(cams))!=len(cams):
            raise ValueError(f'Invalid task labels or cameras: {path}')
        if stable < 10 or stable > t or not np.all(f['task/success'][-stable:]):
            raise ValueError(f'Not a stable demonstration: {path}')
        if len(f['task/iou']) != t+1 or float(f['task/iou'][-1]) < .85:
            raise ValueError(f'Invalid terminal IoU: {path}')
        size = f['observations/images/'+cams[0]].shape[1]
        for c in cams:
            d=f['observations/images/'+c]
            if d.shape != (t+1,size,size,3) or d.dtype != np.uint8:
                raise ValueError(f'Invalid image shape: {path}, {c}')
        if not np.isfinite(q[:]).all() or not np.isfinite(f['action'][:]).all() or np.max(np.abs(f['action'][:]))>1.00001:
            raise ValueError(f'Invalid numeric data: {path}')
        return dict(path=str(path),seed=int(meta['seed']),angle=float(meta['initial_yaw_deg']),steps=t,
                    cameras=cams,size=size,qdim=q.shape[1],split=meta['split'],
                    calibration=meta.get('calibration_sha256',{}))


def balanced_order(rows, rng):
    # Round-robin across signed 15-degree bins, including the difficult range.
    groups=defaultdict(list)
    for r in sorted(rows,key=lambda r:r['seed']):
        a=r['angle']; groups[(min(11,int(abs(a)//15)),a<0)].append(r)
    keys=sorted(groups)
    for k in keys: rng.shuffle(groups[k])
    rng.shuffle(keys); result=[]
    while any(groups.values()):
        for k in keys:
            if groups[k]: result.append(groups[k].pop())
    return result


def split_data(roots, path, val_count, seed):
    paths=sorted({p.resolve() for root in roots for p in Path(root).rglob('*.hdf5') if not p.name.endswith('.partial.hdf5')})
    rows=[inspect(p) for p in paths]
    rows=[r for r in rows if r['split']=='train_candidate']
    if not rows: raise ValueError('No training candidates; collect with --collect --act-images')
    ids=[r['seed'] for r in rows]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate seed: keep one episode per seed across all directories')
    cfg={k:rows[0][k] for k in ('cameras','size','qdim','calibration')}
    if any(any(r[k]!=v for k,v in cfg.items()) for r in rows): raise ValueError('Camera, dimensions or calibration differ across episodes')
    path=Path(path)
    if path.exists():
        saved=json.loads(path.read_text()); by={r['seed']:r for r in rows}
        old=saved['train']+saved['val']
        if set(old)!=set(ids) or len(set(old))!=len(old) or saved['config']!=cfg:
            raise ValueError('Dataset changed; create a new split file, or use original dataset')
        if len(saved['val'])!=val_count: raise ValueError('Existing split has a different validation count')
        return [by[i] for i in saved['train']], [by[i] for i in saved['val']], cfg
    if not 1<=val_count<len(rows): raise ValueError('Need at least one validation episode and one training episode')
    order=balanced_order(rows,np.random.default_rng(seed)); val=order[:val_count]
    train=balanced_order(order[val_count:],np.random.default_rng(seed+1))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'train':[r['seed'] for r in train],'val':[r['seed'] for r in val],'config':cfg},indent=2))
    return train,val,cfg


def normalization(rows):
    moments={k:[0,None,None] for k in ('qpos','action')}
    for row in rows:
        with h5py.File(row['path'],'r') as f:
            for k,a in [('qpos',f['observations/qpos'][:-1]),('action',f['action'][:])]:
                a=np.asarray(a,np.float64); n,s,ss=moments[k]
                v=a.sum(0); vv=(a*a).sum(0)
                moments[k]=[n+len(a),v if s is None else s+v,vv if ss is None else ss+vv]
    out={}
    for k,(n,s,ss) in moments.items():
        m=s/n; sd=np.maximum(np.sqrt(np.maximum(ss/n-m*m,0)),.01)
        out[k+'_mean']=m.tolist(); out[k+'_std']=sd.tolist()
    return out


def normalize(a,stats,key):
    return ((a-np.asarray(stats[key+'_mean'],np.float32))/np.asarray(stats[key+'_std'],np.float32)).astype(np.float32)


class Chunks(Dataset):
    def __init__(self,rows,stats,chunk,samples=16,training=True):
        self.rows=rows; self.stats=stats; self.chunk=chunk; self.samples=samples; self.training=training
    def __len__(self): return len(self.rows)*self.samples
    def __getitem__(self,i):
        row=self.rows[i//self.samples]; tmax=row['steps']
        # Independent random start at training time; fixed coverage in validation.
        t=int(np.random.randint(tmax)) if self.training else min(tmax-1,int((i%self.samples+.5)*tmax/self.samples))
        n=min(self.chunk,tmax-t)
        with h5py.File(row['path'],'r') as f:
            q=normalize(f['observations/qpos'][t],self.stats,'qpos')
            images=np.stack([f['observations/images/'+c][t] for c in row['cameras']]).transpose(0,3,1,2).copy()
            actions=np.zeros((self.chunk,7),np.float32); actions[:n]=normalize(f['action'][t:t+n],self.stats,'action')
        pad=np.arange(self.chunk)>=n
        return torch.from_numpy(q),torch.from_numpy(images),torch.from_numpy(actions),torch.from_numpy(pad)
