"""Train the visual action-chunk policy; validation uses inference with z=0."""
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from act.data import split_data, normalization, Chunks
from act.model import ACT, objective, masked_l1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',nargs='+',required=True); p.add_argument('--split-file',required=True); p.add_argument('--output',required=True)
    p.add_argument('--train-count',type=int); p.add_argument('--val-count',type=int,default=10)
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--samples-per-episode',type=int,default=16)
    p.add_argument('--chunk',type=int,default=50); p.add_argument('--hidden',type=int,default=256); p.add_argument('--latent',type=int,default=32); p.add_argument('--layers',type=int,default=3)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--kl-weight',type=float,default=10.); p.add_argument('--workers',type=int,default=2); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--pretrained',action='store_true',help='Use ImageNet ResNet18 weights (may download)')
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.samples_per_episode,a.chunk,a.hidden,a.latent,a.layers)<1 or a.hidden%8 or a.workers<0: p.error('Invalid training dimensions')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    tr,val,cfg=split_data(a.data,a.split_file,a.val_count,a.seed); all_seeds=[r['seed'] for r in tr+val]
    if a.train_count is not None:
        if not 1<=a.train_count<=len(tr): p.error(f'Only {len(tr)} training episodes available after validation split')
        tr=tr[:a.train_count]
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'history.csv').exists() or (out/'best.pt').exists(): p.error('Choose a new output directory')
    stats=normalization(tr); config=dict(qdim=cfg['qdim'],cameras=len(cfg['cameras']),chunk=a.chunk,hidden=a.hidden,latent=a.latent,layers=a.layers)
    model=ACT(**config,pretrained=a.pretrained).to(a.device)
    opt=torch.optim.AdamW([{'params':model.cnn.parameters(),'lr':a.lr*.1 if a.pretrained else a.lr}, {'params':[v for k,v in model.named_parameters() if not k.startswith('cnn.')]}],lr=a.lr,weight_decay=1e-4)
    def loader(rows,training):
        return DataLoader(Chunks(rows,stats,a.chunk,a.samples_per_episode,training),batch_size=a.batch_size,shuffle=training,num_workers=a.workers,pin_memory=a.device.startswith('cuda'),persistent_workers=a.workers>0)
    tl,vl=loader(tr,True),loader(val,False); best=float('inf')
    base=dict(model_config=config,data_config=cfg,stats=stats,train_seeds=[r['seed'] for r in tr],val_seeds=[r['seed'] for r in val],reserved_seeds=all_seeds,args=vars(a))
    (out/'config.json').write_text(json.dumps(base,indent=2)); print(f'Train {len(tr)} / validation {len(val)} episodes, device={a.device}',flush=True)
    with (out/'history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['epoch','loss','l1','kl','val_prior_l1']); w.writeheader()
        for ep in range(1,a.epochs+1):
            model.train(); sums=np.zeros(3); n=0
            for q,im,y,pad in tl:
                q,im,y,pad=[x.to(a.device,non_blocking=True) for x in (q,im,y,pad)]
                opt.zero_grad(set_to_none=True); pred,mu,lv=model(q,im,y,pad); loss,l1,kl=objective(pred,y,pad,mu,lv,a.kl_weight)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
                b=len(q); sums+=b*np.array([loss.item(),l1.item(),kl.item()]); n+=b
            model.eval(); err=0.; count=0
            with torch.no_grad():
                for q,im,y,pad in vl:
                    q,im,y,pad=[x.to(a.device,non_blocking=True) for x in (q,im,y,pad)]
                    pred,_,_=model(q,im); k=int((~pad).sum())*7; err+=masked_l1(pred,y,pad).item()*k; count+=k
            v=err/count; row=dict(epoch=ep,loss=sums[0]/n,l1=sums[1]/n,kl=sums[2]/n,val_prior_l1=v); w.writerow(row); f.flush(); print(row,flush=True)
            if not np.isfinite(v): raise RuntimeError('Nonfinite validation loss')
            if v<best:
                best=v; tmp=out/'best.tmp'; torch.save(dict(base,model=model.state_dict(),epoch=ep,val_prior_l1=v),tmp); tmp.replace(out/'best.pt')
        torch.save(dict(base,model=model.state_dict(),epoch=a.epochs,val_prior_l1=v),out/'last.pt')

if __name__=='__main__': main()
