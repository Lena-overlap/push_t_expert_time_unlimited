"""Self-contained ACT-style visual CVAE / action-chunk Transformer for Panda OSC."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ACT(nn.Module):
    def __init__(self,qdim,cameras,chunk=50,hidden=256,latent=32,heads=8,layers=3,pretrained=False):
        super().__init__(); self.chunk=chunk; self.latent=latent; self.cameras=cameras
        backbone=resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.cnn=nn.Sequential(*list(backbone.children())[:-2]); self.pool=nn.AdaptiveAvgPool2d((4,4)); self.image_proj=nn.Linear(512,hidden)
        self.cam_pos=nn.Parameter(torch.randn(1,cameras,1,hidden)*.02)
        self.spatial_pos=nn.Parameter(torch.randn(1,1,16,hidden)*.02)
        self.q_proj=nn.Linear(qdim,hidden); self.act_proj=nn.Linear(7,hidden)
        self.cls=nn.Parameter(torch.zeros(1,1,hidden)); self.enc_pos=nn.Parameter(torch.randn(1,chunk+2,hidden)*.02)
        def enc(): return nn.TransformerEncoderLayer(hidden,heads,hidden*4,.1,batch_first=True,norm_first=True)
        self.posterior=nn.TransformerEncoder(enc(),layers,enable_nested_tensor=False)
        self.z_params=nn.Linear(hidden,latent*2); self.z_proj=nn.Linear(latent,hidden)
        self.obs_encoder=nn.TransformerEncoder(enc(),layers,enable_nested_tensor=False)
        self.extra_pos=nn.Parameter(torch.randn(1,2,hidden)*.02)
        self.queries=nn.Parameter(torch.randn(1,chunk,hidden)*.02)
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(hidden,heads,hidden*4,.1,batch_first=True,norm_first=True),layers,norm=nn.LayerNorm(hidden))
        self.out=nn.Linear(hidden,7)
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,1,3,1,1))

    def forward(self,q,images,actions=None,pad=None):
        b,c=images.shape[:2]; qp=self.q_proj(q); mu=logvar=None
        if actions is not None:
            if pad is None: raise ValueError('Posterior requires padding mask')
            tokens=torch.cat((self.cls.expand(b,-1,-1),qp[:,None],self.act_proj(actions)),dim=1)+self.enc_pos
            mask=torch.cat((torch.zeros(b,2,dtype=torch.bool,device=q.device),pad),dim=1)
            mu,logvar=self.z_params(self.posterior(tokens,src_key_padding_mask=mask)[:,0]).chunk(2,dim=-1)
            logvar=logvar.clamp(-10,10); z=mu+torch.randn_like(mu)*torch.exp(.5*logvar)
        else: z=q.new_zeros(b,self.latent)
        x=(images.float()/255.-self.mean)/self.std
        feat=self.pool(self.cnn(x.flatten(0,1))).flatten(2).transpose(1,2)
        feat=self.image_proj(feat).reshape(b,c,16,-1)+self.cam_pos+self.spatial_pos
        memory=torch.cat((torch.stack((qp,self.z_proj(z)),dim=1)+self.extra_pos,feat.flatten(1,2)),dim=1)
        memory=self.obs_encoder(memory)
        y=self.decoder(self.queries.expand(b,-1,-1),memory)
        return self.out(y),mu,logvar


def masked_l1(pred,target,pad):
    mask=(~pad).unsqueeze(-1)
    return ((pred-target).abs()*mask).sum()/(mask.sum()*target.shape[-1]).clamp_min(1)


def objective(pred,target,pad,mu,logvar,kl_weight):
    l1=masked_l1(pred,target,pad)
    kl=-.5*(1+logvar-mu.square()-logvar.exp()).sum(-1).mean()
    return l1+kl_weight*kl,l1,kl
