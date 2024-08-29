import sys, os
sys.path.append('versatile_diffusion')
# import os
# import os.path as osp
import PIL
from PIL import Image
# from pathlib import Path
import numpy as np
# import numpy.random as npr

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
# from torch.utils.data import DataLoader, Dataset

# from lib.model_zoo.vd import VD
# from lib.cfg_holder import cfg_unique_holder as cfguh
# from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
# import matplotlib.pyplot as plt
# from skimage.transform import resize, downscale_local_mean

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
# parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-diff_str", "--diff_str",help="Diffusion Strength",default=0.5)
parser.add_argument("-mix_str", "--mix_str",help="Mixing Strength",default=0.6)
args = parser.parse_args()
# sub=int(args.sub)
# assert sub in [1,2,5,7]
strength = float(args.diff_str)
mixing = float(args.mix_str)

print(f"strength: {strength}; mixing: {mixing}")

from scripts.reconstruct_config import *
from new_utils import load_train_test_splits

def regularize_image(x):
        newsize = [512, 512]
        BICUBIC = PIL.Image.Resampling.BICUBIC
        if isinstance(x, str):
            x = Image.open(x).resize(newsize, resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, PIL.Image.Image):
            x = x.resize(newsize, resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, np.ndarray):
            x = PIL.Image.fromarray(x).resize(newsize, resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            assert False, 'Unknown image type'

        assert (x.shape[1]==512) & (x.shape[2]==512), \
            'Wrong image size'
        return x

cfgm_name = 'vd_noema'
sampler = DDIMSampler_VD
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    


# Might require editing the GPU assignments due to Memory issues
net.clip.cuda(0)
net.autokl.cuda(0)

#net.model.cuda(1)
sampler = sampler(net)
#sampler.model.model.cuda(1)
#sampler.model.cuda(1)
batch_size = 1

print("loading predicted feats")
pred_text = np.load(f'ecog_data/predicted_features/cliptext/{out_config}/allpreds.npy')
pred_text = torch.tensor(pred_text).half().cuda(0)

pred_vision = np.load(f'ecog_data/predicted_features/clipvision/{out_config}/allpreds.npy')
pred_vision = torch.tensor(pred_vision).half().cuda(0)

_, train_inds, test_inds = load_train_test_splits(nfolds, split_ii)

out_dir = f"results/versatile_diffusion/{out_config}/diff-{strength}_mix-{mixing}"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


n_samples = 1
ddim_steps = 50
ddim_eta = 0
scale = 7.5
xtype = 'image'
ctype = 'prompt'
net.autokl.half()

torch.manual_seed(0)
for im_id, fid in enumerate(test_inds):
    print(f"image {im_id}/{len(test_inds)}")

    zim = Image.open(f'results/vdvae/{out_config}/{fid}.png')
   
    zim = regularize_image(zim)
    zin = zim*2 - 1
    zin = zin.unsqueeze(0).cuda(0).half()

    init_latent = net.autokl_encode(zin)
    
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    #strength=0.75
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    device = 'cuda:0'
    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))
    #z_enc,_ = sampler.encode(init_latent.cuda(1).half(), c.cuda(1).half(), torch.tensor([t_enc]).to(sampler.model.model.diffusion_model.device))

    dummy = ''
    utx = net.clip_encode_text(dummy)
    utx = utx.cuda(0).half()
    
    dummy = torch.zeros((1,3,224,224)).cuda(0)
    uim = net.clip_encode_vision(dummy)
    uim = uim.cuda(0).half()
    
    z_enc = z_enc.cuda(0)

    h, w = 512,512
    shape = [n_samples, 4, h//8, w//8]

    cim = pred_vision[im_id].unsqueeze(0)
    ctx = pred_text[im_id].unsqueeze(0)
    
    #c[:,0] = u[:,0]
    #z_enc = z_enc.cuda(1).half()
    
    sampler.model.model.diffusion_model.device='cuda:0'
    sampler.model.model.diffusion_model.half().cuda(0)
    #mixing = 0.4
    
    z = sampler.decode_dc(
        x_latent=z_enc,
        first_conditioning=[uim, cim],
        second_conditioning=[utx, ctx],
        t_start=t_enc,
        unconditional_guidance_scale=scale,
        xtype='image', 
        first_ctype='vision',
        second_ctype='prompt',
        mixed_ratio=(1-mixing), )
    
    z = z.cuda(0).half()
    x = net.autokl_decode(z)
    color_adj='None'
    #color_adj_to = cin[0]
    color_adj_flag = (color_adj!='none') and (color_adj!='None') and (color_adj is not None)
    color_adj_simple = (color_adj=='Simple') or color_adj=='simple'
    color_adj_keep_ratio = 0.5
    
    if color_adj_flag and (ctype=='vision'):
        x_adj = []
        for xi in x:
            color_adj_f = color_adjust(ref_from=(xi+1)/2, ref_to=color_adj_to)
            xi_adj = color_adj_f((xi+1)/2, keep=color_adj_keep_ratio, simple=color_adj_simple)
            x_adj.append(xi_adj)
        x = x_adj
    else:
        x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
        x = [tvtrans.ToPILImage()(xi) for xi in x]
    

    x[0].save(f'{out_dir}/{fid}.png')

    if im_id > 4: 
        break
      

