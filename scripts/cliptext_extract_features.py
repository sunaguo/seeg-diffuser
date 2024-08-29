import sys
sys.path.append('versatile_diffusion')
# import os
import numpy as np
import json

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
# from torch.utils.data import DataLoader, Dataset

# from lib.model_zoo.vd import VD
# from lib.cfg_holder import cfg_unique_holder as cfguh
# from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
# import matplotlib.pyplot as plt
# import torchvision.transforms as T

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
# parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
# sub=int(args.sub)
# assert sub in [1,2,5,7]

## load model
cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.clip = net.clip.to(device)
   
# train_caps = np.load('data/processed_data/subj{:02d}/nsd_train_cap_sub{}.npy'.format(sub,sub)) 
# test_caps = np.load('data/processed_data/subj{:02d}/nsd_test_cap_sub{}.npy'.format(sub,sub))  

# num_embed, num_features, num_test, num_train = 77, 768, len(test_caps), len(train_caps)

# train_clip = np.zeros((num_train,num_embed, num_features))
# test_clip = np.zeros((num_test,num_embed, num_features))
# with torch.no_grad():
#     for i,annots in enumerate(test_caps):
#         cin = list(annots[annots!=''])
#         print(i)
#         c = net.clip_encode_text(cin)
#         test_clip[i] = c.to('cpu').numpy().mean(0)
    
#     np.save('data/extracted_features/subj{:02d}/nsd_cliptext_test.npy'.format(sub),test_clip)
        
#     for i,annots in enumerate(train_caps):
#         cin = list(annots[annots!=''])
#         print(i)
#         c = net.clip_encode_text(cin)
#         train_clip[i] = c.to('cpu').numpy().mean(0)
#     np.save('data/extracted_features/subj{:02d}/nsd_cliptext_train.npy'.format(sub),train_clip)


## SEEG params
full_resp_length = 9216
resp_freq = 2048
scene_start = 2*resp_freq
## 0.15 sec around best EM performance (375-400ms)
resp_offset_second = 0.25
resp_seconds = 0.3
resp_offset = int(scene_start + resp_offset_second*resp_freq)
resp_length = round(resp_seconds*resp_freq) 

## LR params
split_ii = 2
nfolds = 10
ridge_mode = "ridgeCV"
cv = 5

## load captions
with open("ecog_data/cars-2/frame_captions_upto-frame-2397.json","r") as f:
    all_caps = json.load(f)
# train_caps = [all_caps[ii] for ii in train_inds]
# test_caps = [all_caps[ii] for ii in test_inds]

num_embed, num_features, num_frames = 77, 768, len(all_caps)

all_clip = np.zeros((num_frames, num_embed, num_features))
with torch.no_grad():
    for i, (fid, cin) in enumerate(all_caps.items()):
        print(fid)
        print(cin, len(cin))
        c = net.clip_encode_text(cin).to('cpu').numpy()
        print(c.shape)
        all_clip[i] = c

        # cin = list(cin)
        # c = net.clip_encode_text(cin).to('cpu').numpy()
        # print(c.shape)
        # all_clip[i+1] = c.mean(0)

        # break
    
    np.save('ecog_data/extracted_features/cars-2_cliptext.npy', all_clip)
        
#     for i,annots in enumerate(train_caps):
#         cin = list(annots[annots!=''])
#         print(i)
#         c = net.clip_encode_text(cin)
#         train_clip[i] = c.to('cpu').numpy().mean(0)
#     np.save('data/extracted_features/subj{:02d}/nsd_cliptext_train.npy'.format(sub),train_clip)


