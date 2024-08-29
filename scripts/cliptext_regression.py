# import sys
import os
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
# parser = argparse.ArgumentParser(description='Argument Parser')
# parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
# args = parser.parse_args()
# sub=int(args.sub)
# assert sub in [1,2,5,7]

import time
from scripts.regression_config import *
from new_utils import load_train_test_splits, load_seeg_resp

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-offset", "--offset", type=float, help="seeg resp offset in seconds",default=0.25)
parser.add_argument("-length", "--length", type=float, help="seeg resp length in seconds",default=0.2)
args = parser.parse_args()
resp_offset_second = args.offset
resp_seconds = args.length

resp_offset = int(scene_start + resp_offset_second*resp_freq)
resp_length = round(resp_seconds*resp_freq)

print(f"recording window: ({resp_offset_second}, {resp_offset_second+resp_seconds}) seconds / ({resp_offset}, {resp_offset+resp_length}) frames")
out_config = f"{ridge_mode}_fold-{nfolds}_split-{split_ii}_frame-{resp_offset_second}-{resp_offset_second+resp_seconds}"
print("out_config:", out_config)

pred_outdir = f"ecog_data/predicted_features/cliptext/{out_config}"
wts_outdir = f"ecog_data/regression_weights/cliptext/{out_config}"
if not os.path.exists(pred_outdir):
    os.makedirs(pred_outdir)
if not os.path.exists(wts_outdir):
    os.makedirs(wts_outdir)

# train_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
# train_fmri = np.load(train_path)
# test_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
# test_fmri = np.load(test_path)

# ## Preprocessing fMRI

# train_fmri = train_fmri/300
# test_fmri = test_fmri/300


# norm_mean_train = np.mean(train_fmri, axis=0)
# norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
# train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
# test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

# print(np.mean(train_fmri),np.std(train_fmri))
# print(np.mean(test_fmri),np.std(test_fmri))

# print(np.max(train_fmri),np.min(train_fmri))
# print(np.max(test_fmri),np.min(test_fmri))

_, train_inds, test_inds = load_train_test_splits(nfolds, split_ii)
train_resp, test_resp = load_seeg_resp(train_inds, test_inds, resp_offset, resp_length)

# num_voxels, num_train, num_test = train_fmri.shape[1], len(train_fmri), len(test_fmri)
num_voxels, num_train, num_test = train_resp.shape[1], len(train_resp), len(test_resp)

## load cliptext feats
# train_clip = np.load('data/extracted_features/subj{:02d}/nsd_cliptext_train.npy'.format(sub))
# test_clip = np.load('data/extracted_features/subj{:02d}/nsd_cliptext_test.npy'.format(sub))
all_clip = np.load('ecog_data/extracted_features/cars-2_cliptext.npy')
train_clip = all_clip[train_inds]
test_clip = all_clip[test_inds]

## Regression
num_samples,num_embed,num_dim = train_clip.shape

# print("Training Regression")
# reg_w = np.zeros((num_embed,num_dim,num_voxels)).astype(np.float32)
# reg_b = np.zeros((num_embed,num_dim)).astype(np.float32)
# pred_clip = np.zeros_like(test_clip)

# for i in range(num_embed):
#     reg = skl.Ridge(alpha=100000, max_iter=50000, fit_intercept=True)
#     reg.fit(train_fmri, train_clip[:,i])
#     reg_w[i] = reg.coef_
#     reg_b[i] = reg.intercept_
    
#     pred_test_latent = reg.predict(test_fmri)
#     std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
#     pred_clip[:,i] = std_norm_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)
#     print(i,reg.score(test_fmri,test_clip[:,i]))

# np.save('data/predicted_features/subj{:02d}/nsd_cliptext_predtest_nsdgeneral.npy'.format(sub),pred_clip)


# datadict = {
#     'weight' : reg_w,
#     'bias' : reg_b,

# }

# with open('data/regression_weights/subj{:02d}/cliptext_regression_weights.pkl'.format(sub),"wb") as f:
#   pickle.dump(datadict,f)


print("Training Regression")
# reg_w = np.zeros((num_embed,num_dim,num_voxels)).astype(np.float32)
# reg_b = np.zeros((num_embed,num_dim)).astype(np.float32)
pred_clip = np.zeros_like(test_clip)
alphas = np.logspace(2,5,5)
scores = []
for i in range(num_embed):
    if ridge_mode == "ridge":
        reg = skl.Ridge(alpha=100000, max_iter=50000, fit_intercept=True)
    elif ridge_mode == "ridgeCV":
        reg = skl.RidgeCV(alphas=alphas, cv=cv, scoring="explained_variance")
    t1 = time.time()
    reg.fit(train_resp, train_clip[:,i])
    t2 = time.time()
    print(f"time: {((t2-t1)/60)} min")

    # reg_w[i] = reg.coef_
    # reg_b[i] = reg.intercept_
    
    pred_test_latent = reg.predict(test_resp)
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
    pred = std_norm_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)
    pred_clip[:,i] = pred
    score = reg.score(test_resp,test_clip[:,i])
    print(i,score)
    scores.append(score)

    np.save(f'{pred_outdir}/layer-{i}.npy',pred)

    datadict = {
        'weight' : reg.coef_,
        'bias' : reg.intercept_,
    }
    with open(f'{wts_outdir}/layer-{i}.pkl',"wb") as f:
        pickle.dump(datadict,f)

    # break

np.save(f'ecog_data/regression_weights/cliptext/overall_score/{out_config}.npy',scores)
np.save(f'{pred_outdir}/allpreds.npy',pred_clip)

# datadict = {
#     'weight' : reg_w,
#     'bias' : reg_b,

# }

# with open(f'data/regression_weights/cliptext_regression_weights/{out_config}.pkl',"wb") as f:
#     pickle.dump(datadict,f)