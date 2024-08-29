# import sys
import os
import time
import h5py
import pickle
import numpy as np
import sklearn.linear_model as skl
import argparse
from new_utils import load_seeg_resp, load_train_test_splits

from scripts.regression_config import *

parser = argparse.ArgumentParser(description='Argument Parser')
# parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-offset", "--offset", type=float, help="seeg resp offset in seconds",default=0.25)
parser.add_argument("-length", "--length", type=float, help="seeg resp length in seconds",default=0.2)
args = parser.parse_args()
# sub=int(args.sub)
# assert sub in [1,2,5,7]
resp_offset_second = args.offset
resp_seconds = args.length

resp_offset = int(scene_start + resp_offset_second*resp_freq)
resp_length = round(resp_seconds*resp_freq)

print(f"recording window: ({resp_offset_second}, {resp_offset_second+resp_seconds}) seconds / ({resp_offset}, {resp_offset+resp_length}) frames")
out_config = f"{ridge_mode}_fold-{nfolds}_split-{split_ii}_frame-{resp_offset_second}-{resp_offset_second+resp_seconds}"
print("out_config:", out_config)

pred_outdir = f"ecog_data/predicted_features/vdvae/{out_config}"
wts_outdir = f"ecog_data/regression_weights/vdvae/{out_config}"
if not os.path.exists(pred_outdir):
   os.makedirs(pred_outdir)
if not os.path.exists(wts_outdir):
   os.makedirs(wts_outdir)

## linreg param
max_iter = 10000

## load cval splits
_, train_inds, test_inds = load_train_test_splits(nfolds, split_ii)

print("loading feats")
# nsd_features = np.load('data/extracted_features/subj{:02d}/nsd_vdvae_features_31l.npz'.format(sub))
# train_latents = nsd_features['train_latents']
# test_latents = nsd_features['test_latents']
feats = np.load("ecog_data/extracted_features/cars-2_vdvae_features_2397_31l.npz", allow_pickle=True)["latents"]
train_latents = feats[train_inds]
test_latents = feats[test_inds]
print("train latens:", train_latents.shape)
print("test latens:", test_latents.shape)

layer_sizes = np.load("vdvae/layer_sizes.npy")
layer_inds = np.concatenate([[0], np.cumsum(layer_sizes)])

# train_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
# train_fmri = np.load(train_path)
# test_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
# test_fmri = np.load(test_path)

## Preprocessing fMRI

# train_fmri = train_fmri/300
# test_fmri = test_fmri/300

# norm_mean_train = np.mean(train_resp, axis=0)
# norm_scale_train = np.std(train_resp, axis=0, ddof=1)
# train_resp = (train_resp - norm_mean_train) / norm_scale_train
# test_resp = (test_resp - norm_mean_train) / norm_scale_train

# print("mean, std")
# print("train:", np.mean(train_resp),np.std(train_resp))
# print("test:", np.mean(test_resp),np.std(test_resp))

# print("min, max")
# print("train:", np.max(train_resp),np.min(train_resp))
# print("test:", np.max(test_resp),np.min(test_resp))
train_resp, test_resp = load_seeg_resp(train_inds, test_inds, resp_offset, resp_length)

# num_voxels, num_train, num_test = train_resp.shape[1], len(train_resp), len(test_resp)

## latents Features Regression
print('Training latents Feature Regression')
alphas = np.logspace(2,5,5)
scores = []

for lii, feats in enumerate(layer_inds[:-1]): 
    print(f"=====layer {lii}  =====")
    feate = layer_inds[lii+1]

    train_latents_layer = train_latents[:, feats:feate]
    test_latents_layer = test_latents[:, feats:feate]
    assert train_latents_layer.shape[-1] == test_latents_layer.shape[-1] == layer_sizes[lii]

    print("resp size:", train_resp.shape)
    print("feat size:", train_latents_layer.shape)

    if ridge_mode == "ridge":
        reg = skl.Ridge(alpha=50000, max_iter=max_iter, fit_intercept=True)
    elif ridge_mode == "ridgeCV":
        reg = skl.RidgeCV(alphas=alphas, cv=cv, scoring="explained_variance")
    t1 = time.time()
    reg.fit(train_resp, train_latents_layer)
    t2 = time.time()
    print(f"max iter: {max_iter} | time: {((t2-t1)/60)} min")

    pred_test_latent = reg.predict(test_resp)
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
    pred_latents = std_norm_test_latent * np.std(train_latents_layer,axis=0) + np.mean(train_latents_layer,axis=0)
    score = reg.score(test_resp,test_latents_layer)
    print(score)
    scores.append(score)

    # print("weight size:", reg.coef_.shape)

    ## save each layer
    np.save(f'{pred_outdir}/cars-2_vdvae_nsdgeneral_pred_alpha50k_layer-{lii}.npy',pred_latents)

    datadict = {
        'weight' : reg.coef_,
        'bias' : reg.intercept_,
    }

    with open(f'{wts_outdir}/vdvae_regression_weights_layer-{lii}.pkl',"wb") as f:
      pickle.dump(datadict,f)

    # break
      
np.save(f"ecog_data/regression_weights/vdvae/overall_score/{out_config}.npy", scores)

# print("exited loop")

# np.save('data/predicted_features/subj{:02d}/nsd_vdvae_nsdgeneral_pred_sub{}_31l_alpha50k.npy'.format(sub,sub),pred_latents)


# datadict = {
#     'weight' : reg.coef_,
#     'bias' : reg.intercept_,

# }

# with open('data/regression_weights/vdvae_regression_weights.pkl',"wb") as f:
#   pickle.dump(datadict,f)