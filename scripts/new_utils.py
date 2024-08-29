import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class batch_generator_external_images(Dataset):

    def __init__(self, data_path, transpose=(0,2,3,1)):
        self.data_path = data_path
        # self.im = np.load(data_path).astype(np.uint8)
        self.im = np.load(data_path).astype(np.uint8).transpose(transpose)

    def __getitem__(self,idx):
        # img = Image.fromarray(self.im[idx])
        # img = T.functional.resize(img,(64,64))
        # img = torch.tensor(np.array(img)).float()
        #img = img/255
        #img = img*2 - 1

        img = self.im[idx]
        img = torch.tensor(img).float()

        return img

    def __len__(self):
        return  len(self.im)

def load_train_test_splits(nfolds, split_ii):
    print("loading splits")
    ## load cval splits
    splits = np.load(f"cval_splits_fold-{nfolds}.npz", allow_pickle=True)["splits"][()]
    train_inds = splits[split_ii]["train"]
    test_inds = splits[split_ii]["test"]
    print(f"split {split_ii} | n train: {len(train_inds)}, n test: {len(test_inds)}")
    return splits, train_inds, test_inds

def load_seeg_resp(train_inds, test_inds, resp_offset, resp_length, norm=True):
    print("loading resp")
    resp_path = "ecog_data/m00185-scene_info/m00185_scene_response_data_lap-reref.h5"
    resp_data = h5py.File(resp_path, 'r')['neural_data']
    ## channel x frame x length --> frame x channel x length
    ## slicing h5 dataset is much faster than getting whole thing out as nparray
    train_resp = resp_data[:,train_inds,resp_offset:resp_offset+resp_length].transpose(1,0,2).reshape(len(train_inds), -1)
    test_resp = resp_data[:,test_inds,resp_offset:resp_offset+resp_length].transpose(1,0,2).reshape(len(test_inds), -1)
    print("train resp:", train_resp.shape)
    print("test resp:", test_resp.shape)

    if norm: 
        print("normalizing resp")
        train_resp = train_resp/300
        test_resp = test_resp/300

        norm_mean_train = np.mean(train_resp, axis=0)
        norm_scale_train = np.std(train_resp, axis=0, ddof=1)
        train_resp = (train_resp - norm_mean_train) / norm_scale_train
        test_resp = (test_resp - norm_mean_train) / norm_scale_train

        print("mean, std")
        print("train:", np.mean(train_resp),np.std(train_resp))
        print("test:", np.mean(test_resp),np.std(test_resp))

        print("min, max")
        print("train:", np.max(train_resp),np.min(train_resp))
        print("test:", np.max(test_resp),np.min(test_resp))
        

    return train_resp, test_resp
