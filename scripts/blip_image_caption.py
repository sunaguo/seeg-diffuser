import time
import requests
import json
import glob

import numpy as np
import pandas as pd

def query_blip(filename):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": "Bearer <huggingfacetoken>"}
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def query_blip_rec(frameid):
    ## query
    if frameid > len(stim_metadata): return frameid-1
    
    filename = f"ecog_data/cars-2/frames/cars-2-{frameid}.jpg"
    cap = query_blip(filename)
    print(frameid, cap)
    
    ## if success: add to caps, call next
    if type(cap) == list:
        blip_caps[int(frameid)] = cap[0]["generated_text"]
#         time.sleep(1)
        return query_blip_rec(frameid+1)
    ## else: sleep, call again
    elif type(cap) == dict:
        ## reaching hourly rate
        if "reset hourly" in cap["error"]:
            return frameid-1
        ## just model busy
        time.sleep(3)
        return query_blip_rec(frameid)


stim_metadata = pd.read_csv("ecog_data/m00185-scene_info/m00185_scene_stimulus_metadata.csv")

fns = glob.glob("ecog_data/cars-2/frame_captions_upto-frame*")
ids = [int(fn.split("/")[-1].split(".")[0].split("-")[-1]) for fn in fns]
lastID = max(ids)
with open(f"ecog_data/cars-2/frame_captions_upto-frame-{lastID}.json","r") as f:
    blip_caps = json.load(f)

# blip_caps = {}
# for ind in np.arange(len(stim_metadata))+1:
#     print(ind)  #if ind % 100 == 0: 
#     filename = f"ecog_data/cars-2/cars-2-{ind}.jpg"
    
#     cap = query_blip(filename)
#     print(cap)
#     blip_caps[int(ind)] = cap[0]["generated_text"]
#     time.sleep(1)

# blip_caps = {}
while lastID < len(stim_metadata):
    lastID = query_blip_rec(lastID+1)

    with open(f"ecog_data/cars-2/frame_captions_upto-frame-{lastID}.json", "w") as f: 
        json.dump(blip_caps, f)

    time.sleep(60*58)