
## SEEG params
full_resp_length = 9216
resp_freq = 2048
scene_start = 2*resp_freq
## 0.15 sec around best EM performance (375-400ms)
resp_offset_second = 0.3
resp_seconds = 0.2
resp_offset = int(scene_start + resp_offset_second*resp_freq)
resp_length = round(resp_seconds*resp_freq)

## LR params
split_ii = 2
nfolds = 5
ridge_mode = "ridge"
cv = 5

print(f"recording window: ({resp_offset_second}, {resp_offset_second+resp_seconds}) seconds / ({resp_offset}, {resp_offset+resp_length}) frames")
out_config = f"{ridge_mode}_fold-{nfolds}_split-{split_ii}_frame-{resp_offset_second}-{resp_offset_second+resp_seconds}"
print("out_config:", out_config)
