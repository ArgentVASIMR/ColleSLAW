import yaml # type: ignore
import math
import os

# Future:
# - Cache latents to disk

with open ("lora-config.yaml", "r") as file:
    config = yaml.safe_load(file)

args = "--max_data_loader_n_workers=1 --persistent_data_loader_workers --caption_extension=\".txt\" --max_token_length=225 --prior_loss_weight=1 --cache_latents --save_model_as=safetensors --enable_bucket --seed=\"0\" --bucket_reso_steps=64 --shuffle_caption "
opt_args = ""
extra_args = ""
train_file = "train_network.py"

def new_arg(argument_name,value):
    return "--" + argument_name + "=\"" + str(value) + "\" "

# All arrays follow this structure:
#   ["name_of_argument","name_of_config_option"]
# Which results in this when used with the new_arg function:
#   --name_of_argument=name_of_config_option

# Simple As-Is Group
as_is_args = [
    ["train_data_dir","dataset_img_dir"],
    ["reg_data_dir","class_img_dir"],
    ["clip_skip","clip_skip"],
    ["output_name","lora_name"],
    ["resolution","resolution"],
    ["bucket_reso_step","bucket_res_step"],
    ["keep_tokens","lock_front_tags"],
    ["lr_scheduler","lr_scheduler"],
    ["train_batch_size","batch_size"],
    ["gradient_accumulation_steps","grad_accum_step"],
    ["network_dim","network_dim"],
    ["network_alpha","network_dim"],
    ["optimizer_type","optimiser"],
    ["loss_type","loss_type"],
    ["network_module","network_module"],
    ["mixed_precision","precision"],
    ["save_precision","precision"],
]

args += new_arg("output_dir", str(config["output_dir"]) + "/" + str(config["lora_name"]))
args += new_arg("pretrained_model_name_or_path", str(config["base_model_dir"]) + "/" + str(config["base_model_name"]))
args += new_arg("max_bucket_reso", str(config["resolution"]*2))

for k in range(len(as_is_args)):
    args += new_arg(as_is_args[k][0],str(config[as_is_args[k][1]]))

# Simple Not-Zero Group
not_zero_args = [
    ["lr_warmup_steps","warmup_steps"],
    ["caption_dropout_rate","caption_dropout"],
    ["caption_tag_dropout_rate","tag_dropout"],
    ["min_snr_gamma","min_snr_gamma"],
    ["scale_weight_norms","scale_weight"],
    ["network_dropout","network_dropout"]
]

for k in range(len(not_zero_args)):
    if(not_zero_args[k][1] != 0):
        args += new_arg(not_zero_args[k][0],str(config[not_zero_args[k][1]]))

# Simple Boolean Group
if(config["sdxl"]):
    train_file = "sdxl_train_network.py" + " "

if(config["v_prediction"] != True):
    args += new_arg("noise_offset",str(config["noise_offset"]))
    args += new_arg("min_snr_gamma",str(config["min_snr_gamma"]))

bool_args = [
    ["v_parameterization --zero_terminal_snr","v_prediction"],
    ["fp8_base","sdxl"],
    ["flip_aug","flip_aug"],
    ["network_train_unet_only","unet_only"],
    ["gradient_checkpointing","grad_checkpoint"],
    ["mem_eff_attn","mem_eff_attn"],
    ["xformers","xformers"],
    ["sdpa","sdpa"]
]

for k in range(len(bool_args)):
    if(config[bool_args[k][1]]):
        args += "--" + bool_args[k][0] + " "

# Batch Size Scaling
final_steps = config["base_steps"]
final_unet_lr = config["unet_lr"]
final_te_lr = config["text_encoder_lr"]
eff_batch_size = config["batch_size"] * config["grad_accum_step"]

if (config["scale_steps"]):
    final_steps = round(final_steps / eff_batch_size)
if (config["scale_lr"]):
    final_unet_lr *= math.sqrt(eff_batch_size)
    final_te_lr *= math.sqrt(eff_batch_size)

# Save Amount Handling
if(config["save_amount"] != 0):
    save_nth_step = final_steps / config["save_amount"]
    args += "--save_every_n_steps=\"" + str(int(save_nth_step)) + "\" "

# LR-free optimisers
lr_free_list = [
    "dadaptation",
    "dadaptadam",
    "prodigy"
]

for k in range(len(lr_free_list)):
    if(config["optimiser"] == lr_free_list[k]):
        final_unet_lr = 1.0
        final_te_lr = 1.0
        new_arg("max_grad_norm",str(config["max_grad_norm"]))
        opt_args += "decouple=True use_bias_correction=True "
        print(lr_free_list[k])
        if(config["optimiser"].lower == "prodigy"):
            opt_args += "d_coef=" + str(config['optimiser_args']['d_coef']) + " "
        break

# Optimiser Args
for k in config["optimiser_args"]:
    opt_args += f"{k}={config['optimiser_args'][k]}" + " "

args += "--optimizer_args " + str(opt_args)

# Apply LR and step count args after all processing finished
args += new_arg("max_train_steps",str(final_steps))
args += new_arg("unet_lr",str(final_unet_lr))
args += new_arg("text_encoder_lr",str(final_te_lr))

command = "accelerate launch --num_cpu_threads_per_process 8 " + train_file + " " + args + " " + extra_args

os.chdir(config["trainer_dir"])
os.system(command)
