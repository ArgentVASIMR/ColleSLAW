# ColleSLAW
Collection of Scripts for LoRA training in Argent's Wrapper

## Install:
Note: It is recommended to not use git bash; users have experienced problems running `accelerate config` with it, which is vital to installation.
```bash
git clone https://github.com/ArgentVASIMR/ColleSLAW.git
cd ./ColleSLAW

git clone -b sd3 https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
```
**You must activate the venv, do not skip the command below:**
- If using Windows: `venv\Scripts\activate`
- If using Linux: `source venv/bin/activate`
```bash
pip install torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt
pip install --force-reinstall "numpy<2"
pip install --force-reinstall --no-deps "bitsandbytes==0.45.5"

accelerate config
```
The answers for the questions from `accelerate config` are:
```
Q: Which type?        A: No distributed training
Q: CPU only?          A: no
Q: Torch dynamo?      A: no
Q: DeepSpeed?         A: no
Q: What GPUs?         A: all
Q: Numa efficiency?   A: no
Q: Mixed precision?   A: fp16
```

## First time run:
1. Inside of config.yaml, set `base_model_dir` to where your models are located.
2. Add a dataset to the `dataset` folder. This must be of the structure `ColleSLAW/dataset/#_first,#_second,#_others`.
3. Run the `run-training` file for your respective operating system. For Windows, it's `.bat`. For Linux, it's `.sh`.
