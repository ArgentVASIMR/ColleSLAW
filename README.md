# ColleSLAW
Collection of Scripts for LoRA training in Argent's Wrapper

Install (Linux):
```
git clone https://github.com/ArgentVASIMR/ColleSLAW.git
cd ./ColleSLAW

git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
source venv/bin/activate

pip install torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt
pip install --force-reinstall "numpy<2"
pip install --force-reinstall --no-deps "bitsandbytes==0.45.5"
pip install dadaptation

accelerate config
```
