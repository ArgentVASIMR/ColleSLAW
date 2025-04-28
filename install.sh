#!/bin/bash

mkdir ColleSLAW
cd ./ColleSLAW
mkdir dataset
mkdir output
mkdir class

wget https://raw.githubusercontent.com/ArgentVASIMR/ColleSLAW/refs/heads/main/lora.py
wget https://raw.githubusercontent.com/ArgentVASIMR/ColleSLAW/refs/heads/main/lora-config.yaml
# wget https://raw.githubusercontent.com/ArgentVASIMR/ColleSLAW/refs/heads/main/run-training.sh

echo "ColleSLAW structure built"

git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
source venv/bin/activate

pip install torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt
pip install --force-reinstall "numpy<2"
pip install --force-reinstall --no-deps "bitsandbytes==0.45.5"
pip install dadaptation

echo ""
echo "Answers for the following questions:"
echo "- This machine"
echo "- No distributed training"
echo "- no"
echo "- no"
echo "- no"
echo "- all (or 0 if \"all\" fails)"
echo "- fp16"
echo ""

accelerate config

echo "ColleSLAW has reached the end of set-up. Refer back to wiki for further instructions."
echo "Do not close this window yet. If you experience technical issues when training your first LoRA, look for error messages further above."