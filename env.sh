#!/usr/bin/env bash

export blob_dir="https://cloudaigcrdataeus.blob.core.windows.net/tunnel-data-code-eus/zhengjiani"
export SAS="?sv=2023-01-03&st=2025-02-13T03%3A15%3A52Z&se=2025-02-20T03%3A15%3A00Z&skoid=d42edb90-9b8e-4f54-aaa5-dc6c37cabd88&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2025-02-13T03%3A15%3A52Z&ske=2025-02-20T03%3A15%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=kV%2BNKMOobRgqCNzoCeq0gjd08vOvA2DLxn%2F84ax7URA%3D"
# ./azcopy copy "$blob_dir/checkpoints/Auto-UI-Base$SAS" /mimer/NOBACKUP/groups/softenable-design/zhengjiani --recursive
# azcopy copy ./checkpoints "$blob_dir$SAS" --recursive

# get data
# wget https://aka.ms/downloadazcopy-v10-linux
# tar -xvf downloadazcopy-v10-linux

# ./azcopy copy "$blob_dir/checkpoints/Auto-UI-Base$SAS" ./checkpoints --recursive
# ./azcopy copy "$blob_dir/checkpoints/cogagent-9b-20241220$SAS" ./checkpoints --recursive
# ./azcopy copy "$blob_dir/checkpoints/blip2-opt-2.7b$SAS" ./checkpoints --recursive
# ./azcopy copy "$blob_dir/checkpoints/roberta-base$SAS" ./checkpoints --recursive

# ./azcopy copy "$blob_dir/checkpoints/critic_1218$SAS" ./checkpoints --recursive
./azcopy copy "$blob_dir/checkpoints/Qwen2.5-VL-7B-Instruct$SAS" ./checkpoints --recursive

# ./azcopy copy "$blob_dir/checkpoints/SeeClick-aitw$SAS" ./checkpoints --recursive
# ./azcopy copy "$blob_dir/checkpoints/rl-webshop$SAS" ./checkpoints --recursive
# ./azcopy copy "$blob_dir/checkpoints/Qwen2-VL-7B-Instruct/model-00003-of-00005.safetensors$SAS" ./checkpoints/Qwen2-VL-7B-Instruct --recursive
# ./azcopy copy ./images "$blob_dir$SAS" --recursive

# if [ ! -d "images" ]; then
#    ./azcopy copy "$blob_dir/images$SAS" ./ --recursive
# fi

#conda create -n digirl python==3.10 -y
#conda activate digirl
#pip install -r digirl_requirements.txt
#sudo apt-get update
#sudo apt-get install libglib2.0-0 -y
#pip install openai azure-identity-broker --upgrade
#
#git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
#cd LLaMA-Factory
#pip install -e ".[torch,metrics]"

#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python /cosmos/zhengjiani/DPO/DPO_train.py

# i1 -A berzelius-2024-341 --gres=gpu:1  --time 5:00:00 --mail-type "BEGIN,END,FAIL" --mail-user "v-zhengjiani@microsoft.com"
# squeue -u x_wenyi -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"
# export PATH=/home/x_wenyi/.conda/envs/digirl/bin:$PATH

# module load virtualenv/20.24.6-GCCcore-13.2.0
