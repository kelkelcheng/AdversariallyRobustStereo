CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset=1 --nEpochs=20 --batchSize=2 --threads=2 --crop_height=240 --crop_width=576