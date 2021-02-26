CUDA_VISIBLE_DEVICES=1 python rebalselftraining.py /data/GTA-5 /data/cityscapes -s GTA5 -t Cityscapes --log logs/rebalselftraining/gtav2cityscapes
#CUDA_VISIBLE_DEVICES=0 python advent.py data/synthia data/Cityscapes -s Synthia -t Cityscapes \
#    --log logs/advent/synthia2cityscapes