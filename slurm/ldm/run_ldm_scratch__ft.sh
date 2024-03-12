#!/bin/bash --login
#SBATCH --time=23:30:00
#SBATCH --mail-user=habib.slim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -J ldm_scratch__ft
#SBATCH -o slurm/log/ldm_scratch__ft.log
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:v100:8
##SBATCH --account conf-cvpr-2023.11.17-elhosemh

# Activate conda environment
source /home/slimhy/conda/bin/activate
conda activate shape2vecset
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment."
    exit 1
fi

# Run the job
cd /ibex/user/slimhy/Shape2VecSet
torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:8888 \
    --nproc_per_node=8 code/main_node2node.py \
    --accum_iter 2 \
    --model kl_d512_m512_l8_d24_edm \
    --ae kl_d512_m512_l8 \
    --ae_pth output/graph_edit/ae/ae_m512.pth \
    --output_dir output/graph_edit/dm/ldm_scratch__ft \
    --log_dir output/graph_edit/dm/ldm_scratch__ft \
    --resume output/graph_edit/dm/ldm_scratch__ft/checkpoint-42.pth \
    --resume_full_weights \
    --num_workers 8 \
    --point_cloud_size 2048 \
    --batch_size 16 \
    --save_every_n 5 \
    --text_model_name "bert-base-uncased" \
    --dataset graphedits \
    --data_path /ibex/user/slimhy/ShapeWalk/ \
    --data_type release \
    --exp_name ldm_scratch__ft \
    --dist_eval \
    --use_embeds \
    --n_replicas 1 \
    --clip_grad 3.0
