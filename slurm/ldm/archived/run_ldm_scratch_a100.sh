#!/bin/bash --login
#SBATCH --time=15:00:00
#SBATCH --mail-user=habib.slim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -J ldm_scratch_a100_01
#SBATCH -o slurm/log/ldm_scratch_a100_01.log
#SBATCH --partition=batch
#SBATCH --mem=250G
#SBATCH --nodes=1
#SBATCH --reservation=A100
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
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
    --nproc_per_node=4 code/main_node2node.py \
    --accum_iter 2 \
    --model kl_d512_m512_l8_d24_edm \
    --ae kl_d512_m512_l8 \
    --ae-pth output/graph_edit/ae/ae_m512.pth \
    --output_dir output/graph_edit/dm/ldm_scratch_a100_01 \
    --log_dir output/graph_edit/dm/ldm_scratch_a100_01 \
    --num_workers 8 \
    --point_cloud_size 2048 \
    --batch_size 32 \
    --text_model_name "bert-base-uncased" \
    --dataset graphedits \
    --data_path /ibex/user/slimhy/ShapeWalk/ \
    --exp_name ldm_scratch_a100_01 \
    --dist_eval \
    --lr 1e-4 \
    --clip_grad 3.0

# To run with CLIP: "openai/clip-vit-base-patch32"
