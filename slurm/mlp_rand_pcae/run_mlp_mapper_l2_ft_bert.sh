#!/bin/bash --login
#SBATCH --time=15:00:00
#SBATCH --mail-user=habib.slim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -J mlp_mapper_l2_ft_bert
#SBATCH -o slurm/log/mlp_mapper_l2_ft_bert.log
#SBATCH --partition=batch
#SBATCH --mem=64G
#SBATCH --nodes=1
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
    --nproc_per_node=8 code/main_node2node.py \
    --accum_iter 1 \
    --model mlp_mapper_bert_l2 \
    --output_dir output/graph_edit/dm/mlp_mapper_l2_ft_bert \
    --log_dir output/graph_edit/dm/mlp_mapper_l2_ft_bert \
    --num_workers 8 \
    --batch_size 32 \
    --text_model_name "bert-base-uncased" \
    --dataset graphedits \
    --data_path /ibex/user/slimhy/ShapeWalk_RND/ \
    --data_type release \
    --exp_name mlp_mapper_l2_ft_bert \
    --dist_eval \
    --use_embeds \
    --ft_bert \
    --epochs 500 \
    --lr 1e-4 \
    --clip_grad 3.0
