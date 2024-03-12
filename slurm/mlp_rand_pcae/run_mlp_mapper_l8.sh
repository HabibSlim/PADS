#!/bin/bash --login
#SBATCH --time=15:00:00
#SBATCH --mail-user=habib.slim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -J mlp_mapper_bert_l8_pcae
#SBATCH -o slurm/log/mlp_mapper_bert_l8_pcae.log
#SBATCH --partition=batch
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:v100:4
#SBATCH --account conf-cvpr-2023.11.17-elhosemh

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
    --accum_iter 1 \
    --model mlp_mapper_bert_l8_pcae \
    --ae kl_d512_m512_l8 \
    --ae_pth output/graph_edit/ae/ae_m512.pth \
    --output_dir output/graph_edit/dm/mlp_mapper_bert_l8_pcae \
    --log_dir output/graph_edit/dm/mlp_mapper_bert_l8_pcae \
    --num_workers 8 \
    --batch_size 32 \
    --text_model_name "bert-base-uncased" \
    --dataset graphedits \
    --data_path /ibex/user/slimhy/ShapeWalk_RND/ \
    --data_type release \
    --exp_name mlp_mapper_bert_l8_pcae \
    --dist_eval \
    --use_embeds \
    --alt_ae_embeds pc_ae \
    --lr 1e-4 \
    --clip_grad 3.0
