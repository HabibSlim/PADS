#!/bin/bash --login
#SBATCH --time=15:00:00
#SBATCH --mail-user=habib.slim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -J ldm_finetune_v100_01
#SBATCH -o slurm/log/ldm_finetune_v100_01.log
#SBATCH --partition=batch
#SBATCH --mem=250G
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
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:8888 \
    --nproc_per_node=8 code/main_node2node.py \
    --accum_iter 1 \
    --model kl_d512_m512_l8_d24_edm \
    --resume output/graph_edit/dm/ldm_finetune_v100_01/checkpoint-10.pth \
    --resume_full_weights \
    --ae kl_d512_m512_l8 \
    --ae_pth output/graph_edit/ae/ae_m512.pth \
    --output_dir output/graph_edit/dm/ldm_finetune_v100_01 \
    --log_dir output/graph_edit/dm/ldm_finetune_v100_01 \
    --num_workers 8 \
    --batch_size 16 \
    --text_model_name "bert-base-uncased" \
    --dataset graphedits \
    --data_path /ibex/user/slimhy/ShapeWalk/ \
    --data_type basic_edit \
    --exp_name ldm_finetune_v100_01 \
    --dist_eval \
    --use_embeds \
    --lr 1e-4 \
    --clip_grad 3.0
