#!/bin/bash --login
#SBATCH --time=05:00:00
#SBATCH --mail-user=habib.slim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -J extract_features_01
#SBATCH -o slurm/log/extract_features_01.log
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:a100:1
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
srun python3 code/extract_features.py \
    --ae-pth /ibex/user/slimhy/Shape2VecSet/output/graph_edit/ae/ae_m512.pth \
    --ae kl_d512_m512_l8 \
    --ae-latent-dim 4096 \
    --text_model_name bert-base-uncased \
    --dataset graphedits \
    --data_path /ibex/user/slimhy/ShapeWalk/ \
    --data_type basic_edit \
    --batch_size 32 \
    --num_workers 8 \
    --device cuda \
    --fetch_keys \
    --seed 0
