#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J process_voxels
#SBATCH -o process_voxels.17.out
#SBATCH -e process_voxels.17.err
#SBATCH --mail-user=habib.slim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --constraint=cascadelake

module load cuda

# Activate conda environment
source /home/slimhy/conda/bin/activate
conda activate shape2vecset
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment."
    exit 1
fi

# Run the job
cd /ibex/user/slimhy/PADS/code/

# Run all processes
for i in $(seq 8); do
    process_id=$((i + 136 - 1))
    echo "Running process $process_id"
    python3 voxel_prep.py \
        --root_dir /ibex/user/slimhy/PADS/data/ \
        --n_processes 256 \
        --process_id $process_id &
done

# Wait for all processes to finish
wait
