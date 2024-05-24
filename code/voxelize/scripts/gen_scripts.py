total_processes = 256
processes_per_node = 8
in_script = """#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J process_voxels
#SBATCH -o process_voxels.%d.out
#SBATCH -e process_voxels.%d.err
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
for i in $(seq %d); do
    process_id=$((i + %d - 1))
    echo "Running process $process_id"
    python3 voxel_prep.py \\
        --root_dir /ibex/user/slimhy/PADS/data/ \\
        --n_processes %d \\
        --process_id $process_id &
done

# Wait for all processes to finish
wait
"""

# Write to file
n_scripts = total_processes // processes_per_node
for i in range(n_scripts):
    with open("scripts/process_voxels_%d.sh" % i, "w") as f:
        f.write(
            in_script
            % (i, i, processes_per_node, i * processes_per_node, total_processes)
        )
