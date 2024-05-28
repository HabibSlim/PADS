"""
Prepare voxelized meshes for training.
"""

import argparse
import os
import glob
from voxelize.preprocess import (
    process_mesh,
)


def chunks(list, n):
    """
    Yield n number of striped chunks from l.
    """
    for i in range(0, n):
        yield list[i::n]


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Root directory of the datasets.",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        help="Number of processes to use for parallel processing.",
    )
    parser.add_argument(
        "--process_id",
        type=int,
        help="Process ID for parallel processing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Process ID: %d" % args.process_id)
    obj_path = "%s/obj/" % args.root_dir
    out_path = "%s/obj_manifold/" % args.root_dir
    out_path_vox = "%s/obj_manifold_vox/" % args.root_dir

    # Create output directory if it does not exist
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    if not os.path.exists(out_path_vox):
        os.makedirs(out_path_vox, exist_ok=True)

    all_obj_files = list(glob.glob(os.path.join(obj_path, "*.obj")))

    # List previously processed files in out_path_vox
    processed_files = list(glob.glob(os.path.join(out_path_vox, "*.npy")))
    processed_files = set([os.path.basename(f) for f in processed_files])
    if len(processed_files) > 0:
        print("Found %d previously processed files" % len(processed_files))
        all_obj_files = [
            f
            for f in all_obj_files
            if os.path.basename(f).replace(".obj", ".npy") not in processed_files
        ]
        print("Processing %d total files" % len(all_obj_files))

    all_obj_files.sort()
    process_files = list(chunks(all_obj_files, args.n_processes))[args.process_id]

    # Write the list of files to process to a file
    with open("process_files_%d.txt" % args.process_id, "w") as f:
        for obj_file in process_files:
            f.write("%s\n" % obj_file)

    print("Starting processing of %d files..." % len(process_files))
    for obj_index in range(len(process_files)):
        obj_model = process_files[obj_index]
        # Verify that the file exists
        if not os.path.exists(obj_model):
            print("File %s does not exist" % obj_model)
            continue
        process_mesh(obj_model, out_path, out_path_vox, flood_fill=True)
        if obj_index % 50 == 0:
            print(
                "Processed %d/%d files [proc: %d]"
                % (obj_index, len(process_files), args.process_id)
            )

    print("All done!")


if __name__ == "__main__":
    main()
