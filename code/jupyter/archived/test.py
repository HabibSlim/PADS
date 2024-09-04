def get_datasets(active_class):
    compat_dataset = CoMPaTSegmentDataset(
        "/ibex/project/c2273/3DCoMPaT/manifold_part_instances/",
        shape_cls=active_class,
        n_points=2**20,
        sampling_method="surface",
        recenter_mesh=True,
        process_mesh=True,
        scale_to_shapenet=True,
        align_to_shapenet=True,
        random_transform=False,
        force_retransform=False,
        remove_small_parts=False,
    )

    compat_part_drop_dataset = CoMPaTSegmentDataset(
        "/ibex/project/c2273/3DCoMPaT/manifold_part_instances/",
        shape_cls=active_class,
        n_points=2**20,
        sampling_method="surface",
        recenter_mesh=True,
        process_mesh=True,
        scale_to_shapenet=True,
        align_to_shapenet=True,
        random_transform=False,
        force_retransform=True,
        random_part_drop=True,
        n_parts_to_drop=1,
        remove_small_parts=False,
    )

    compat_random_aug_rotation_dataset = CoMPaTSegmentDataset(
        "/ibex/project/c2273/3DCoMPaT/manifold_part_instances/",
        shape_cls=active_class,
        n_points=2**20,
        sampling_method="surface",
        recenter_mesh=True,
        process_mesh=True,
        scale_to_shapenet=True,
        align_to_shapenet=True,
        random_transform=True,
        force_retransform=True,
        random_rotation=True,
    )

    compat_random_aug_no_rotation_dataset = CoMPaTSegmentDataset(
        "/ibex/project/c2273/3DCoMPaT/manifold_part_instances/",
        shape_cls=active_class,
        n_points=2**20,
        sampling_method="surface",
        recenter_mesh=True,
        process_mesh=True,
        scale_to_shapenet=True,
        align_to_shapenet=True,
        random_transform=True,
        force_retransform=True,
        random_rotation=False,
    )

    compat_random_all_aug_dataset = CoMPaTSegmentDataset(
        "/ibex/project/c2273/3DCoMPaT/manifold_part_instances/",
        shape_cls=active_class,
        n_points=2**20,
        sampling_method="surface",
        recenter_mesh=True,
        process_mesh=True,
        scale_to_shapenet=True,
        align_to_shapenet=True,
        random_transform=True,
        force_retransform=True,
        random_rotation=True,
        random_part_drop=True,
        n_parts_to_drop=1,
    )

    return {
        "orig": compat_dataset,
        "part_drop": compat_part_drop_dataset,
        "rand_rot": compat_random_aug_rotation_dataset,
        "rand_no_rot": compat_random_aug_no_rotation_dataset,
        "all_aug": compat_random_all_aug_dataset,
    }


def get_points(dataset, transform=None, obj_k=0):
    surface_points, occs, bbs = next(dataset[obj_k])
    return surface_points, occs, bbs


def export_dataset_entry(dataset, model_id, aug_id, out_path):
    surface_points, occs, bbs = get_points(dataset, obj_k=model_id)
    # Store the points
    torch.save("%s/%s_%s_points.npy" % (out_path, model_id, aug_id), surface_points)

    # Store the occupancy grid
    torch.save("%s/%s_%s_occs.npy" % (out_path, model_id, aug_id), occs)

    # Pickle the bounding boxes
    with open("%s/%s_%s_bbs.pkl" % (out_path, model_id, aug_id), "wb") as f:
        pickle.dump(bbs, f)


OUT_PATH = "/ibex/scratch/3dcot/3DCoMPaT/manifold_points"

# Info about the object to augment
MESH_ID = 0
ACTIVE_CLASS = "chair"
SAMPLES_PER_DATASET = 16

all_datasets = get_datasets(ACTIVE_CLASS)

for dataset_name, dataset in all_datasets.items():
    for i in range(SAMPLES_PER_DATASET):
        export_dataset_entry(dataset, MESH_ID, dataset_name + "_" + str(i), OUT_PATH)
