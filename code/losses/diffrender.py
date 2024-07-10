"""
Differentiable rendering functions from nvdiffrec/FlexiCubes.
"""

import numpy as np
import math

import torch
import nvdiffrast.torch as dr
import kaolin as kal
import util


def get_random_camera_batch(
    batch_size,
    fovy=np.deg2rad(45),
    iter_res=[512, 512],
    cam_near_far=[0.1, 1000.0],
    cam_radius=3.0,
    device="cuda",
    use_kaolin=True,
    seed=None,
):
    """
    Compute a batch of random cameras.
    """
    if seed is not None:
        kal.ops.random.manual_seed(seed)
    if use_kaolin:
        camera_pos = torch.stack(
            kal.ops.coords.spherical2cartesian(
                *kal.ops.random.sample_spherical_coords(
                    (batch_size,),
                    azimuth_low=0.0,
                    azimuth_high=math.pi * 2,
                    elevation_low=-math.pi / 2.0,
                    elevation_high=math.pi / 2.0,
                    device="cuda",
                ),
                cam_radius,
            ),
            dim=-1,
        )
        return kal.render.camera.Camera.from_args(
            eye=camera_pos + torch.rand((batch_size, 1), device="cuda") * 0.5 - 0.25,
            at=torch.zeros(batch_size, 3),
            up=torch.tensor([[0.0, 1.0, 0.0]]),
            fov=fovy,
            near=cam_near_far[0],
            far=cam_near_far[1],
            height=iter_res[0],
            width=iter_res[1],
            device="cuda",
        )
    else:

        def get_random_camera():
            proj_mtx = util.perspective(
                fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1]
            )
            mv = util.translate(0, 0, -cam_radius) @ util.random_rotation_translation(
                0.25
            )
            mvp = proj_mtx @ mv
            return mv, mvp

        mv_batch = []
        mvp_batch = []
        for i in range(batch_size):
            mv, mvp = get_random_camera()
            mv_batch.append(mv)
            mvp_batch.append(mvp)
        return torch.stack(mv_batch).to(device), torch.stack(mvp_batch).to(device)


def get_rotate_camera(
    itr,
    fovy=np.deg2rad(45),
    iter_res=[512, 512],
    cam_near_far=[0.1, 1000.0],
    cam_radius=3.0,
    device="cuda",
    use_kaolin=True,
):
    """
    Compute a camera that rotates around the object.
    """
    if use_kaolin:
        ang = (itr / 10) * np.pi * 2
        camera_pos = torch.stack(
            kal.ops.coords.spherical2cartesian(
                torch.tensor(ang), torch.tensor(0.4), -torch.tensor(cam_radius)
            )
        )
        return kal.render.camera.Camera.from_args(
            eye=camera_pos,
            at=torch.zeros(3),
            up=torch.tensor([0.0, 1.0, 0.0]),
            fov=fovy,
            near=cam_near_far[0],
            far=cam_near_far[1],
            height=iter_res[0],
            width=iter_res[1],
            device="cuda",
        )
    else:
        proj_mtx = util.perspective(
            fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1]
        )

        # Smooth rotation for display.
        ang = (itr / 10) * np.pi * 2
        mv = util.translate(0, 0, -cam_radius) @ (
            util.rotate_x(-0.4) @ util.rotate_y(ang)
        )
        mvp = proj_mtx @ mv
        return mv.to(device), mvp.to(device)


glctx = dr.RasterizeGLContext()


def render_mesh(
    mesh,
    camera,
    iter_res,
    return_types=["mask", "depth"],
    white_bg=False,
    wireframe_thickness=0.4,
):
    """
    Render a mesh using nvdiffrast, following the input cameras.
    """
    vertices_camera = camera.extrinsics.transform(mesh.vertices)

    # Projection: nvdiffrast take clip coordinates as input to apply barycentric perspective correction.
    # Using `camera.intrinsics.transform(vertices_camera) would return the normalized device coordinates.
    proj = camera.projection_matrix().unsqueeze(1)
    proj[:, :, 1, 1] = -proj[:, :, 1, 1]
    homogeneous_vecs = kal.render.camera.up_to_homogeneous(vertices_camera)
    vertices_clip = (proj @ homogeneous_vecs.unsqueeze(-1)).squeeze(-1)
    faces_int = mesh.faces.int()

    rast, _ = dr.rasterize(glctx, vertices_clip, faces_int, iter_res)

    out_dict = {}
    for type in return_types:
        if type == "mask":
            img = dr.antialias(
                (rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int
            )
        elif type == "depth":
            img = dr.interpolate(homogeneous_vecs, rast, faces_int)[0]
        elif type == "wireframe":
            img = torch.logical_or(
                torch.logical_or(
                    rast[..., 0] < wireframe_thickness,
                    rast[..., 1] < wireframe_thickness,
                ),
                (rast[..., 0] + rast[..., 1]) > (1.0 - wireframe_thickness),
            ).unsqueeze(-1)
        elif type == "normals":
            img = dr.interpolate(
                mesh.face_normals.reshape(len(mesh), -1, 3),
                rast,
                torch.arange(
                    mesh.faces.shape[0] * 3, device="cuda", dtype=torch.int
                ).reshape(-1, 3),
            )[0]
        if white_bg:
            bg = torch.ones_like(img)
            alpha = (rast[..., -1:] > 0).float()
            img = torch.lerp(bg, img, alpha)
        out_dict[type] = img

    return out_dict
