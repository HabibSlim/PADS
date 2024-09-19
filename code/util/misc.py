"""
Miscellaneous utility functions/classes.
"""

import builtins
import datetime
import json
import os
import sys
import time
from contextlib import contextmanager
from collections import defaultdict, deque
from pathlib import Path

import cProfile
import numpy as np
import pstats
import random
import torch
import torch.distributed as dist
import trimesh
import wandb
from torch import inf
from functools import wraps
from torch_cluster import fps


"""
Logging and statistics classes.
"""


def init_wandb(project_name, exp_name, model_config, wandb_id=None):
    """
    Initialize a new W&B run.
    """
    do_resume = wandb_id is not None
    # Generate a random hex string for the run ID
    wandb_id = wandb.util.generate_id() if wandb_id is None else wandb_id
    print(project_name, exp_name, wandb_id)
    wandb.init(
        project=project_name,
        name=exp_name + "__" + wandb_id,
        config=model_config,
        resume=do_resume,
    )

    # Log the model params
    print("Model params:\n", json.dumps(model_config, indent=4, sort_keys=True))


def load_pickle(path):
    """
    Load a Python object from a pickle file.
    """
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(pythn_obj, path):
    """
    Dump a Python object to a pickle file.
    """
    import pickle

    with open(path, "wb") as f:
        pickle.dump(pythn_obj, f)


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def reset(self):
        for name, meter in self.meters.items():
            self.meters[name] = SmoothedValue(fmt="{avg:.4f}")

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


@contextmanager
def stderr_redirected(to=os.devnull):
    """
    Redirect standard error (by default: to os.devnull).
    Useful to silence the noisy output of Blender.
    """
    fd = sys.stderr.fileno()

    def _redirect_stderr(to):
        sys.stderr.close()
        os.dup2(to.fileno(), fd)
        sys.stderr = os.fdopen(fd, "w")

    with os.fdopen(os.dup(fd), "w") as old_stderr:
        with open(to, "w") as file:
            _redirect_stderr(to=file)
        try:
            yield
        finally:
            _redirect_stderr(to=old_stderr)


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    Redirect standard output (by default: to os.devnull).
    Useful to silence the noisy output of Blender.
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


"""
Basic benchmarking methods.
"""
LAST_TIME = 0.0
BENCHMARK = False


def toggle_benchmarking():
    global BENCHMARK
    BENCHMARK = not BENCHMARK
    print("Benchmarking is now [%s]." % ("on" if BENCHMARK else "off"))


def timer_init(section_name):
    if not BENCHMARK:
        return
    global LAST_TIME
    LAST_TIME = time.time()
    print("Starting [%s]..." % section_name)


def timer_end():
    if not BENCHMARK:
        return
    global LAST_TIME
    print("Done. Elapsed time: %.2f" % (time.time() - LAST_TIME))


def profile_func(function_wrapper, exp_name):
    with cProfile.Profile() as pr:
        function_wrapper()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    filename = "%s.prof" % exp_name
    stats.dump_stats(filename=filename)
    print("Dumped profile to [%s]." % filename)


"""
Distributed training utilities.
"""


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def d_CPU(tensor):
    """
    Detach, clone and transfer tensor to CPU.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().clone().cpu()
    return tensor


def d_GPU(tensor):
    """
    Detach, clone and transfer tensor to GPU.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().clone().cuda()
    return tensor


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler=None):
    output_dir = Path(args.output_dir)
    checkpoint_paths = [output_dir / ("checkpoint-%s.pth" % str(epoch))]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if loss_scaler is not None:
            to_save["scaler"] = loss_scaler.state_dict()

        save_on_master(to_save, checkpoint_path)


def save_bert(args, epoch, text_model):
    # Create output directory if needed
    bert_dir = Path(args.output_dir) / (
        "bert-checkpoints-%s_%s" % (str(epoch), args.exp_name)
    )
    if not os.path.exists(bert_dir):
        os.makedirs(bert_dir, exist_ok=True)
    text_model.save_pretrained(
        save_directory=bert_dir,
    )


def load_model(args, model_without_ddp, optimizer, loss_scaler=None):
    if args.resume.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(args.resume, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"])
    print("Resume checkpoint %s" % args.resume)
    if (
        "optimizer" in checkpoint
        and "epoch" in checkpoint
        and not (hasattr(args, "eval") and args.eval)
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint["epoch"] + 1
        if loss_scaler is not None and "scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["scaler"])
        print("With optim & sched!")


def load_model_only(args, model_without_ddp):
    if args.resume.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(args.resume, map_location="cpu")
    model_without_ddp.load_state_dict(checkpoint["model"])
    print("Resume checkpoint %s" % args.resume)
    if "epoch" in checkpoint:
        args.start_epoch = checkpoint["epoch"] + 1


def load_model_mismatch(args, model_without_ddp):
    checkpoint = torch.load(args.resume, map_location="cpu")
    # Load state dict selectively
    # Copy projection layer
    proj_layer = checkpoint["model"]["model.proj_in.weight"]
    checkpoint["model"]["model.proj_in.weight"] = torch.cat(
        [proj_layer, proj_layer], dim=1
    )
    # Remove category embedding layer
    del checkpoint["model"]["category_emb.weight"]
    model_without_ddp.load_state_dict(checkpoint["model"])
    print("Resumed checkpoint %s" % args.resume)


def load_model_mismatch_no_ca(args, model_without_ddp):
    checkpoint = torch.load(args.resume, map_location="cpu")
    # Load state dict selectively
    # Copy projection layer
    proj_layer = checkpoint["model"]["model.proj_in.weight"]
    checkpoint["model"]["model.proj_in.weight"] = torch.cat([proj_layer] * 3, dim=1)
    model_without_ddp.load_state_dict(checkpoint["model"])
    print("Resumed checkpoint %s" % args.resume)


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def set_all_seeds(seed, use_rank=True):
    """
    Set all random seeds for reproducibility.
    """
    if use_rank:
        base_rank = get_rank()
    torch.manual_seed(seed + base_rank)
    torch.cuda.manual_seed_all(seed + base_rank)
    torch.cuda.manual_seed(seed + base_rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    print("Set seed to %d" % seed)


def dump_json(data, path, f_name):
    """
    Dump a dict to a JSON  file.
    Convert all floats to strings.
    """
    import json

    path = os.path.join(path, f_name)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True, default=str)


def normalize_trimesh(mesh):
    """
    Normalize a trimesh object to fit inside a unit cube.
    """
    # Get the bounding box
    bbox = mesh.bounding_box
    scale_factor = 1.0 / np.max(bbox.extents)
    center = bbox.centroid

    # Create a transformation matrix for centering/scaling
    translation = trimesh.transformations.translation_matrix(-center)
    scaling = trimesh.transformations.scale_matrix(scale_factor)

    # Combine the transformations
    transform = np.dot(scaling, translation)

    return mesh.apply_transform(transform)


"""
Sampling/modules utilities.
"""


def count_params(module, human_readable=True):
    """
    Count the number of parameters in a module.
    """
    total_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if human_readable:
        return "{:,}".format(total_count)
    return total_count


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def fps_subsample(pc, ratio, return_idx=False):
    """
    Subsample a point cloud using farthest point sampling.
    """
    B, N, D = pc.shape
    flattened = pc.view(B * N, D)

    batch = torch.arange(B).to(pc.device)
    batch = torch.repeat_interleave(batch, N)

    pos = flattened
    idx = fps(pos, batch, ratio=ratio)

    sampled_pc = pos[idx]
    sampled_pc = sampled_pc.view(B, -1, 3)

    if return_idx:
        return sampled_pc, idx
    return sampled_pc


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


"""
Bounding-box related utilities.
"""


def find_face_centers(triangles):
    """
    Find the center of each face in a list of triangles.
    """
    # Ensure input is a numpy array
    triangles = np.array(triangles)

    # Check input shape
    if triangles.shape[1:] != (3, 3):
        raise ValueError("Input should be of shape (N, 3, 3)")

    # Calculate pairwise distances for each triangle
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]

    dist_ab = np.linalg.norm(a - b, axis=1)
    dist_ac = np.linalg.norm(a - c, axis=1)
    dist_bc = np.linalg.norm(b - c, axis=1)

    # Find the indices of the two furthest points
    max_dist = np.maximum(dist_ab, np.maximum(dist_ac, dist_bc))

    # Calculate midpoints based on which distance is largest
    midpoints = np.where(
        max_dist[:, None] == dist_ab[:, None],
        (a + b) / 2,
        np.where(max_dist[:, None] == dist_ac[:, None], (a + c) / 2, (b + c) / 2),
    )

    return midpoints


def remove_flipped_points(points):
    """
    Remove flipped points from a list of vectors.
    """
    # Ensure points is a 2D numpy array of shape (N, 3)
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be a 2D array with shape (N, 3)")

    n = len(points)
    mask = np.ones(n, dtype=bool)

    # Compute the dot product of each point with all subsequent points
    dot_products = np.einsum("ij,kj->ik", points, points)

    # Compute the squared magnitudes
    magnitudes_squared = np.sum(points**2, axis=1)

    for i in range(n - 1):
        if not mask[i]:
            continue
        # Check if any subsequent point is a flipped version
        flipped = np.isclose(
            dot_products[i, i + 1 :], -magnitudes_squared[i], atol=1e-16
        )
        # Mark flipped points for removal
        mask[i + 1 :][flipped] = False

    return points[mask]


def flip_vecs(vecs):
    """
    Flip vectors to ensure they are all pointing y-up.
    """
    for v in vecs:
        if v[1] < 0:
            v *= -1
    return vecs


def get_bb_vecs(bb_prim):
    """
    Compute the vectors directing outwards from the centroid of a bounding box primitive.
    """
    bb_verts = bb_prim.vertices
    vert_array = np.array(bb_verts)

    # Get a list of 3 points for each face triangle
    triangles = vert_array[bb_prim.faces]
    midpoints = find_face_centers(triangles)
    face_centers = np.unique(midpoints, axis=0)

    assert face_centers.shape[0] == 6

    # Filter out duplicate vectors
    centroid = np.mean(bb_verts, axis=0)
    vecs = face_centers - centroid
    vecs = remove_flipped_points(vecs)

    # Ensure vectors are pointing in the same direction
    vecs = flip_vecs(vecs)

    assert vecs.shape[0] == 3

    return centroid, vecs
