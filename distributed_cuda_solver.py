
import os
import sys
import json
import math
import hashlib
import subprocess
import time
import numpy as np

from mpi4py import MPI
from numba import cuda, float64

# ============================================================
# GLOBAL CONSTANTS
# ============================================================

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2

GLOBAL_NX = 512
GLOBAL_NY = 512

LOCAL_NX = 256
LOCAL_NY = 256

STEPS = 400
LAMBDA_DAMP = 0.15

DX = 1.0 / 513.0
DY = 1.0 / 513.0

DT = 0.000001

# ============================================================
# GPU UUID
# ============================================================

def get_gpu_uuid() -> str:

    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader"
        ],
        encoding="utf-8"
    )

    return out.strip().splitlines()[0].strip()

# ============================================================
# STUDENT SEED
# ============================================================

def derive_student_seed(github_user: str, gpu_uuid: str) -> int:

    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")
    ).hexdigest()

    return int(digest[:16], 16)

# ============================================================
# TASK 1: DISTRIBUTED MPI TOPOLOGY
# ============================================================

def init_distributed_context(github_user: str):

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 4:
        raise RuntimeError(
            f"Expected exactly 4 MPI ranks, got {size}"
        )

    # Bind every MPI process to GPU 0
    cuda.select_device(0)

    # Obtain GPU UUID on rank 0
    gpu_uuid = get_gpu_uuid() if rank == 0 else None

    # Share UUID with all MPI processes
    gpu_uuid = comm.bcast(gpu_uuid, root=0)

    # Derive hardware-bound seed
    instance_seed = derive_student_seed(
        github_user,
        gpu_uuid
    )

    # Create non-periodic 2x2 Cartesian topology
    dims = [2, 2]
    periods = [False, False]

    cart_comm = comm.Create_cart(
        dims=dims,
        periods=periods,
        reorder=False
    )

    # Cartesian coordinates
    coords = cart_comm.Get_coords(rank)

    row_rank = coords[0]
    col_rank = coords[1]

    # Neighbors along vertical axis
    nbr_up, nbr_down = cart_comm.Shift(0, 1)

    # Neighbors along horizontal axis
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    if rank == 0:

        print("=" * 60)
        print("DISTRIBUTED CUDA STENCIL SOLVER")
        print("=" * 60)

        print(f"GitHub username: {github_user}")
        print(f"GPU UUID: {gpu_uuid}")
        print(f"INSTANCE_SEED: {instance_seed}")
        print(f"MPI processes: {size}")
        print(f"Cartesian grid: {dims}")
        print("=" * 60)

    return {
        "comm": comm,
        "cart_comm": cart_comm,
        "rank": rank,
        "size": size,
        "coords": coords,
        "row_rank": row_rank,
        "col_rank": col_rank,
        "nbr_up": nbr_up,
        "nbr_down": nbr_down,
        "nbr_left": nbr_left,
        "nbr_right": nbr_right,
        "gpu_uuid": gpu_uuid,
        "instance_seed": instance_seed
    }

# ============================================================
# TASK 2: SEEDED HETEROGENEOUS SUBDOMAIN INITIALIZATION
# ============================================================

def initialize_subdomains(ctx):

    rank = ctx["rank"]
    row_rank = ctx["row_rank"]
    col_rank = ctx["col_rank"]

    # Reconstruct the same deterministic random parameters
    # on every MPI rank.
    rng = np.random.default_rng(ctx["instance_seed"])

    num_pulses = 3

    pulse_centers = rng.uniform(
        0.15, 0.85, size=(num_pulses, 2)
    )

    pulse_amplitudes = rng.uniform(
        0.5, 1.0, size=num_pulses
    )

    pulse_widths = rng.uniform(
        0.035, 0.075, size=num_pulses
    )

    source_center = rng.uniform(0.2, 0.8, size=2)

    # Local padded grid coordinates.
    # Interior indices are 1..256.
    # Ghost indices are 0 and 257.

    global_x_indices = (
        col_rank * LOCAL_NX
        + np.arange(-1, LOCAL_NX + 1)
    )

    global_y_indices = (
        row_rank * LOCAL_NY
        + np.arange(-1, LOCAL_NY + 1)
    )

    x = (global_x_indices + 1) * DX
    y = (global_y_indices + 1) * DY

    X, Y = np.meshgrid(x, y)

    # --------------------------------------------------------
    # Heterogeneous diffusivity field: 0.08 <= kappa <= 0.24
    # --------------------------------------------------------

    kappa_host = (
        0.16
        + 0.08 * np.sin(2.0 * np.pi * X)
        * np.cos(2.0 * np.pi * Y)
    )

    kappa_host = np.clip(
        kappa_host, 0.08, 0.24
    ).astype(np.float64)

    # --------------------------------------------------------
    # Initial Gaussian thermal pulses
    # --------------------------------------------------------

    u_host = np.zeros(
        (LOCAL_NY + 2, LOCAL_NX + 2),
        dtype=np.float64
    )

    for p in range(num_pulses):

        cx, cy = pulse_centers[p]
        amplitude = pulse_amplitudes[p]
        sigma = pulse_widths[p]

        gaussian = amplitude * np.exp(
            -(
                (X - cx) ** 2
                + (Y - cy) ** 2
            ) / (2.0 * sigma ** 2)
        )

        u_host += gaussian

    # --------------------------------------------------------
    # Localized source field
    # --------------------------------------------------------

    sx, sy = source_center

    source_host = (
        0.05 * np.exp(
            -(
                (X - sx) ** 2
                + (Y - sy) ** 2
            ) / (2.0 * 0.04 ** 2)
        )
    ).astype(np.float64)

    # --------------------------------------------------------
    # Homogeneous Dirichlet physical boundaries.
    # PROC_NULL identifies the global outer boundaries.
    # --------------------------------------------------------

    if ctx["nbr_up"] == MPI.PROC_NULL:
        u_host[0, :] = 0.0

    if ctx["nbr_down"] == MPI.PROC_NULL:
        u_host[-1, :] = 0.0

    if ctx["nbr_left"] == MPI.PROC_NULL:
        u_host[:, 0] = 0.0

    if ctx["nbr_right"] == MPI.PROC_NULL:
        u_host[:, -1] = 0.0

    # --------------------------------------------------------
    # Transfer local arrays to GPU
    # --------------------------------------------------------

    u_device = cuda.to_device(u_host)
    kappa_device = cuda.to_device(kappa_host)
    source_device = cuda.to_device(source_host)

    if rank == 0:

        print("\nTASK 2: INITIALIZATION")
        print("-" * 45)

        print("Local array shape:", u_host.shape)
        print("Kappa minimum:", float(kappa_host.min()))
        print("Kappa maximum:", float(kappa_host.max()))
        print("Thermal pulses:", num_pulses)
        print("Source field initialized: True")

        print("-" * 45)

    return {
        "u": u_device,
        "kappa": kappa_device,
        "source": source_device
    }

# ============================================================
# TASK 3: GPU HALO PACK / UNPACK KERNELS
# ============================================================

@cuda.jit
def pack_halos_kernel(u, top, bottom, left, right, n):

    idx = cuda.grid(1)

    if idx < n:

        # Pack top and bottom interior rows
        top[idx] = u[1, idx + 1]
        bottom[idx] = u[n, idx + 1]

        # Pack left and right interior columns
        left[idx] = u[idx + 1, 1]
        right[idx] = u[idx + 1, n]


@cuda.jit
def unpack_halos_kernel(
    u, top, bottom, left, right,
    n, has_top, has_bottom, has_left, has_right
):

    idx = cuda.grid(1)

    if idx < n:

        # Unpack top ghost row
        if has_top:
            u[0, idx + 1] = top[idx]

        # Unpack bottom ghost row
        if has_bottom:
            u[n + 1, idx + 1] = bottom[idx]

        # Unpack left ghost column
        if has_left:
            u[idx + 1, 0] = left[idx]

        # Unpack right ghost column
        if has_right:
            u[idx + 1, n + 1] = right[idx]

# ============================================================
# TASK 3: HALO BUFFER ALLOCATION
# ============================================================

def create_halo_buffers():

    n = LOCAL_NX

    # Device send buffers
    device_send = {
        "top": cuda.device_array(n, dtype=np.float64),
        "bottom": cuda.device_array(n, dtype=np.float64),
        "left": cuda.device_array(n, dtype=np.float64),
        "right": cuda.device_array(n, dtype=np.float64)
    }

    # Device receive buffers
    device_recv = {
        "top": cuda.device_array(n, dtype=np.float64),
        "bottom": cuda.device_array(n, dtype=np.float64),
        "left": cuda.device_array(n, dtype=np.float64),
        "right": cuda.device_array(n, dtype=np.float64)
    }

    # Page-locked host send buffers
    host_send = {
        name: cuda.pinned_array(n, dtype=np.float64)
        for name in ("top", "bottom", "left", "right")
    }

    # Page-locked host receive buffers
    host_recv = {
        name: cuda.pinned_array(n, dtype=np.float64)
        for name in ("top", "bottom", "left", "right")
    }

    return {
        "device_send": device_send,
        "device_recv": device_recv,
        "host_send": host_send,
        "host_recv": host_recv
    }

# ============================================================
# TASK 3: ASYNCHRONOUS MPI HALO EXCHANGE
# ============================================================

def exchange_halos_async(
    u_device,
    ctx,
    buffers,
    stream_boundary=None
):

    if stream_boundary is None:
        stream_boundary = cuda.stream()

    cart_comm = ctx["cart_comm"]

    nbr_up = ctx["nbr_up"]
    nbr_down = ctx["nbr_down"]
    nbr_left = ctx["nbr_left"]
    nbr_right = ctx["nbr_right"]

    n = LOCAL_NX

    device_send = buffers["device_send"]
    device_recv = buffers["device_recv"]

    host_send = buffers["host_send"]
    host_recv = buffers["host_recv"]

    threads = 256
    blocks = (n + threads - 1) // threads

    # --------------------------------------------------------
    # Stage A: GPU halo packing
    # --------------------------------------------------------

    pack_halos_kernel[blocks, threads, stream_boundary](
        u_device,
        device_send["top"],
        device_send["bottom"],
        device_send["left"],
        device_send["right"],
        n
    )

    # Asynchronous Device-to-Host transfers
    for direction in ("top", "bottom", "left", "right"):

        device_send[direction].copy_to_host(
            host_send[direction],
            stream=stream_boundary
        )

    # Ensure pinned send buffers are ready before MPI reads them
    stream_boundary.synchronize()

    # --------------------------------------------------------
    # Stage B: Non-blocking MPI communication
    # --------------------------------------------------------

    requests = []

    # Receive halos from neighbors.
    # Tags identify the direction in which the sender sent data.

    requests.append(
        cart_comm.Irecv(
            host_recv["top"],
            source=nbr_up,
            tag=11
        )
    )

    requests.append(
        cart_comm.Irecv(
            host_recv["bottom"],
            source=nbr_down,
            tag=10
        )
    )

    requests.append(
        cart_comm.Irecv(
            host_recv["left"],
            source=nbr_left,
            tag=13
        )
    )

    requests.append(
        cart_comm.Irecv(
            host_recv["right"],
            source=nbr_right,
            tag=12
        )
    )

    # Send local boundary data to neighboring ranks

    requests.append(
        cart_comm.Isend(
            host_send["top"],
            dest=nbr_up,
            tag=10
        )
    )

    requests.append(
        cart_comm.Isend(
            host_send["bottom"],
            dest=nbr_down,
            tag=11
        )
    )

    requests.append(
        cart_comm.Isend(
            host_send["left"],
            dest=nbr_left,
            tag=12
        )
    )

    requests.append(
        cart_comm.Isend(
            host_send["right"],
            dest=nbr_right,
            tag=13
        )
    )

    # Complete all MPI transfers
    MPI.Request.Waitall(requests)

    # --------------------------------------------------------
    # Stage C: Host-to-Device transfers
    # --------------------------------------------------------

    neighbor_exists = {
        "top": nbr_up != MPI.PROC_NULL,
        "bottom": nbr_down != MPI.PROC_NULL,
        "left": nbr_left != MPI.PROC_NULL,
        "right": nbr_right != MPI.PROC_NULL
    }

    for direction in ("top", "bottom", "left", "right"):

        if neighbor_exists[direction]:

            device_recv[direction].copy_to_device(
                host_recv[direction],
                stream=stream_boundary
            )

    # --------------------------------------------------------
    # GPU halo unpacking
    # --------------------------------------------------------

    unpack_halos_kernel[blocks, threads, stream_boundary](
        u_device,
        device_recv["top"],
        device_recv["bottom"],
        device_recv["left"],
        device_recv["right"],
        n,
        neighbor_exists["top"],
        neighbor_exists["bottom"],
        neighbor_exists["left"],
        neighbor_exists["right"]
    )

    stream_boundary.synchronize()

# ============================================================
# TASK 4: 2D SHARED-MEMORY TILED STENCIL
# ============================================================

@cuda.jit
def tiled_stencil_kernel(
    u, kappa, source, output,
    r_start, r_end,
    c_start, c_end,
    boundary_only
):

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Each block covers a 16x16 interior region.
    row = cuda.blockIdx.y * TILE_DIM + ty + 1
    col = cuda.blockIdx.x * TILE_DIM + tx + 1

    # Shared-memory tiles: 18x18 including halo apron.
    smem_u = cuda.shared.array(
        shape=(SHARED_DIM, SHARED_DIM),
        dtype=float64
    )

    smem_k = cuda.shared.array(
        shape=(SHARED_DIM, SHARED_DIM),
        dtype=float64
    )

    # --------------------------------------------------------
    # Cooperative loading of the 18x18 shared-memory tiles
    # --------------------------------------------------------

    tid = ty * TILE_DIM + tx

    for index in range(0, SHARED_DIM * SHARED_DIM, TILE_DIM * TILE_DIM):

        pos = tid + index

        if pos < SHARED_DIM * SHARED_DIM:

            sy = pos // SHARED_DIM
            sx = pos % SHARED_DIM

            global_row = cuda.blockIdx.y * TILE_DIM + sy
            global_col = cuda.blockIdx.x * TILE_DIM + sx

            smem_u[sy, sx] = u[global_row, global_col]
            smem_k[sy, sx] = kappa[global_row, global_col]

    # All threads must finish loading before stencil evaluation.
    cuda.syncthreads()

    # --------------------------------------------------------
    # Determine whether this cell belongs to the requested area
    # --------------------------------------------------------

    inside_box = (
        row >= r_start and row <= r_end
        and col >= c_start and col <= c_end
    )

    if inside_box and row <= LOCAL_NY and col <= LOCAL_NX:

        is_boundary = (
            row == 1 or row == LOCAL_NY
            or col == 1 or col == LOCAL_NX
        )

        if boundary_only:
            should_update = is_boundary
        else:
            should_update = not is_boundary

        if should_update:

            center = smem_u[ty + 1, tx + 1]

            north = smem_u[ty, tx + 1]
            south = smem_u[ty + 2, tx + 1]
            west = smem_u[ty + 1, tx]
            east = smem_u[ty + 1, tx + 2]

            k_center = smem_k[ty + 1, tx + 1]

            k_north = 0.5 * (
                k_center + smem_k[ty, tx + 1]
            )

            k_south = 0.5 * (
                k_center + smem_k[ty + 2, tx + 1]
            )

            k_west = 0.5 * (
                k_center + smem_k[ty + 1, tx]
            )

            k_east = 0.5 * (
                k_center + smem_k[ty + 1, tx + 2]
            )

            diffusion_x = (
                k_east * (east - center)
                - k_west * (center - west)
            ) / (DX * DX)

            diffusion_y = (
                k_south * (south - center)
                - k_north * (center - north)
            ) / (DY * DY)

            reaction = LAMBDA_DAMP * center ** 3

            result = center + DT * (
                diffusion_x
                + diffusion_y
                - reaction
                + source[row, col]
            )

            output[row, col] = result

# ============================================================
# TASK 4: DUAL-STREAM OVERLAPPED STEP
# ============================================================

def step_overlapped(
    u_device,
    kappa_device,
    source_device,
    output_device,
    ctx,
    buffers,
    stream_boundary,
    stream_interior
):

    cart_comm = ctx["cart_comm"]

    nbr_up = ctx["nbr_up"]
    nbr_down = ctx["nbr_down"]
    nbr_left = ctx["nbr_left"]
    nbr_right = ctx["nbr_right"]

    n = LOCAL_NX

    threads = 256
    blocks_1d = (n + threads - 1) // threads

    device_send = buffers["device_send"]
    device_recv = buffers["device_recv"]

    host_send = buffers["host_send"]
    host_recv = buffers["host_recv"]

    # --------------------------------------------------------
    # STAGE A: Pack halos and start asynchronous D2H transfers
    # --------------------------------------------------------

    pack_halos_kernel[blocks_1d, threads, stream_boundary](
        u_device,
        device_send["top"],
        device_send["bottom"],
        device_send["left"],
        device_send["right"],
        n
    )

    for direction in ("top", "bottom", "left", "right"):

        device_send[direction].copy_to_host(
            host_send[direction],
            stream=stream_boundary
        )

    stream_boundary.synchronize()

    # --------------------------------------------------------
    # STAGE B: Post non-blocking MPI transfers
    # --------------------------------------------------------

    requests = []

    requests.append(
        cart_comm.Irecv(
            host_recv["top"],
            source=nbr_up,
            tag=11
        )
    )

    requests.append(
        cart_comm.Irecv(
            host_recv["bottom"],
            source=nbr_down,
            tag=10
        )
    )

    requests.append(
        cart_comm.Irecv(
            host_recv["left"],
            source=nbr_left,
            tag=13
        )
    )

    requests.append(
        cart_comm.Irecv(
            host_recv["right"],
            source=nbr_right,
            tag=12
        )
    )

    requests.append(
        cart_comm.Isend(
            host_send["top"],
            dest=nbr_up,
            tag=10
        )
    )

    requests.append(
        cart_comm.Isend(
            host_send["bottom"],
            dest=nbr_down,
            tag=11
        )
    )

    requests.append(
        cart_comm.Isend(
            host_send["left"],
            dest=nbr_left,
            tag=12
        )
    )

    requests.append(
        cart_comm.Isend(
            host_send["right"],
            dest=nbr_right,
            tag=13
        )
    )

    # --------------------------------------------------------
    # Launch deep-interior computation while MPI is active
    # --------------------------------------------------------

    threads_2d = (TILE_DIM, TILE_DIM)
    blocks_2d = (16, 16)

    tiled_stencil_kernel[blocks_2d, threads_2d, stream_interior](
        u_device,
        kappa_device,
        source_device,
        output_device,
        2, LOCAL_NY - 1,
        2, LOCAL_NX - 1,
        False
    )

    # --------------------------------------------------------
    # STAGE C: Complete MPI and transfer received halos
    # --------------------------------------------------------

    MPI.Request.Waitall(requests)

    neighbor_exists = {
        "top": nbr_up != MPI.PROC_NULL,
        "bottom": nbr_down != MPI.PROC_NULL,
        "left": nbr_left != MPI.PROC_NULL,
        "right": nbr_right != MPI.PROC_NULL
    }

    for direction in ("top", "bottom", "left", "right"):

        if neighbor_exists[direction]:

            device_recv[direction].copy_to_device(
                host_recv[direction],
                stream=stream_boundary
            )

    unpack_halos_kernel[blocks_1d, threads, stream_boundary](
        u_device,
        device_recv["top"],
        device_recv["bottom"],
        device_recv["left"],
        device_recv["right"],
        n,
        neighbor_exists["top"],
        neighbor_exists["bottom"],
        neighbor_exists["left"],
        neighbor_exists["right"]
    )

    # --------------------------------------------------------
    # Calculate the outer perimeter ring
    # --------------------------------------------------------

    tiled_stencil_kernel[blocks_2d, threads_2d, stream_boundary](
        u_device,
        kappa_device,
        source_device,
        output_device,
        1, LOCAL_NY,
        1, LOCAL_NX,
        True
    )

    # Wait for both streams before returning output.
    stream_interior.synchronize()
    stream_boundary.synchronize()

    return output_device

# ============================================================
# TASK 5: Distributed reductions, PTX audit and benchmark
# ============================================================

def run_one_mode(ctx, mode):
    """Run 400 time steps using serialized or overlapped communication."""
    rank = ctx["rank"]
    cart_comm = ctx["cart_comm"]

    # Reinitialize the same deterministic initial condition for each mode.
    fields = initialize_subdomains(ctx)
    u = fields["u"]
    kappa = fields["kappa"]
    source = fields["source"]

    output = cuda.device_array(
        (LOCAL_NY + 2, LOCAL_NX + 2), dtype=np.float64
    )

    buffers = create_halo_buffers()
    stream_boundary = cuda.stream()
    stream_interior = cuda.stream()

    energy_trace = []
    residual_trace = []

    def record_metrics(step, old_u, new_u):
        old_host = old_u.copy_to_host()
        new_host = new_u.copy_to_host()

        owned = new_host[1:-1, 1:-1]
        old_owned = old_host[1:-1, 1:-1]

        local_energy = float(np.sum(owned) * DX * DY)
        local_residual_sq = float(np.sum((owned - old_owned) ** 2))

        global_energy = cart_comm.allreduce(
            local_energy, op=MPI.SUM
        )
        global_residual_sq = cart_comm.allreduce(
            local_residual_sq, op=MPI.SUM
        )

        energy_trace.append((step, global_energy))
        residual_trace.append((step, math.sqrt(global_residual_sq)))

        if rank == 0:
            print(
                f"{mode} | step {step:3d} | "
                f"energy={global_energy:.16e} | "
                f"L2 residual={math.sqrt(global_residual_sq):.16e}"
            )

    # Initial energy and zero initial residual.
    record_metrics(0, u, u)

    cart_comm.Barrier()
    start_time = MPI.Wtime()

    for step in range(1, STEPS + 1):
        old_u = u

        if mode == "serialized":
            exchange_halos_async(
                u, ctx, buffers, stream_boundary
            )

            tiled_stencil_kernel[
                ((LOCAL_NX + 15) // 16, (LOCAL_NY + 15) // 16),
                (16, 16)
            ](
                u, kappa, source, output,
                1, LOCAL_NY + 1,
                1, LOCAL_NX + 1,
                False
            )
            cuda.synchronize()

        elif mode == "overlapped":
            step_overlapped(
                u, kappa, source, output,
                ctx, buffers,
                stream_boundary, stream_interior
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        u = output
        output = old_u

        if step % 100 == 0:
            record_metrics(step, old_u, u)

    cart_comm.Barrier()
    elapsed = MPI.Wtime() - start_time
    max_elapsed = cart_comm.allreduce(elapsed, op=MPI.MAX)

    return {
        "u": u,
        "energy_trace": energy_trace,
        "residual_trace": residual_trace,
        "elapsed_ms": max_elapsed * 1000.0,
        "kappa_max": float(np.max(kappa.copy_to_host()))
    }


def audit_ptx():
    """Compile the tiled kernel and count required PTX instructions."""
    dummy = cuda.to_device(
        np.zeros((LOCAL_NY + 2, LOCAL_NX + 2), dtype=np.float64)
    )
    dummy_out = cuda.device_array_like(dummy)

    tiled_stencil_kernel[
        ((LOCAL_NX + 15) // 16, (LOCAL_NY + 15) // 16),
        (16, 16)
    ](
        dummy, dummy, dummy, dummy_out,
        1, LOCAL_NY + 1, 1, LOCAL_NX + 1, False
    )
    cuda.synchronize()

    asm_result = tiled_stencil_kernel.inspect_asm()
    if isinstance(asm_result, dict):
        ptx_text = "\n".join(str(v) for v in asm_result.values())
    else:
        ptx_text = str(asm_result)

    counts = {
        "st.shared.f64": len(
            __import__("re").findall(r"\bst\.shared\.f64\b", ptx_text)
        ),
        "ld.shared.f64": len(
            __import__("re").findall(r"\bld\.shared\.f64\b", ptx_text)
        ),
        "bar.sync": len(
            __import__("re").findall(r"\bbar\.sync\b", ptx_text)
        )
    }

    passed = all(value > 0 for value in counts.values())

    print("\nPTX ASSEMBLY AUDIT")
    for instruction, count in counts.items():
        print(f"{instruction}: {count}")
    print(f"PTX audit passed: {passed}")

    return counts, passed


def run_solver(github_user):
    ctx = init_distributed_context(github_user)
    rank = ctx["rank"]
    comm = ctx["cart_comm"]

    if rank == 0:
        print("\nStarting serialized benchmark...")
    serialized = run_one_mode(ctx, "serialized")

    if rank == 0:
        print("\nStarting overlapped benchmark...")
    overlapped = run_one_mode(ctx, "overlapped")

    # Compare the final local subdomains on every rank.
    serialized_host = serialized["u"].copy_to_host()
    overlapped_host = overlapped["u"].copy_to_host()

    local_max_diff = float(
        np.max(np.abs(serialized_host - overlapped_host))
    )
    parity_max_diff = comm.allreduce(local_max_diff, op=MPI.MAX)
    parity_passed = parity_max_diff < 1e-12

    ptx_counts, ptx_passed = audit_ptx()

    if rank == 0:
        print("\nBENCHMARK SUMMARY")
        print(f"Serialized time: {serialized['elapsed_ms']:.3f} ms")
        print(f"Overlapped time: {overlapped['elapsed_ms']:.3f} ms")
        print(f"Parity max difference: {parity_max_diff:.16e}")
        print(f"Parity passed: {parity_passed}")

        manifest = {
            "github_username": github_user,
            "mpi_ranks": ctx["size"],
            "cartesian_grid": [2, 2],
            "gpu_uuid": ctx["gpu_uuid"],
            "instance_seed": int(ctx["instance_seed"]),
            "steps": STEPS,
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "local_grid": [LOCAL_NY, LOCAL_NX],
            "serialized_ms": serialized["elapsed_ms"],
            "overlapped_ms": overlapped["elapsed_ms"],
            "parity_max_diff": parity_max_diff,
            "parity_passed": parity_passed,
            "ptx_counts": ptx_counts,
            "ptx_audit_passed": ptx_passed,
            "kappa_max": serialized["kappa_max"],
            "dt": DT,
            "dx": DX,
            "energy_trace": serialized["energy_trace"],
            "residual_trace": serialized["residual_trace"]
        }

        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=4)

        print("\nManifest written: submission_manifest.json")

        if not parity_passed:
            print("WARNING: serialized and overlapped results do not match.")
        if not ptx_passed:
            print("WARNING: PTX audit did not pass.")

    return serialized, overlapped


if __name__ == "__main__":
    GITHUB_USERNAME = os.environ.get(
        "GITHUB_USERNAME", "230103225-lgtm"
    )
    run_solver(GITHUB_USERNAME)
