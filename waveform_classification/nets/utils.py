"""Shared utilities for network modules.

Provides device-agnostic timing primitives for accurate performance benchmarking.
GPU/MPS operations are asynchronous: the CPU submits work and continues without
waiting.

For CUDA: Uses CUDA Events to measure GPU execution time directly. Events
timestamp on the GPU timeline, eliminating synchronization overhead from
measurements.

For MPS/CPU: Uses synchronize + perf_counter (no Events API available).

This is critical for comparing inference time across architectures (MLP, CNN,
PIN, Spectral) where differences of 0.01-0.05ms are significant.

References:
    How to Accurately Time CUDA Kernels in PyTorch:
    https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch

    PyTorch Benchmark Tutorial:
    https://pytorch.org/tutorials/recipes/recipes/benchmark.html
"""

import time
from collections.abc import Callable

import torch


def synchronize(device: torch.device) -> None:
    """Synchronize accelerator to ensure all queued operations complete.

    GPU/MPS operations are asynchronous: the CPU queues work and returns
    immediately. Without synchronization, CPU-side timing would only measure
    the time to queue operations, not execute them.

    This function blocks until all previously queued operations on the device
    complete, enabling accurate wall-clock timing measurements.

    Args:
        device: The device to synchronize (cuda, mps, or cpu)

    Note:
        No-op on CPU devices where operations are synchronous by default.
    """
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _timed_forward_cuda[T](fn: Callable[[], T], device: torch.device) -> tuple[T, float]:
    """CUDA Events timing - measures GPU execution time only.

    CUDA Events timestamp on the GPU timeline, eliminating sync overhead from
    measurements. This is the correct approach for sub-millisecond precision.

    Uses device context to ensure events are created/recorded on the correct GPU
    in multi-GPU setups.
    """
    with torch.cuda.device(device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000  # ms → s


def timed_forward[T](fn: Callable[[], T], device: torch.device) -> tuple[T, float]:
    """Execute fn with accurate timing on accelerators.

    For CUDA: Uses CUDA Events to measure GPU execution time directly, avoiding
    sync overhead in the measurement.

    For MPS/CPU: Uses synchronize + perf_counter (no alternative available).

    Args:
        fn: Zero-argument callable to time (typically lambda: model(x))
        device: Device tensor resides on (determines timing method)

    Returns:
        Tuple of (fn result, elapsed time in seconds)

    Example:
        logits, elapsed = timed_forward(lambda: model(x), x.device)
        time_per_sample = elapsed / x.shape[0]  # seconds per sample
    """
    if device.type == "cuda":
        return _timed_forward_cuda(fn, device)
    # MPS/CPU: synchronize + perf_counter
    synchronize(device)
    t0 = time.perf_counter()
    result = fn()
    synchronize(device)
    return result, time.perf_counter() - t0
