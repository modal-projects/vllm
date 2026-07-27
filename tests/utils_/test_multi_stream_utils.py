# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
    record_stream_if_safe,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def inputs():
    x = torch.randn(64, 64, device="cuda")
    w = torch.randn(64, 64, device="cuda")
    return x, w


def test_record_stream_if_safe_ignores_non_tensors():
    stream = torch.cuda.Stream()
    tensor = torch.randn(8, device="cuda")

    record_stream_if_safe(tensor, stream)
    record_stream_if_safe((tensor, None, "not a tensor"), stream)
    record_stream_if_safe([tensor], stream)
    record_stream_if_safe(None, stream)


def test_record_stream_if_safe_is_noop_during_capture():
    """record_stream is meaningless for graph-pool blocks, so skip it."""
    stream = torch.cuda.Stream()
    tensor = torch.randn(8, device="cuda")

    # Warm up on a side stream, as CUDA graph capture requires.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        tensor.mul(2.0)
    torch.cuda.current_stream().wait_stream(side)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = tensor * 2.0
        assert torch.cuda.is_current_stream_capturing()
        record_stream_if_safe(captured, stream)

    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(captured, tensor * 2.0)


def test_maybe_execute_in_parallel_matches_sequential(inputs):
    x, w = inputs
    aux = torch.cuda.Stream()
    event0, event1 = torch.cuda.Event(), torch.cuda.Event()

    parallel = maybe_execute_in_parallel(
        lambda: x @ w, lambda: x * 2.0, event0, event1, aux
    )
    sequential = maybe_execute_in_parallel(
        lambda: x @ w, lambda: x * 2.0, event0, event1, None
    )
    torch.accelerator.synchronize()

    for got, want in zip(parallel, sequential):
        torch.testing.assert_close(got, want)


def test_execute_in_parallel_matches_sequential(inputs):
    x, w = inputs
    start = torch.cuda.Event()
    done = [torch.cuda.Event(), torch.cuda.Event()]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    fns = [lambda: x * 2.0, None]

    default, aux = execute_in_parallel(
        lambda: x @ w, fns, start, done, streams, enable=True
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(default, x @ w)
    torch.testing.assert_close(aux[0], x * 2.0)
    assert aux[1] is None
