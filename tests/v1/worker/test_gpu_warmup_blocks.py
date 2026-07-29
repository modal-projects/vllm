# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V2 warmup must reserve the KV blocks the scheduler would reserve.

`KVCacheManager.allocate_slots` sizes every allocation for
`num_computed + num_scheduled + num_lookahead_tokens`, because the speculator
writes KV past the token range the target model was scheduled for. The warmup
hand-builds its `SchedulerOutput`s, so it has to reserve the same blocks.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.config.vllm import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup, warmup_kernels

BLOCK_SIZE = 16
MAX_MODEL_LEN = 1024
NUM_SPEC_STEPS = 3

# `warmup_kernels` ends on `torch.accelerator.synchronize()`.
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="warmup synchronizes on the accelerator"
)


def _attention_group() -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        ["layer"],
        FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
    )


def _mamba_group(mamba_cache_mode: str) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        ["mamba"],
        MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode=mamba_cache_mode,
            num_speculative_blocks=NUM_SPEC_STEPS,
        ),
    )


def _make_runner(
    kv_cache_groups: list[KVCacheGroupSpec],
    num_lookahead_tokens: int,
    num_spec_steps: int = NUM_SPEC_STEPS,
) -> SimpleNamespace:
    """Stub model runner exposing only what the warmup entry points read."""
    return SimpleNamespace(
        num_speculative_steps=num_spec_steps,
        decode_query_len=num_spec_steps + 1,
        is_pooling_model=False,
        is_encoder_decoder=False,
        is_last_pp_rank=True,
        max_num_reqs=4,
        max_model_len=MAX_MODEL_LEN,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
        model_state=SimpleNamespace(max_encoder_len=0),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=2048),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=kv_cache_groups, num_blocks=1024
        ),
        vllm_config=SimpleNamespace(num_lookahead_tokens=num_lookahead_tokens),
        kv_block_zeroer=None,
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
    )


class _StepRecorder:
    """Rebuilds each request's per-group block holdings from the warmup steps."""

    def __init__(self) -> None:
        # (blocks held per group, num_computed_tokens, num_scheduled_tokens)
        self.steps: list[tuple[list[int], int, int]] = []
        self._held: dict[str, list[int]] = {}

    def execute_model(self, scheduler_output) -> None:
        for new_req in scheduler_output.scheduled_new_reqs:
            self._held[new_req.req_id] = [len(ids) for ids in new_req.block_ids]
            self._record(new_req.req_id, new_req.num_computed_tokens, scheduler_output)
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            new_block_ids = cached.new_block_ids[i]
            if new_block_ids is not None:
                self._held[req_id] = [
                    held + len(ids)
                    for held, ids in zip(self._held[req_id], new_block_ids)
                ]
            self._record(req_id, cached.num_computed_tokens[i], scheduler_output)

    def _record(self, req_id: str, num_computed: int, scheduler_output) -> None:
        self.steps.append(
            (
                list(self._held[req_id]),
                num_computed,
                scheduler_output.num_scheduled_tokens[req_id],
            )
        )

    def sample_tokens(self, grammar_output=None) -> None:
        return None


def _assert_covers_lookahead(
    steps: list[tuple[list[int], int, int]], num_lookahead_tokens: int
) -> None:
    assert steps, "warmup ran no steps"
    for num_blocks, num_computed, num_scheduled in steps:
        num_tokens = min(
            num_computed + num_scheduled + num_lookahead_tokens, MAX_MODEL_LEN
        )
        assert num_blocks[0] >= cdiv(num_tokens, BLOCK_SIZE), (
            f"{num_blocks[0]} blocks for {num_computed}+{num_scheduled} tokens "
            f"and {num_lookahead_tokens} lookahead tokens"
        )


# 0 covers eagle / MTP / draft models, 1 covers DFlash's extra in-fill query.
@pytest.mark.parametrize("extra_lookahead", [0, 1])
@pytest.mark.parametrize("num_spec_steps", [2, 3, 5, 7])
def test_warmup_kernels_reserves_lookahead_blocks(num_spec_steps, extra_lookahead):
    num_lookahead_tokens = num_spec_steps + extra_lookahead
    recorder = _StepRecorder()

    warmup_kernels(
        _make_runner([_attention_group()], num_lookahead_tokens, num_spec_steps),
        recorder.execute_model,
        recorder.sample_tokens,
    )

    _assert_covers_lookahead(recorder.steps, num_lookahead_tokens)


def test_mixed_warmup_reserves_lookahead_blocks():
    num_lookahead_tokens = NUM_SPEC_STEPS + 1
    recorder = _StepRecorder()

    assert run_mixed_prefill_decode_warmup(
        _make_runner([_attention_group()], num_lookahead_tokens),
        worker_execute_model=recorder.execute_model,
        worker_sample_tokens=recorder.sample_tokens,
        num_tokens=128,
    )

    _assert_covers_lookahead(recorder.steps, num_lookahead_tokens)


@pytest.mark.parametrize("mamba_cache_mode", ["none", "all", "align"])
def test_warmup_reserves_mamba_speculative_blocks(mamba_cache_mode):
    """Mamba groups hold the running-state block plus the speculative tail.

    `MambaManager` reserves `num_speculative_blocks` blocks past the token
    range in every cache mode, and the mamba kernels read all
    `1 + num_speculative_blocks` of those block-table columns (see
    `mamba_get_block_table_tensor`).
    """
    recorder = _StepRecorder()

    warmup_kernels(
        _make_runner(
            [_attention_group(), _mamba_group(mamba_cache_mode)], NUM_SPEC_STEPS
        ),
        recorder.execute_model,
        recorder.sample_tokens,
    )

    assert recorder.steps, "warmup ran no steps"
    for num_blocks, num_computed, num_scheduled in recorder.steps:
        # `MambaManager` drops the lookahead tokens in align mode to keep the
        # allocation block-aligned, and keeps them otherwise. Pin the token
        # range and the speculative tail separately, so an implementation that
        # returned a flat `1 + num_speculative_blocks` would fail.
        if mamba_cache_mode == "align":
            # Align mode sizes from the uncapped main-model range.
            lookahead, num_tokens = 0, num_computed + num_scheduled
        else:
            lookahead = NUM_SPEC_STEPS
            num_tokens = min(num_computed + num_scheduled + lookahead, MAX_MODEL_LEN)
        assert num_blocks[1] == cdiv(num_tokens, BLOCK_SIZE) + NUM_SPEC_STEPS, (
            f"{mamba_cache_mode}: {num_blocks[1]} mamba blocks for "
            f"{num_computed}+{num_scheduled} tokens and {lookahead} lookahead"
        )


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("eagle", NUM_SPEC_STEPS),
        ("eagle3", NUM_SPEC_STEPS),
        ("mtp", NUM_SPEC_STEPS),
        ("dspark", NUM_SPEC_STEPS),
        ("draft_model", NUM_SPEC_STEPS),
        # DFlash's in-fill decoding adds a query for the last sampled token.
        ("dflash", NUM_SPEC_STEPS + 1),
        ("ngram", 0),
        ("ngram_gpu", 0),
        ("medusa", 0),
        ("mlp_speculator", 0),
        ("suffix", 0),
        ("extract_hidden_states", 0),
    ],
)
def test_num_lookahead_tokens_per_method(method: str, expected: int):
    """`VllmConfig.num_lookahead_tokens` is the single source of the reservation.

    Both the scheduler and the warmup read it, so a wrong answer here silently
    changes how many blocks every request holds. Exercises the real property
    and the real `SpeculativeConfig` predicates; the config is built without
    `__post_init__` because the speculative methods otherwise require a draft
    model to be resolvable.
    """

    class _Config:
        num_speculative_tokens = VllmConfig.num_speculative_tokens
        num_lookahead_tokens = VllmConfig.num_lookahead_tokens

    speculative_config = object.__new__(SpeculativeConfig)
    object.__setattr__(speculative_config, "method", method)
    object.__setattr__(speculative_config, "num_speculative_tokens", NUM_SPEC_STEPS)

    config = _Config()
    config.speculative_config = speculative_config
    config.diffusion_config = None

    assert config.num_lookahead_tokens == expected


def test_num_lookahead_tokens_without_speculation():
    class _Config:
        num_speculative_tokens = VllmConfig.num_speculative_tokens
        num_lookahead_tokens = VllmConfig.num_lookahead_tokens

    config = _Config()
    config.speculative_config = None
    config.diffusion_config = None

    assert config.num_lookahead_tokens == 0
