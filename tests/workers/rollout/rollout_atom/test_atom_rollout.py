import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_atom_rollout_registries_resolve():
    from verl.workers.rollout.atom_rollout.atom_async_server import ATOMReplica
    from verl.workers.rollout.atom_rollout.atom_rollout import ServerAdapter
    from verl.workers.rollout.base import get_rollout_class
    from verl.workers.rollout.replica import get_rollout_replica_class

    assert get_rollout_class("atom", "async") is ServerAdapter
    assert get_rollout_replica_class("atom") is ATOMReplica


def _make_rollout_config(**overrides):
    values = {
        "tensor_model_parallel_size": 2,
        "data_parallel_size": 1,
        "expert_parallel_size": 1,
        "max_num_seqs": 32,
        "max_model_len": 2048,
        "max_num_batched_tokens": 4096,
        "gpu_memory_utilization": 0.7,
        "enforce_eager": True,
        "load_format": "dummy",
        "cudagraph_capture_sizes": None,
        "engine_kwargs": {"atom": {"use_cuda_ipc": True, "kv_cache_dtype": "bf16"}},
        "prompt_length": 1024,
        "response_length": 512,
        "ignore_eos": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_server(monkeypatch, rollout_config=None):
    from verl.workers.rollout.atom_rollout import atom_async_server as atom_server
    from verl.workers.rollout.replica import RolloutMode

    monkeypatch.setattr(atom_server, "omega_conf_to_dataclass", lambda cfg, **kwargs: cfg)
    monkeypatch.setattr(atom_server.ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(atom_server, "get_free_port", lambda *args, **kwargs: (12345, None))
    monkeypatch.setattr(
        atom_server.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_job_id=lambda: "job-test"),
    )

    model_config = SimpleNamespace(
        local_path="/tmp/model",
        trust_remote_code=False,
        hf_config=SimpleNamespace(),
    )
    return atom_server.ATOMHttpServer(
        config=rollout_config or _make_rollout_config(),
        model_config=model_config,
        rollout_mode=RolloutMode.HYBRID,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=2,
        nnodes=1,
        cuda_visible_devices="0,1",
    )


def test_atom_engine_kwargs_filter_verl_only_keys(monkeypatch):
    server = _make_server(
        monkeypatch,
        _make_rollout_config(
            engine_kwargs={
                "atom": {
                    "use_cuda_ipc": True,
                    "bucket_size_mb": 2048,
                    "kv_cache_dtype": "bf16",
                }
            }
        ),
    )

    kwargs = server._build_engine_kwargs()

    assert kwargs["model"] == "/tmp/model"
    assert kwargs["tensor_parallel_size"] == 2
    assert kwargs["load_dummy"] is True
    assert kwargs["kv_cache_dtype"] == "bf16"
    assert "use_cuda_ipc" not in kwargs
    assert "bucket_size_mb" not in kwargs


def test_atom_sampling_params_translate_verl_request(monkeypatch):
    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    atom_module = types.ModuleType("atom")
    sampling_module = types.ModuleType("atom.sampling_params")
    sampling_module.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "atom", atom_module)
    monkeypatch.setitem(sys.modules, "atom.sampling_params", sampling_module)

    server = _make_server(monkeypatch)
    params = server._build_sampling_params(
        {
            "max_new_tokens": 1000,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 50,
            "n": 2,
            "logprobs": True,
            "stop": ["</answer>"],
        },
        prompt_length=1800,
    )

    assert params.kwargs == {
        "max_tokens": 248,
        "temperature": 0.5,
        "logprobs": True,
        "top_k": 50,
        "top_p": 0.9,
        "n": 2,
        "ignore_eos": False,
        "stop_strings": ["</answer>"],
    }


def test_atom_reset_deferred_state_forwards_to_core_mgr(monkeypatch):
    server = _make_server(monkeypatch)
    server.engine = SimpleNamespace(core_mgr=MagicMock())

    asyncio.run(server.reset_deferred_state())

    server.engine.core_mgr.broadcast_utility_command_sync.assert_called_once_with("reset_deferred_state")
