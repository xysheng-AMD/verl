import logging
import os
import json
import time
from typing import Generator, Optional

import ray
import torch
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.device import is_support_ipc
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.atom_rollout.bucketed_weight_transfer import BucketedWeightSender
from verl.workers.rollout.atom_rollout.constants import ATOMDefaults, SleepLevel
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _atom_agent_log(event: str, **fields):
    path = os.getenv("VERL_ATOM_AGENT_LOG") or os.getenv("VERL_MEMORY_AGENT_LOG")
    if not path:
        return
    try:
        payload = {
            "tag": "ATOM_DIAG",
            "event": event,
            "ts": time.time(),
            "pid": os.getpid(),
            **fields,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    except Exception:
        pass


class ServerAdapter(BaseRollout):
    """Async rollout adapter for ATOM.

    The adapter mirrors the vLLM async rollout contract: training workers call
    resume/release/update_weights on this object, and rank 0 forwards control
    messages to the colocated ATOM server actor.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        replica_rank: int = -1,
    ):
        super().__init__(config, model_config, device_mesh)

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.replica_rank = rank // rollout_world_size if replica_rank == -1 else replica_rank
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size

        self.sleep_level = (
            SleepLevel.RELEASE_KV_CACHE_ONLY if self.config.layered_summon else ATOMDefaults.SLEEP_LEVEL
        )
        local_rank = self.rollout_rank % local_world_size
        job_id = ray.get_runtime_context().get_job_id()
        self.zmq_handle = (
            f"ipc:///tmp/rl-colocate-zmq-atom-{job_id}-replica-{self.replica_rank}-rank-{local_rank}.sock"
        )
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        self.use_shm = self._should_use_shared_memory()
        self.server_handle: Optional[ray.actor.ActorHandle] = None
        _atom_agent_log(
            "adapter_init",
            rank=rank,
            rollout_rank=self.rollout_rank,
            replica_rank=self.replica_rank,
            node_rank=self.node_rank,
            use_shm=self.use_shm,
            zmq_handle=self.zmq_handle,
        )

    def _should_use_shared_memory(self) -> bool:
        atom_kwargs = (getattr(self.config, "engine_kwargs", {}) or {}).get("atom", {}) or {}
        use_cuda_ipc = atom_kwargs.get("use_cuda_ipc")
        if use_cuda_ipc is not None:
            return not use_cuda_ipc
        return not is_support_ipc()

    def _get_server_name_prefix(self) -> str:
        return "atom_server"

    def _ensure_server_handle(self) -> bool:
        if self.rollout_rank != 0:
            return False
        if self.server_handle is None:
            prefix = self._get_server_name_prefix()
            self.server_handle = ray.get_actor(f"{prefix}_{self.replica_rank}_{self.node_rank}")
        return True

    async def _execute_method(self, method: str, *args, non_block: bool = False, **kwargs):
        if not self._ensure_server_handle():
            return None
        future = getattr(self.server_handle, method).remote(*args, **kwargs)
        return future if non_block else await future

    async def resume(self, tags: list[str]):
        if self.config.free_cache_engine:
            _atom_agent_log(
                "adapter_resume",
                rollout_rank=self.rollout_rank,
                replica_rank=self.replica_rank,
                tags=tags,
            )
            await self._execute_method("wake_up", tags=tags)

    async def release(self):
        if self.config.free_cache_engine:
            _atom_agent_log(
                "adapter_release",
                rollout_rank=self.rollout_rank,
                replica_rank=self.replica_rank,
                sleep_level=self.sleep_level,
            )
            await self._execute_method("sleep", level=self.sleep_level)

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int = None,
        **kwargs,
    ):
        """Update ATOM rollout weights through the colocated ZMQ endpoint."""
        if self.rollout_rank != 0:
            for _ in weights:
                pass
            return

        start_time = time.time()
        _atom_agent_log(
            "adapter_weight_update_start",
            rollout_rank=self.rollout_rank,
            replica_rank=self.replica_rank,
            node_rank=self.node_rank,
            global_steps=global_steps,
            use_shm=self.use_shm,
        )
        future = await self._execute_method(
            "update_weights_from_zmq",
            use_shm=self.use_shm,
            non_block=True,
            **kwargs,
        )

        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=self.config.checkpoint_engine.update_weights_bucket_megabytes,
            use_shm=self.use_shm,
        )
        try:
            await sender.async_send_weights(weights)
            _atom_agent_log(
                "adapter_weight_send_done",
                rollout_rank=self.rollout_rank,
                replica_rank=self.replica_rank,
                global_steps=global_steps,
                elapsed_s=round(time.time() - start_time, 3),
            )
        except Exception as exc:
            _atom_agent_log(
                "adapter_weight_send_error",
                rollout_rank=self.rollout_rank,
                replica_rank=self.replica_rank,
                global_steps=global_steps,
                error=repr(exc),
            )
            raise

        if future is not None:
            await future
        await self._execute_method("clear_kv_cache")
        if global_steps is not None:
            await self._execute_method("set_global_steps", global_steps)

        _atom_agent_log(
            "adapter_weight_update_done",
            rollout_rank=self.rollout_rank,
            replica_rank=self.replica_rank,
            global_steps=global_steps,
            elapsed_s=round(time.time() - start_time, 3),
        )
        if self.replica_rank == 0:
            logger.info(f"ATOM update_weights done, time cost: {time.time() - start_time:.2f}s")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        raise NotImplementedError(
            "ATOM ServerAdapter does not support synchronous generate_sequences(). "
            "Use async rollout mode via ATOMReplica and LLMServerClient."
        )
