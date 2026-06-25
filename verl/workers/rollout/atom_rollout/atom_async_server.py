import asyncio
import gc
import json
import logging
import os
import time
from multiprocessing import shared_memory
from typing import Any
from uuid import uuid4

import ray
import torch
import zmq
from ray.actor import ActorHandle

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_resource_name, get_visible_devices_keyword
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.atom_rollout.constants import ATOMDefaults, IPCConfig, SleepLevel
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import get_max_position_embeddings, run_uvicorn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


class ATOMHttpServer:
    """Ray actor hosting an ATOM async engine plus VERL-compatible HTTP/RPC APIs."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        visible_devices_key = get_visible_devices_keyword()
        os.environ[visible_devices_key] = cuda_visible_devices
        if visible_devices_key == "CUDA_VISIBLE_DEVICES":
            # Ray's ROCm accelerator manager may inject HIP/ROCR_VISIBLE_DEVICES.
            # ATOM follows vLLM-style CUDA_VISIBLE_DEVICES, so keep one mask.
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        if self.config.max_model_len is None or self.config.max_model_len <= 0:
            self.config.max_model_len = get_max_position_embeddings(self.model_config.hf_config)

        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        self.job_id = ray.get_runtime_context().get_job_id()
        self.engine = None
        self.global_steps = None

        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None
        self._server_task = None
        self._master_address = None
        self._master_port = None
        self._master_sock = None
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address, with_alive_sock=True)

        self._batch_size, self._batch_timeout = self._compute_batch_params()
        self._pending_requests: list[tuple] = []
        self._batch_lock = asyncio.Lock()
        self._batch_event = asyncio.Event()
        self._batch_processor_task = None

        logger.info(
            "ATOMHttpServer replica=%s node=%s %s=%s master=%s:%s",
            self.replica_rank,
            self.node_rank,
            visible_devices_key,
            cuda_visible_devices,
            self._master_address,
            self._master_port,
        )
        _atom_agent_log(
            "server_init",
            replica_rank=self.replica_rank,
            node_rank=self.node_rank,
            visible_devices_key=visible_devices_key,
            visible_devices=cuda_visible_devices,
            batch_size=self._batch_size,
            batch_timeout=self._batch_timeout,
        )

    def _compute_batch_params(self) -> tuple[int, float]:
        batch_size = max(1, self.config.data_parallel_size * self.config.max_num_seqs)
        return batch_size, ATOMDefaults.BATCH_TIMEOUT

    def get_master_address(self):
        return self._master_address, self._master_port

    def get_server_address(self):
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            self._master_address = master_address
            self._master_port = master_port

        from atom.rollout.async_engine import AsyncLLMEngine

        # Release the reservation socket before torch.distributed binds the
        # selected master port. get_free_port(..., with_alive_sock=True) keeps
        # the port unique across Ray actors until this point.
        if self._master_sock is not None:
            self._master_sock.close()
            self._master_sock = None

        self.engine = AsyncLLMEngine(**self._build_engine_kwargs())
        logger.info("ATOMHttpServer: AsyncLLMEngine created")
        _atom_agent_log(
            "engine_created",
            replica_rank=self.replica_rank,
            node_rank=self.node_rank,
            master_address=self._master_address,
            master_port=self._master_port,
        )

        if self.node_rank == 0:
            await self._launch_http_server()

    def _build_engine_kwargs(self) -> dict[str, Any]:
        engine_kwargs = {
            "model": self.model_config.local_path,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "data_parallel_size": self.config.data_parallel_size,
            "enable_expert_parallel": self.config.expert_parallel_size > 1,
            "max_num_seqs": self.config.max_num_seqs,
            "max_model_len": self.config.max_model_len,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "enforce_eager": self.config.enforce_eager,
            "trust_remote_code": self.model_config.trust_remote_code,
            "load_dummy": self.config.load_format == "dummy",
        }

        if self.config.max_num_batched_tokens:
            engine_kwargs["max_num_batched_tokens"] = self.config.max_num_batched_tokens
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["compilation_config"] = {"cudagraph_capture_sizes": self.config.cudagraph_capture_sizes}

        atom_kwargs = dict((getattr(self.config, "engine_kwargs", {}) or {}).get("atom", {}) or {})
        for verl_only_key in ("bucket_size_mb", "use_cuda_ipc"):
            atom_kwargs.pop(verl_only_key, None)
        engine_kwargs.update(atom_kwargs)
        return engine_kwargs

    def _build_sampling_params(self, sampling_params: dict[str, Any], prompt_length: int):
        from atom.sampling_params import SamplingParams

        sampling_params = dict(sampling_params)
        max_possible_tokens = self.config.max_model_len - prompt_length
        if max_possible_tokens < 1:
            raise ValueError(
                f"Prompt length ({prompt_length}) leaves no room to generate within max_model_len "
                f"({self.config.max_model_len})."
            )

        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", None))
        if max_tokens is None:
            max_tokens = min(
                self.config.response_length,
                self.config.prompt_length + self.config.response_length - prompt_length,
            )
        max_tokens = max(1, min(int(max_tokens), max_possible_tokens))

        temperature = sampling_params.pop("temperature", ATOMDefaults.TEMPERATURE)
        return_logprobs = sampling_params.pop("logprobs", False)
        top_k = sampling_params.pop("top_k", -1)
        top_p = sampling_params.pop("top_p", 1.0)
        n = sampling_params.pop("n", 1)
        ignore_eos = sampling_params.pop("ignore_eos", self.config.ignore_eos)
        stop_strings = sampling_params.pop("stop_strings", sampling_params.pop("stop", None))

        if sampling_params.pop("do_sample", None) is False:
            temperature = 0.0
            top_k = -1
            top_p = 1.0

        for key in ("repetition_penalty", "stop_token_ids", "min_tokens"):
            if key in sampling_params:
                logger.warning(
                    "ATOM rollout does not support sampling param %s=%r; dropping it.",
                    key,
                    sampling_params.pop(key),
                )
        if sampling_params:
            logger.warning("ATOM rollout dropping unsupported sampling params: %s", sorted(sampling_params.keys()))

        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=return_logprobs,
            top_k=top_k,
            top_p=top_p,
            n=n,
            ignore_eos=ignore_eos,
            stop_strings=stop_strings,
        )

    async def _launch_http_server(self):
        import time as _time

        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field

        app = FastAPI(title="ATOM verl Server")

        class GenerateRequest(BaseModel):
            prompt_ids: list[int]
            sampling_params: dict[str, Any] = Field(default_factory=dict)
            request_id: str = ""

        class ChatMessage(BaseModel):
            role: str
            content: str

        class ChatCompletionRequest(BaseModel):
            model: str
            messages: list[ChatMessage]
            temperature: float = 1.0
            top_p: float = 1.0
            n: int = 1
            max_tokens: int | None = None
            logprobs: bool = False

        class ScoreRequest(BaseModel):
            model: str
            input: str
            tokenize: bool = True

        @app.post("/generate")
        async def generate_endpoint(request: GenerateRequest):
            try:
                result = await self.generate(
                    prompt_ids=request.prompt_ids,
                    sampling_params=request.sampling_params,
                    request_id=request.request_id or str(uuid4()),
                )
                return result.model_dump()
            except Exception as exc:
                logger.error(f"Generate error: {exc}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            try:
                tokenizer = self.model_config.tokenizer
                messages = [{"role": m.role, "content": m.content} for m in request.messages]
                prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
                sampling_params = {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "n": request.n,
                    "logprobs": request.logprobs,
                }
                if request.max_tokens is not None:
                    sampling_params["max_tokens"] = request.max_tokens
                result = await self.generate(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    request_id=str(uuid4()),
                )
                output_text = tokenizer.decode(result.token_ids, skip_special_tokens=True)
                finish_reason = "stop" if result.stop_reason in ("completed", "stop") else "length"
                return {
                    "id": f"chatcmpl-{uuid4()}",
                    "object": "chat.completion",
                    "created": int(_time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": output_text},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": len(result.token_ids),
                        "total_tokens": len(prompt_ids) + len(result.token_ids),
                    },
                }
            except Exception as exc:
                logger.error(f"Chat completion error: {exc}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        @app.post("/score")
        async def score_endpoint(request: ScoreRequest):
            try:
                tokenizer = self.model_config.tokenizer
                if request.tokenize:
                    prompt_ids = tokenizer.encode(request.input)
                else:
                    prompt_ids = tokenizer.encode(request.input, add_special_tokens=False)
                result = await self.generate(
                    prompt_ids=prompt_ids,
                    sampling_params={"max_tokens": 1, "temperature": 1.0, "logprobs": True},
                    request_id=str(uuid4()),
                )
                score = float(result.log_probs[0]) if result.log_probs else 0.0
                return {
                    "id": f"score-{uuid4()}",
                    "object": "score",
                    "model": request.model,
                    "score": score,
                    "usage": {"prompt_tokens": len(prompt_ids), "total_tokens": len(prompt_ids) + 1},
                }
            except Exception as exc:
                logger.error(f"Score error: {exc}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        self._server_port, self._server_task = await run_uvicorn(app, None, self._server_address)
        self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
        logger.info("ATOM HTTP server started at %s:%s", self._server_address, self._server_port)
        _atom_agent_log(
            "http_server_started",
            replica_rank=self.replica_rank,
            node_rank=self.node_rank,
            server_address=self._server_address,
            server_port=self._server_port,
            batch_size=self._batch_size,
        )

    async def _batch_processor_loop(self):
        while True:
            try:
                await self._batch_event.wait()
                deadline = asyncio.get_event_loop().time() + self._batch_timeout
                while True:
                    async with self._batch_lock:
                        if len(self._pending_requests) >= self._batch_size:
                            break
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(0.005, remaining))

                async with self._batch_lock:
                    if not self._pending_requests:
                        self._batch_event.clear()
                        continue
                    batch = self._pending_requests
                    self._pending_requests = []
                    self._batch_event.clear()
                if batch:
                    logger.info("ATOM dispatching batch of %s requests", len(batch))
                    _atom_agent_log(
                        "batch_dispatch",
                        replica_rank=self.replica_rank,
                        node_rank=self.node_rank,
                        batch_size=len(batch),
                        request_ids=[item[2] for item in batch[:8]],
                        diag_preview=[item[4] for item in batch[:8] if len(item) > 4],
                    )
                    await self._process_batch(batch)
            except Exception as exc:
                logger.error(f"ATOM batch processor error: {exc}", exc_info=True)
                _atom_agent_log(
                    "batch_processor_error",
                    replica_rank=self.replica_rank,
                    node_rank=self.node_rank,
                    error=repr(exc),
                )

    async def _process_batch(self, batch: list[tuple]):
        prompts, sampling_params, request_ids, futures, diag_infos = [], [], [], [], []
        for item in batch:
            prompt_ids, params, request_id, future = item[:4]
            diag_info = item[4] if len(item) > 4 else {}
            prompts.append(prompt_ids)
            sampling_params.append(params)
            request_ids.append(request_id)
            futures.append(future)
            diag_infos.append(diag_info)

        loop = asyncio.get_event_loop()

        def _generate_blocking():
            return self.engine.generate(prompts, sampling_params, request_ids=request_ids)

        try:
            start_time = time.time()
            outputs = await loop.run_in_executor(None, _generate_blocking)
            elapsed = time.time() - start_time
            output_lens = []
            finish_reasons = []
            for output in outputs:
                if isinstance(output, dict):
                    output_lens.append(len(output.get("token_ids", [])))
                    finish_reasons.append(output.get("finish_reason", output.get("stop_reason", "unknown")))
                else:
                    output_lens.append(len(getattr(output, "token_ids", [])))
                    finish_reasons.append(getattr(output, "finish_reason", getattr(output, "stop_reason", "unknown")))
            finish_reason_counts = {
                reason: finish_reasons.count(reason) for reason in sorted(set(finish_reasons))
            }
            max_token_outputs = sum(length >= self.config.response_length for length in output_lens)
            _atom_agent_log(
                "batch_complete",
                replica_rank=self.replica_rank,
                node_rank=self.node_rank,
                batch_size=len(batch),
                outputs=len(outputs),
                elapsed_s=round(elapsed, 3),
                min_output_len=min(output_lens) if output_lens else 0,
                max_output_len=max(output_lens) if output_lens else 0,
                max_token_outputs=max_token_outputs,
                finish_reason_counts=finish_reason_counts,
                diag_preview=diag_infos[:8],
            )
            for idx, future in enumerate(futures):
                if idx >= len(outputs):
                    _atom_agent_log(
                        "batch_missing_output",
                        replica_rank=self.replica_rank,
                        node_rank=self.node_rank,
                        batch_size=len(batch),
                        output_count=len(outputs),
                        missing_index=idx,
                    )
                    future.set_exception(RuntimeError(f"Missing ATOM output for request {idx}"))
                    continue
                future.set_result(self._to_token_output(outputs[idx]))
        except Exception as exc:
            logger.error(f"ATOM batch generation error: {exc}", exc_info=True)
            _atom_agent_log(
                "batch_generation_error",
                replica_rank=self.replica_rank,
                node_rank=self.node_rank,
                batch_size=len(batch),
                error=repr(exc),
            )
            for future in futures:
                if not future.done():
                    future.set_exception(exc)

    def _to_token_output(self, output: Any) -> TokenOutput:
        if isinstance(output, dict):
            token_ids = output.get("token_ids", [])
            log_probs = output.get("logprobs", output.get("log_probs"))
            finish_reason = output.get("finish_reason", output.get("stop_reason", "stop"))
        else:
            token_ids = getattr(output, "token_ids", [])
            log_probs = getattr(output, "logprobs", getattr(output, "log_probs", None))
            finish_reason = getattr(output, "finish_reason", getattr(output, "stop_reason", "stop"))
        stop_reason = "completed" if finish_reason in ("stop", "length", "completed") else finish_reason
        return TokenOutput(token_ids=token_ids, log_probs=log_probs, routed_experts=None, stop_reason=stop_reason)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        audio_data: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TokenOutput:
        if image_data or video_data or audio_data or mm_processor_kwargs:
            logger.warning("ATOM rollout does not support multimodal inputs; ignoring multimodal request fields.")
        diag_info = kwargs.pop("diag_info", {}) or {}
        if kwargs:
            logger.debug("ATOM rollout ignoring extra generate kwargs: %s", sorted(kwargs.keys()))

        prompt_ids = normalize_token_ids(prompt_ids)
        params = self._build_sampling_params(sampling_params, prompt_length=len(prompt_ids))
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        async with self._batch_lock:
            self._pending_requests.append((prompt_ids, params, request_id, future, diag_info))
            pending = len(self._pending_requests)
            if pending <= 3 or pending == self._batch_size:
                _atom_agent_log(
                    "request_enqueued",
                    replica_rank=self.replica_rank,
                    node_rank=self.node_rank,
                    request_id=request_id,
                    prompt_len=len(prompt_ids),
                    pending=pending,
                    target_batch_size=self._batch_size,
                    diag_info=diag_info,
                    sampling_params={
                        "max_tokens": getattr(params, "max_tokens", None),
                        "temperature": getattr(params, "temperature", None),
                        "top_p": getattr(params, "top_p", None),
                        "top_k": getattr(params, "top_k", None),
                        "logprobs": getattr(params, "logprobs", None),
                        "ignore_eos": getattr(params, "ignore_eos", None),
                    },
                )
            if len(self._pending_requests) >= self._batch_size:
                self._batch_event.set()
        self._batch_event.set()
        result = await future
        _atom_agent_log(
            "request_returned",
            replica_rank=self.replica_rank,
            node_rank=self.node_rank,
            request_id=request_id,
            output_len=len(result.token_ids),
            stop_reason=result.stop_reason,
            hit_max_tokens=len(result.token_ids) >= self.config.response_length,
            diag_info=diag_info,
        )
        return result

    async def wake_up(self, tags=None):
        if self.rollout_mode not in (RolloutMode.HYBRID, RolloutMode.COLOCATED) or self.engine is None:
            return
        try:
            self.engine.wake_up(tags=tags or ["kv_cache", "weights"])
        except TypeError:
            self.engine.wake_up()

    async def sleep(self, level=None):
        if not self.config.free_cache_engine or self.rollout_mode not in (RolloutMode.HYBRID, RolloutMode.COLOCATED):
            return
        if self.engine is not None:
            try:
                self.engine.sleep(level=level or ATOMDefaults.SLEEP_LEVEL)
            except TypeError:
                self.engine.sleep()

    async def release_kv_cache(self):
        if self.engine is not None:
            await self.sleep(level=SleepLevel.RELEASE_KV_CACHE_ONLY)

    async def resume_kv_cache(self):
        if self.engine is not None:
            await self.wake_up(tags=["kv_cache"])

    async def update_weights_from_zmq(self, use_shm=False, **kwargs):
        logger.info("ATOM update_weights_from_zmq start (use_shm=%s)", use_shm)
        _atom_agent_log(
            "weight_sync_start",
            replica_rank=self.replica_rank,
            node_rank=self.node_rank,
            use_shm=use_shm,
        )
        start_time = time.time()
        try:
            self._update_weights_from_zmq_sync(use_shm=use_shm)
        except Exception as exc:
            _atom_agent_log(
                "weight_sync_error",
                replica_rank=self.replica_rank,
                node_rank=self.node_rank,
                use_shm=use_shm,
                error=repr(exc),
            )
            raise
        finally:
            _atom_agent_log(
                "weight_sync_done",
                replica_rank=self.replica_rank,
                node_rank=self.node_rank,
                use_shm=use_shm,
                elapsed_s=round(time.time() - start_time, 3),
            )

    def _update_weights_from_zmq_sync(self, use_shm=False):
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        zmq_handle = f"ipc:///tmp/rl-colocate-zmq-atom-{self.job_id}-replica-{self.replica_rank}-rank-0.sock"
        socket.connect(zmq_handle)

        try:
            comm_metadata = socket.recv_pyobj()
            socket.send(b"")
            if use_shm:
                self._recv_weights_from_shm(socket, comm_metadata)
            else:
                self._recv_weights_from_ipc(socket, comm_metadata)
        finally:
            socket.close()
            ctx.term()
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        logger.info("ATOM update_weights_from_zmq completed")

    def _recv_weights_from_ipc(self, socket, comm_metadata):
        from atom.rollout.weight_sync import rebuild_ipc_handle
        from torch.multiprocessing.reductions import reduce_tensor

        ipc_buffer = rebuild_ipc_handle(comm_metadata, device_id=0)
        bucket_size = ipc_buffer.numel()
        num_gpus = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        per_gpu_buffers = {
            gpu_idx: torch.empty(bucket_size, dtype=torch.uint8, device=f"cuda:{gpu_idx}")
            for gpu_idx in range(num_gpus)
        }
        per_gpu_ipc_handles = {gpu_idx: reduce_tensor(buf) for gpu_idx, buf in per_gpu_buffers.items()}

        try:
            while True:
                metadata = socket.recv_pyobj()
                raw_bucket_meta = metadata["bucket_meta"]
                is_last = metadata["is_last"]
                bucket_meta, direct_tensors, used_bytes = self._prepare_bucket_meta(raw_bucket_meta, rebuild_ipc_handle)
                bucket_names = list(raw_bucket_meta.keys())
                largest_tensors = sorted(
                    (
                        (
                            name,
                            meta["dtype"].itemsize * torch.Size(meta["shape"]).numel(),
                            tuple(meta["shape"]),
                            str(meta["dtype"]),
                        )
                        for name, meta in raw_bucket_meta.items()
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )[:8]
                _atom_agent_log(
                    "weight_ipc_bucket_recv",
                    replica_rank=self.replica_rank,
                    node_rank=self.node_rank,
                    tensors=len(raw_bucket_meta),
                    used_bytes=used_bytes,
                    bucket_size=bucket_size,
                    direct_tensors=len(direct_tensors),
                    is_last=is_last,
                    first_names=bucket_names[:8],
                    last_names=bucket_names[-8:],
                    largest_tensors=largest_tensors,
                )

                if used_bytes > bucket_size:
                    _atom_agent_log(
                        "weight_ipc_resize_buffers",
                        replica_rank=self.replica_rank,
                        node_rank=self.node_rank,
                        old_bucket_size=bucket_size,
                        new_bucket_size=used_bytes,
                        num_gpus=num_gpus,
                    )
                    del per_gpu_buffers
                    del per_gpu_ipc_handles
                    bucket_size = used_bytes
                    per_gpu_buffers = {
                        gpu_idx: torch.empty(bucket_size, dtype=torch.uint8, device=f"cuda:{gpu_idx}")
                        for gpu_idx in range(num_gpus)
                    }
                    per_gpu_ipc_handles = {gpu_idx: reduce_tensor(buf) for gpu_idx, buf in per_gpu_buffers.items()}

                for gpu_idx, dst in per_gpu_buffers.items():
                    copy_start = time.time()
                    _atom_agent_log(
                        "weight_ipc_gpu_copy_start",
                        replica_rank=self.replica_rank,
                        node_rank=self.node_rank,
                        gpu_idx=gpu_idx,
                        used_bytes=used_bytes,
                        direct_tensors=len(direct_tensors),
                    )
                    if direct_tensors:
                        for name, tensor in direct_tensors.items():
                            meta = raw_bucket_meta[name]
                            nbytes = meta["dtype"].itemsize * torch.Size(meta["shape"]).numel()
                            offset = meta["offset"]
                            dst[offset : offset + nbytes].copy_(tensor.view(-1).view(torch.uint8), non_blocking=True)
                    else:
                        dst[:used_bytes].copy_(ipc_buffer[:used_bytes], non_blocking=True)
                    torch.cuda.synchronize(gpu_idx)
                    _atom_agent_log(
                        "weight_ipc_gpu_copy_done",
                        replica_rank=self.replica_rank,
                        node_rank=self.node_rank,
                        gpu_idx=gpu_idx,
                        used_bytes=used_bytes,
                        elapsed_s=round(time.time() - copy_start, 3),
                    )

                if hasattr(self.engine, "core_mgr"):
                    core_start = time.time()
                    _atom_agent_log(
                        "weight_ipc_core_update_start",
                        replica_rank=self.replica_rank,
                        node_rank=self.node_rank,
                        tensors=len(bucket_meta),
                        used_bytes=used_bytes,
                        is_last=is_last,
                    )
                    try:
                        self.engine.core_mgr.broadcast_utility_command_sync(
                            "update_weights_ipc",
                            ipc_handle=None,
                            ipc_handles=per_gpu_ipc_handles,
                            bucket_meta=bucket_meta,
                            is_last=is_last,
                        )
                    except Exception as exc:
                        _atom_agent_log(
                            "weight_ipc_core_update_error",
                            replica_rank=self.replica_rank,
                            node_rank=self.node_rank,
                            tensors=len(bucket_meta),
                            used_bytes=used_bytes,
                            is_last=is_last,
                            elapsed_s=round(time.time() - core_start, 3),
                            error=repr(exc),
                        )
                        raise
                    _atom_agent_log(
                        "weight_ipc_core_update_done",
                        replica_rank=self.replica_rank,
                        node_rank=self.node_rank,
                        tensors=len(bucket_meta),
                        used_bytes=used_bytes,
                        is_last=is_last,
                        elapsed_s=round(time.time() - core_start, 3),
                    )
                else:
                    _atom_agent_log(
                        "weight_ipc_local_load_start",
                        replica_rank=self.replica_rank,
                        node_rank=self.node_rank,
                        tensors=len(bucket_meta),
                        used_bytes=used_bytes,
                        is_last=is_last,
                    )
                    self._load_weights_from_buffer(per_gpu_buffers[0], bucket_meta, mode="ipc")
                    _atom_agent_log(
                        "weight_ipc_local_load_done",
                        replica_rank=self.replica_rank,
                        node_rank=self.node_rank,
                        tensors=len(bucket_meta),
                        used_bytes=used_bytes,
                        is_last=is_last,
                    )

                socket.send(b"")
                if is_last:
                    break
        finally:
            del per_gpu_buffers
            del per_gpu_ipc_handles
            del ipc_buffer
            gc.collect()
            torch.cuda.ipc_collect()
            for gpu_idx in range(num_gpus):
                with torch.cuda.device(gpu_idx):
                    torch.cuda.empty_cache()

    def _prepare_bucket_meta(self, raw_bucket_meta: dict, rebuild_ipc_handle):
        bucket_meta, direct_tensors = {}, {}
        used_bytes = 0
        for name, meta in raw_bucket_meta.items():
            shape = meta["shape"]
            dtype = meta["dtype"]
            offset = meta["offset"]
            handle = meta.get("handle")
            nbytes = dtype.itemsize * torch.Size(shape).numel()
            if handle is not None:
                direct_tensors[name] = rebuild_ipc_handle(handle, device_id=0)
            bucket_meta[name] = {
                "shape": tuple(shape),
                "dtype": str(dtype),
                "offset": offset,
                "nbytes": nbytes,
            }
            used_bytes = max(used_bytes, offset + nbytes)
        return bucket_meta, direct_tensors, used_bytes

    def _recv_weights_from_shm(self, socket, comm_metadata):
        shm = shared_memory.SharedMemory(name=comm_metadata["name"])
        buffer = torch.frombuffer(shm.buf[: comm_metadata["size"]], dtype=torch.uint8)
        all_weights = []
        try:
            while True:
                metadata = socket.recv_pyobj()
                _atom_agent_log(
                    "weight_shm_bucket_recv",
                    replica_rank=self.replica_rank,
                    node_rank=self.node_rank,
                    tensors=len(metadata["bucket_meta"]),
                    is_last=metadata["is_last"],
                )
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset = meta["shape"], meta["dtype"], meta["offset"]
                    nbytes = dtype.itemsize * torch.Size(shape).numel()
                    tensor = buffer[offset : offset + nbytes].view(dtype=dtype).view(shape).to("cuda:0")
                    all_weights.append((name, tensor))
                torch.cuda.synchronize()
                socket.send(b"")
                if metadata["is_last"]:
                    break
            atom_kwargs = (getattr(self.config, "engine_kwargs", {}) or {}).get("atom", {}) or {}
            bucket_size_mb = atom_kwargs.get("bucket_size_mb", IPCConfig.DEFAULT_BUCKET_SIZE_MB)
            self.engine.load_weights(iter(all_weights), bucket_size_mb=bucket_size_mb, mode="shm")
        finally:
            del all_weights
            del buffer
            shm.close()

    def _load_weights_from_buffer(self, buffer: torch.Tensor, bucket_meta: dict, mode: str):
        weights = []
        for name, meta in bucket_meta.items():
            dtype_name = meta["dtype"].removeprefix("torch.")
            dtype = getattr(torch, dtype_name)
            tensor = (
                buffer[meta["offset"] : meta["offset"] + meta["nbytes"]]
                .view(dtype=dtype)
                .view(meta["shape"])
            )
            weights.append((name, tensor))
        atom_kwargs = (getattr(self.config, "engine_kwargs", {}) or {}).get("atom", {}) or {}
        bucket_size_mb = atom_kwargs.get("bucket_size_mb", IPCConfig.DEFAULT_BUCKET_SIZE_MB)
        self.engine.load_weights(iter(weights), bucket_size_mb=bucket_size_mb, mode=mode)

    async def clear_kv_cache(self):
        if self.engine is None:
            return
        _atom_agent_log("clear_kv_cache", replica_rank=self.replica_rank, node_rank=self.node_rank)
        if hasattr(self.engine, "core_mgr"):
            self.engine.core_mgr.broadcast_utility_command("clear_kv_cache")
        elif hasattr(self.engine, "clear_kv_cache"):
            self.engine.clear_kv_cache()

    async def reset_deferred_state(self):
        if self.engine is not None and hasattr(self.engine, "core_mgr"):
            logger.info("reset_deferred_state: clearing ATOM deferred token/logprob state")
            _atom_agent_log("reset_deferred_state", replica_rank=self.replica_rank, node_rank=self.node_rank)
            self.engine.core_mgr.broadcast_utility_command_sync("reset_deferred_state")

    async def set_global_steps(self, global_steps: int):
        self.global_steps = global_steps
        _atom_agent_log(
            "set_global_steps",
            replica_rank=self.replica_rank,
            node_rank=self.node_rank,
            global_steps=global_steps,
        )

    async def abort_all_requests(self) -> dict:
        logger.info("abort_all_requests called (no-op for ATOM)")
        return {"aborted_count": 0, "request_ids": []}

    async def abort_request(self, request_id: str) -> dict:
        logger.info("abort_request(%s) called (no-op for ATOM)", request_id)
        return {"aborted": False, "request_id": request_id, "error": "ATOM does not support request abort"}

    async def resume_generation(self):
        pass

    async def start_profile(self, **kwargs):
        logger.debug("start_profile called (no-op for ATOM)")

    async def stop_profile(self):
        logger.debug("stop_profile called (no-op for ATOM)")

    @property
    def lora_as_adapter(self) -> bool:
        return False


class ATOMReplica(RolloutReplica):
    """ATOM rollout replica manager."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix
        )
        self.server_class = ray.remote(ATOMHttpServer)

    def _get_server_name_prefix(self) -> str:
        if self.is_reward_model:
            return "atom_server_reward"
        if self.is_teacher_model:
            return "atom_server_teacher"
        return "atom_server"

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_node_ids = [info[0] for info in worker_infos]
        worker_gpu_ids = [info[1] for info in worker_infos]

        for node_rank in range(self.nnodes):
            start = node_rank * self.gpus_per_replica_node
            end = start + self.gpus_per_replica_node
            node_cuda_visible_devices = ",".join(worker_gpu_ids[start:end])
            node_id = worker_node_ids[start]
            name = f"{self._get_server_name_prefix()}_{self.replica_rank}_{node_rank}{self.name_suffix}"
            env_vars = {
                "CUDA_VISIBLE_DEVICES": node_cuda_visible_devices,
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
            }
            for key in (
                "ATOM_ISOLATE_TORCH_COMPILE_CACHE",
                "ATOM_USE_TORCH_RMSNORM",
                "ATOM_DISABLE_VLLM_PLUGIN",
                "VERL_ATOM_AGENT_LOG",
                "VERL_MEMORY_AGENT_LOG",
            ):
                if key in os.environ:
                    env_vars[key] = os.environ[key]

            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": env_vars},
                name=name,
                max_concurrency=self.max_concurrency,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=self.workers[start:end],
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=self.gpus_per_replica_node,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
        logger.info("ATOMReplica %s launched at %s", self.replica_rank, self._server_address)
