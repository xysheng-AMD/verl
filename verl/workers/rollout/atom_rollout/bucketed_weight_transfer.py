"""ATOM rollout weight-transfer sender.

ATOM uses the same ZMQ + IPC bucket protocol as the vLLM rollout sender. Keep a
thin module-level alias so Atom imports remain backend-local while the sender
implementation stays shared.
"""

from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

__all__ = ["BucketedWeightSender"]
