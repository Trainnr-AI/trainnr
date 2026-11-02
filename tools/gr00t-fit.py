"""GR00T N1.7 fine-tuning sized for one 24 GB card (docs/86 §5.5): NVIDIA's
launch_finetune, unchanged, with three memory savers it does not expose and
a step timer. Runs inside the Isaac-GR00T checkout's own environment, never
trainnr's; `tools/gr00t-batch-search.sh` drives it:

    uv run --no-sync python tools/gr00t-fit.py [--bf16-backbone] [--optim NAME] \
        [--checkpoint-dit] <launch_finetune args>
"""

import argparse
import runpy
import statistics
import sys
import time

import torch
from gr00t.experiment import experiment
from gr00t.model.modules import dit
from torch.utils.checkpoint import checkpoint
from transformers import TrainerCallback

ours = argparse.ArgumentParser(add_help=False)
# The frozen Cosmos-Reason2 backbone in bf16; trainable backbone params stay
# fp32 (~3 GB saved).
ours.add_argument("--bf16-backbone", action="store_true")
# A TrainingArguments.optim name, e.g. adamw_bnb_8bit: AdamW state in 8 bits.
ours.add_argument("--optim")
# Recompute each DiT block's activations in the backward pass instead of
# storing them.
ours.add_argument("--checkpoint-dit", action="store_true")
args, rest = ours.parse_known_args()


class StepTimer(TrainerCallback):
    """NVIDIA's measure: forward, backward and optimizer step, synchronized,
    data loading excluded."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def on_step_begin(self, *_, **__):
        torch.cuda.synchronize()
        self.start = time.perf_counter()

    def on_step_end(self, *_, **__):
        torch.cuda.synchronize()
        self.times.append(time.perf_counter() - self.start)

    def on_train_end(self, *_, **__):
        timed = self.times[2:] or self.times  # two warmup steps, as NVIDIA's table
        peak = torch.cuda.max_memory_allocated() / 2**20
        print(
            f"gr00t_fit: step_ms={1000 * statistics.median(timed):.1f} "
            f"torch_peak_mib={peak:.0f}",
            flush=True,
        )


class TimedTrainer(experiment.Gr00tTrainer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.add_callback(StepTimer())


nvidia_run = experiment.run


def run(config):
    if args.bf16_backbone:
        config.model.load_bf16 = True
    if args.optim:
        config.training.optim = args.optim
    print(
        f"gr00t_fit: load_bf16={config.model.load_bf16} optim={config.training.optim} "
        f"checkpoint_dit={args.checkpoint_dit}",
        flush=True,
    )
    return nvidia_run(config)


experiment.run = run
experiment.Gr00tTrainer = TimedTrainer

if args.checkpoint_dit:
    # The DiT declares gradient-checkpointing support, but its forward never
    # checkpoints. Keyed on grad mode, not self.training: a frozen DiT is in
    # eval() yet still backpropagates to the projector, so it stores
    # activations all the same.
    block_forward = dit.BasicTransformerBlock.forward

    def checkpointed(self, *a, **k):
        if torch.is_grad_enabled():
            return checkpoint(block_forward, self, *a, use_reentrant=False, **k)
        return block_forward(self, *a, **k)

    dit.BasicTransformerBlock.forward = checkpointed

sys.argv = ["launch_finetune.py", *rest]
runpy.run_path("gr00t/experiment/launch_finetune.py", run_name="__main__")
