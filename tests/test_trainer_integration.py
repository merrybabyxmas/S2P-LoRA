"""Integration tests for the HuggingFace Trainer switch lifecycle."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import TrainingArguments
from transformers.utils.import_utils import check_torch_load_is_safe

from s2p_lora.layer import S2PLoraLinear
from s2p_lora.trainer import S2PLoraTrainer


class TinyDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(123)
        self.inputs = torch.randn(4, 4, generator=generator)
        self.labels = torch.tensor([0, 1, 0, 1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"features": self.inputs[index], "labels": self.labels[index]}


class TinyS2PModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.s2p = S2PLoraLinear(
            nn.Linear(4, 3, bias=False),
            r=2,
            lora_alpha=2.0,
        )
        nn.init.normal_(self.s2p.lora_B.weight, std=0.02)
        self.classifier = nn.Linear(3, 2)

    def forward(self, features, labels=None):
        logits = self.classifier(torch.tanh(self.s2p(features)))
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


def _training_args(tmp_path):
    return TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=1e-2,
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        use_cpu=True,
        seed=123,
    )


def test_trainer_discovers_and_switches_real_layer(tmp_path):
    """A short real training run must switch and update optimizer references."""
    model = TinyS2PModel()
    old_lora_a = model.s2p.lora_A.weight
    trainer = S2PLoraTrainer(
        model=model,
        args=_training_args(tmp_path),
        train_dataset=TinyDataset(),
        switch_ratio=0.5,
        momentum_reset=True,
    )

    trainer.train()

    optimizer_params = [
        param for group in trainer.optimizer.param_groups for param in group["params"]
    ]
    assert trainer.s2p_layers == [model.s2p]
    assert trainer.has_switched
    assert model.s2p.mode == "parallel"
    assert model.s2p.lora_A.weight.shape == (2, 4)
    assert any(param is model.s2p.lora_A.weight for param in optimizer_params)
    assert all(param is not old_lora_a for param in optimizer_params)


def test_trainer_rejects_missing_s2p_layers(tmp_path):
    """A wiring error must fail loudly instead of training as a silent no-op."""
    trainer = S2PLoraTrainer(
        model=nn.Linear(4, 2),
        args=_training_args(tmp_path),
        switch_ratio=0.5,
    )

    try:
        trainer._ensure_layers_cached()
    except RuntimeError as error:
        assert "found no compatible S2P-LoRA layers" in str(error)
    else:
        raise AssertionError("missing S2P-LoRA layers should raise RuntimeError")


@pytest.mark.parametrize("switch_ratio", [0.0, 1.0, -0.1, 1.1])
def test_trainer_requires_an_actual_sequential_to_parallel_switch(tmp_path, switch_ratio):
    with pytest.raises(ValueError, match=r"switch_ratio must be in \(0, 1\)"):
        S2PLoraTrainer(
            model=TinyS2PModel(),
            args=_training_args(tmp_path),
            train_dataset=TinyDataset(),
            switch_ratio=switch_ratio,
        )


def test_evaluation_does_not_switch_and_single_step_schedule_rejects(tmp_path):
    args = _training_args(tmp_path)
    args.max_steps = 1
    model = TinyS2PModel()
    ds = TinyDataset()
    trainer = S2PLoraTrainer(model=model, args=args, train_dataset=ds, eval_dataset=ds)
    trainer.evaluate()
    assert model.s2p.mode == "sequential"
    with pytest.raises(ValueError, match="at least one training step"):
        trainer.train()
    assert model.s2p.mode == "sequential"


def test_resume_on_either_side_of_rectangular_switch(tmp_path):
    try:
        check_torch_load_is_safe()
    except ValueError as error:
        pytest.skip(str(error))

    torch.manual_seed(7)
    base = TinyS2PModel()
    dataset = TinyDataset()

    def make_trainer(output):
        args = _training_args(output)
        args.max_steps = 4
        args.gradient_accumulation_steps = 2
        args.save_strategy = "steps"
        args.save_steps = 1
        return S2PLoraTrainer(model=deepcopy(base), args=args, train_dataset=dataset)

    baseline = make_trainer(tmp_path / "baseline")
    baseline.train()
    assert baseline.switch_events == [{"step": 2, "progress": 0.5, "layers": 1}]
    for step in (2, 3):
        resumed = make_trainer(tmp_path / f"resume-{step}")
        resumed.train(resume_from_checkpoint=str(tmp_path / "baseline" / f"checkpoint-{step}"))
        assert resumed.switch_events == baseline.switch_events
        assert resumed.state.global_step == baseline.state.global_step
        assert resumed.lr_scheduler.state_dict() == baseline.lr_scheduler.state_dict()
        for key, value in baseline.model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[key], value, rtol=0, atol=0)
