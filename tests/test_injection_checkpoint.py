"""Tests for model injection, compact checkpoints, and adapter merging."""

import torch
from torch import nn

from s2p_lora import (
    S2PLoraLinear,
    inject_s2p_adapter,
    load_adapter,
    merge_adapters,
    save_adapter,
)


class TinyAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict(
            {
                "q_proj": nn.Linear(6, 4, bias=False),
                "untouched": nn.Linear(4, 4, bias=False),
            }
        )
        self.classifier = nn.Linear(4, 2)

    def forward(self, features):
        hidden = self.encoder["q_proj"](features)
        hidden = torch.tanh(self.encoder["untouched"](hidden))
        return self.classifier(hidden)


def _prepared_model(seed=7):
    torch.manual_seed(seed)
    model = TinyAttentionModel()
    report = inject_s2p_adapter(
        model,
        ["q_proj"],
        r=2,
        lora_alpha=2,
        modules_to_save=["classifier"],
    )
    return model, report


def test_injection_freezes_everything_except_adapter_and_saved_module():
    model, report = _prepared_model()

    assert report.adapted_modules == ("encoder.q_proj",)
    assert isinstance(model.encoder["q_proj"], S2PLoraLinear)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "encoder.q_proj.lora_A.weight",
        "encoder.q_proj.lora_B.weight",
        "classifier.weight",
        "classifier.bias",
    }


def test_adapter_checkpoint_round_trip_and_merge(tmp_path):
    model, _ = _prepared_model()
    layer = model.encoder["q_proj"]
    nn.init.normal_(layer.lora_B.weight, std=0.1)
    nn.init.normal_(model.classifier.weight, std=0.1)
    layer.switch_to_parallel()
    features = torch.randn(3, 6)
    expected = model(features).detach()

    checkpoint = save_adapter(model, tmp_path / "adapter.pt", metadata={"seed": 7})
    restored, _ = _prepared_model()
    metadata = load_adapter(restored, checkpoint)
    actual = restored(features).detach()

    assert metadata == {"seed": 7}
    assert restored.encoder["q_proj"].mode == "parallel"
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    merged_names = merge_adapters(restored)
    merged = restored(features).detach()
    assert merged_names == ("encoder.q_proj",)
    assert isinstance(restored.encoder["q_proj"], nn.Linear)
    torch.testing.assert_close(merged, expected, rtol=1e-5, atol=1e-6)


def test_injection_fails_loudly_for_missing_target():
    model = TinyAttentionModel()
    try:
        inject_s2p_adapter(model, ["does_not_exist"], r=2)
    except ValueError as error:
        assert "No nn.Linear modules matched" in str(error)
    else:
        raise AssertionError("missing target module should raise ValueError")
