"""Exercise standard LoRA loading in PEFT without S2P layers."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, LlamaForSequenceClassification

from s2p_lora import S2PLoraLinear, export_peft_adapter, inject_s2p_adapter, save_adapter


@pytest.mark.parametrize("classification", [False, True])
def test_peft_load_generate_and_merge(tmp_path, classification):
    peft = pytest.importorskip("peft")
    torch.manual_seed(21)
    config = LlamaConfig(
        vocab_size=24,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        num_labels=2,
    )
    cls = LlamaForSequenceClassification if classification else LlamaForCausalLM
    base = cls(config).eval()
    model = deepcopy(base)
    report = inject_s2p_adapter(
        model,
        ["q_proj", "k_proj", "v_proj"],
        r=2,
        lora_alpha=4,
        modules_to_save=["score"] if classification else (),
    )
    for name in report.adapted_modules:
        layer = model.get_submodule(name)
        nn.init.normal_(layer.lora_B.weight, std=0.02)
        layer.switch_to_parallel()
    if classification:
        nn.init.normal_(model.score.weight, std=0.3)
    ids = torch.tensor([[1, 3, 5, 7]])
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    expected = model(**inputs).logits
    checkpoint = save_adapter(model, tmp_path / "adapter.pt")
    exported = export_peft_adapter(
        checkpoint,
        tmp_path / "export",
        base_model="tiny-llama-fixture",
        task_type="SEQ_CLS" if classification else "CAUSAL_LM",
    )
    reloaded = peft.PeftModel.from_pretrained(deepcopy(base), exported).eval()
    torch.testing.assert_close(reloaded(**inputs).logits, expected, atol=1e-6, rtol=1e-5)
    if not classification:
        assert torch.equal(
            model.generate(**inputs, max_new_tokens=3),
            reloaded.generate(**inputs, max_new_tokens=3),
        )
    merged = reloaded.merge_and_unload(safe_merge=True)
    torch.testing.assert_close(merged(**inputs).logits, expected, atol=1e-6, rtol=1e-5)
    with pytest.raises(FileExistsError):
        export_peft_adapter(checkpoint, exported, base_model="tiny-llama-fixture")


def test_export_rejects_sequential_checkpoint(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4))
    inject_s2p_adapter(model, ["0"], r=2)
    checkpoint = save_adapter(model, tmp_path / "adapter.pt")
    with pytest.raises(ValueError, match="still sequential"):
        export_peft_adapter(checkpoint, tmp_path / "invalid", base_model="example")
    assert not (tmp_path / "invalid").exists()


def test_legacy_export_with_per_layer_rank_and_alpha(tmp_path):
    peft = pytest.importorskip("peft")
    base = nn.Sequential(nn.Linear(6, 4), nn.Linear(4, 5)).eval()
    model = deepcopy(base)
    model[0] = S2PLoraLinear(model[0], r=2, lora_alpha=3.0)
    model[1] = S2PLoraLinear(model[1], r=3, lora_alpha=9.0)
    for layer in model:
        nn.init.normal_(layer.lora_B.weight, std=0.05)
        layer.switch_to_parallel()
    checkpoint = save_adapter(model, tmp_path / "legacy.pt")
    payload = torch.load(checkpoint, weights_only=True)
    for key in ("extra_buffers", "modules_to_save"):
        payload.pop(key)
    for layer in payload["layers"].values():
        layer.pop("dropout")
    torch.save(payload, checkpoint)

    exported = export_peft_adapter(checkpoint, tmp_path / "peft", base_model="tiny-fixture")
    restored = peft.PeftModel.from_pretrained(base, exported).eval()
    x = torch.randn(4, 6)
    torch.testing.assert_close(restored(x), model(x), atol=1e-6, rtol=1e-5)
