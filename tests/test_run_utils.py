"""Tests for deterministic, collision-safe run artifact paths."""

import pytest

from s2p_lora.run_utils import make_run_dir, write_json


def test_run_directory_layout_and_collision_guard(tmp_path):
    run_dir = make_run_dir(
        tmp_path,
        family="nlu",
        task="cola",
        model="microsoft/deberta-v3-base",
        rank=8,
        switch_ratio=0.5,
        seed=42,
    )
    assert run_dir.relative_to(tmp_path).as_posix() == (
        "nlu/cola/microsoft--deberta-v3-base/s2p-r8-sw0.5/seed-42"
    )

    write_json(run_dir / "metrics.json", {"score": 1.0})
    with pytest.raises(FileExistsError):
        make_run_dir(
            tmp_path,
            family="nlu",
            task="cola",
            model="microsoft/deberta-v3-base",
            rank=8,
            switch_ratio=0.5,
            seed=42,
        )

    reused = make_run_dir(
        tmp_path,
        family="nlu",
        task="cola",
        model="microsoft/deberta-v3-base",
        rank=8,
        switch_ratio=0.5,
        seed=42,
        overwrite=True,
    )
    assert reused == run_dir
