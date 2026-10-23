"""torch.distributed: two real processes (gloo, CPU). Rank 0 alone writes; repair is identical on
both ranks; FSDP refuses repair and switches channel checks off, saying so in the manifest."""
from __future__ import annotations

import json
import os

import pytest
import torch
import torch.nn as nn

dist = pytest.importorskip("torch.distributed")
if os.name == "nt":
    pytest.skip("multi-process gloo tests are not run on Windows", allow_module_level=True)
if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("torch.distributed with gloo is not available", allow_module_level=True)

WORLD = 2


def _ddp_worker(rank: int, tmp: str) -> None:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from ormas_monitor import Monitor
    from test_core_helpers import batches, lesion, small_cnn

    dist.init_process_group("gloo", init_method=f"file://{tmp}/pg", rank=rank, world_size=WORLD)
    try:
        model = small_cnn()                          # same seed on both ranks
        ddp = DDP(model)
        opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
        seen = []
        mon = Monitor(ddp, opt, out=os.path.join(tmp, "run"), repair="all", check_every=5, patience=2,
                      verbose=True, sinks=[lambda kind, payload: seen.append(kind)])
        for i, (x, y) in enumerate(batches(100, seed=10 + rank)):     # different data per rank
            if i == 50:
                lesion(model[3], model[4], opt)
            loss = nn.functional.cross_entropy(ddp(x), y)
            opt.zero_grad()
            loss.backward()
            mon.step(loss)
            opt.step()
        report = mon.close(report=False)
        torch.save({"state": {k: v.detach() for k, v in model.named_parameters()}, "sink_calls": len(seen), "counts": mon.counts,
                    "report": report, "mode": mon.dist_mode}, os.path.join(tmp, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _fsdp_worker(rank: int, tmp: str) -> None:
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from ormas_monitor import Monitor
    from test_core_helpers import batches, small_cnn

    dist.init_process_group("gloo", init_method=f"file://{tmp}/pg", rank=rank, world_size=WORLD)
    try:
        model = FSDP(small_cnn(), device_id=torch.device("cpu"))
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        refused = False
        try:
            Monitor(model, opt, out=os.path.join(tmp, "refused"), repair=True, verbose=False)
        except ValueError as e:
            refused = "FSDP" in str(e)
        mon = Monitor(model, opt, out=os.path.join(tmp, "run"), check_every=2, verbose=False)
        for x, y in batches(6, seed=rank):
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            mon.step(loss)
            opt.step()
        mon.close(report=False)
        torch.save({"refused": refused, "mode": mon.dist_mode}, os.path.join(tmp, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(fn, tmp_path) -> list[dict]:
    import torch.multiprocessing as mp
    mp.spawn(fn, args=(str(tmp_path),), nprocs=WORLD, join=True)
    return [torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in range(WORLD)]


def test_ddp_rank0_writes_once_and_repairs_identically(tmp_path, capfd):
    from test_core_helpers import events
    r0, r1 = _spawn(_ddp_worker, tmp_path)
    assert r0["mode"] == r1["mode"] == "ddp"
    ev = events(tmp_path / "run")
    kinds = [e["event"] for e in ev]
    assert kinds.count("run_start") == 1 and kinds.count("run_end") == 1 and kinds.count("structure") == 1
    assert [e["seq"] for e in ev] == list(range(len(ev)))           # one writer, one chain
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["diary.jsonl", "metrics.jsonl"]
    assert r1["sink_calls"] == 0 and r0["sink_calls"] > 0 and r1["report"] is None
    out = capfd.readouterr().out
    assert out.count("re-initialised") == r0["counts"]["repairs"] >= 1    # printed by rank 0 only

    rep = [e for e in ev if e["event"] == "repair"]
    assert rep and rep[0]["layer"] == "3" and "sha256(seed=0" in rep[0]["rng"]
    assert r0["counts"] == r1["counts"]
    for k, v in r0["state"].items():
        assert torch.equal(v, r1["state"][k]), k      # parameters bitwise identical after repair
    # (BatchNorm running stats are not compared: each rank updates them from its own batch)
    assert r0["state"]["3.weight"].abs().amax(dim=(1, 2, 3)).min() > 0

    man = next(e for e in ev if e["event"] == "manifest")
    assert man["distributed"] == {"mode": "ddp", "rank": 0, "world_size": 2,
                                  "wrappers": ["DistributedDataParallel"]}
    assert "silent" not in man["repair"]["targets"]
    assert next(e for e in ev if e["event"] == "run_start")["model"] == "Sequential"


def test_fsdp_refuses_repair_and_disables_the_certificate(tmp_path):
    pytest.importorskip("torch.distributed.fsdp")
    from test_core_helpers import events
    r0, r1 = _spawn(_fsdp_worker, tmp_path)
    assert r0["refused"] and r1["refused"] and r0["mode"] == "fsdp"
    ev = events(tmp_path / "run")
    man = next(e for e in ev if e["event"] == "manifest")
    assert man["channel_checks"] is False and man["distributed"]["mode"] == "fsdp"
    assert any("FSDP" in n for n in man["notes"])
    assert not any(e["event"] in ("certificate", "channels", "repair") for e in ev)
    rows = [json.loads(l) for l in open(tmp_path / "run" / "metrics.jsonl")]
    assert len(rows) == 3 and all(r["stuck"] == {} for r in rows)
