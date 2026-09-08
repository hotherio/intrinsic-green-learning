"""CUDA graph replay: the runner's bookkeeping and the trainer's eligibility rules (CPU-testable parts)."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import igl
from igl import IGLModule, MatryoshkaConfig, MatryoshkaTrainer, MSELoss
from igl.core._graph import GraphRunner
from igl.device import CpuBackend, CudaBackend


class _FakeGraph:
    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


def test_runner_warms_up_captures_once_and_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = GraphRunner(device=torch.device("cpu"), lr=1e-3, batch_size=4, inner_n=None, d_max=3, warmup=2)
    calls = {"step": 0, "captures": 0}
    graphs: list[_FakeGraph] = []

    def fake_capture(self: GraphRunner, step: Any) -> _FakeGraph:
        calls["captures"] += 1
        graphs.append(_FakeGraph())
        return graphs[-1]

    monkeypatch.setattr(GraphRunner, "_capture", fake_capture)

    def step() -> None:
        calls["step"] += 1

    runner.bind(step)
    idx = torch.arange(4)
    mask = torch.tensor([1.0, 1.0, 0.0])
    for _ in range(5):
        runner.run(idx=idx, mask=mask, inner_idx=None)
    assert calls["step"] == 2  # the two warm-up batches ran eagerly
    assert calls["captures"] == 1 and runner.captures == 1
    assert graphs[0].replays == 3 and runner.replays == 3
    assert torch.equal(runner.idx, idx) and torch.equal(runner.mask, mask)


def test_runner_hands_a_scheduler_learning_rate_back_as_its_tensor() -> None:
    runner = GraphRunner(device=torch.device("cpu"), lr=1e-3, batch_size=2, inner_n=None, d_max=2)
    param = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.AdamW([param], lr=runner.lr)
    assert optimizer.param_groups[0]["lr"] is runner.lr
    optimizer.param_groups[0]["lr"] = torch.tensor(5e-4)  # what a scheduler step does
    runner.sync_lr(optimizer)
    assert optimizer.param_groups[0]["lr"] is runner.lr and float(runner.lr) == pytest.approx(5e-4)
    optimizer.param_groups[0]["lr"] = 2e-4  # a plain float works too
    runner.sync_lr(optimizer)
    assert float(runner.lr) == pytest.approx(2e-4)


def test_runner_without_graph_runs_the_bound_step_every_time() -> None:
    runner = GraphRunner(device=torch.device("cpu"), lr=1e-3, batch_size=2, inner_n=3, d_max=2, use_graph=False)
    seen: list[torch.Tensor] = []
    runner.bind(lambda: seen.append(runner.inner_idx.clone() if runner.inner_idx is not None else torch.empty(0)))
    runner.run(idx=torch.tensor([1, 0]), mask=torch.ones(2), inner_idx=torch.tensor([2, 1, 0]))
    assert runner.eager_runs == 1 and seen[0].tolist() == [2, 1, 0]
    with pytest.raises(RuntimeError, match="bind"):
        GraphRunner(device=torch.device("cpu"), lr=1e-3, batch_size=2, inner_n=None, d_max=2).run(
            idx=torch.zeros(2, dtype=torch.long), mask=torch.ones(2), inner_idx=None
        )


def _trainer(**overrides: object) -> MatryoshkaTrainer:
    base: dict[str, object] = {
        "epochs": 1,
        "batch_size": 8,
        "inner_batch_size": 16,
        "early_stop_patience": None,
        "verbose": False,
    }
    base.update(overrides)
    return MatryoshkaTrainer(loss=MSELoss(), config=MatryoshkaConfig(**base))  # type: ignore[arg-type]


def test_graph_runner_eligibility_rules() -> None:
    module = IGLModule(input_dim=3, max_dim=2, output_dim=3, n_anchors=4, n_scales=2)
    common = {"device": torch.device("cpu"), "d_max": 2, "n_samples": 16}
    cuda = CudaBackend()
    assert _trainer()._graph_runner(module, CpuBackend(), extra_losses=(), **common) is None  # noqa: SLF001
    runner = _trainer()._graph_runner(module, cuda, extra_losses=(), **common)  # noqa: SLF001
    assert runner is not None and runner.inner_idx is None  # the subset is the whole set
    partial = _trainer(inner_batch_size=8)._graph_runner(module, cuda, extra_losses=(), **common)  # noqa: SLF001
    assert partial is not None and partial.inner_idx is not None and partial.inner_idx.shape == (8,)
    assert _trainer(cuda_graphs=False)._graph_runner(module, cuda, extra_losses=(), **common) is None  # noqa: SLF001
    compiled = _trainer(cuda_graphs=False, torch_compile=True)._graph_runner(module, cuda, extra_losses=(), **common)  # noqa: SLF001
    assert compiled is not None and not compiled.use_graph and compiled.use_compile

    from igl.spd import OrthogonalityPenalty

    assert _trainer()._graph_runner(module, cuda, extra_losses=[OrthogonalityPenalty(weight=0.1)], **common) is None  # noqa: SLF001


def test_graph_runner_refuses_synchronising_losses_and_data_driven_bases() -> None:
    from igl.spd import AIRMLoss
    from igl.spectral import LearnedLaplacianBasis, SpectralKernel

    module = IGLModule(input_dim=3, max_dim=2, output_dim=3, n_anchors=4, n_scales=2)
    common = {"device": torch.device("cpu"), "d_max": 2, "n_samples": 16}
    cfg = MatryoshkaConfig(epochs=1, batch_size=8, inner_batch_size=16, early_stop_patience=None, verbose=False)
    eigh = MatryoshkaTrainer(loss=AIRMLoss(latent_dim=2), config=cfg)
    iterative = MatryoshkaTrainer(loss=AIRMLoss(latent_dim=2, matrix_method="iterative"), config=cfg)
    assert eigh._graph_runner(module, CudaBackend(), extra_losses=(), **common) is None  # noqa: SLF001
    assert iterative._graph_runner(module, CudaBackend(), extra_losses=(), **common) is not None  # noqa: SLF001
    kernel = SpectralKernel(latent_dim=2, bases=LearnedLaplacianBasis(n_modes=4, k_nn=5), n_anchors=4)
    learned = IGLModule(input_dim=3, max_dim=2, output_dim=3, kernel=kernel)
    assert _trainer()._graph_runner(learned, CudaBackend(), extra_losses=(), **common) is None  # noqa: SLF001


def test_config_round_trips_the_graph_switches() -> None:
    cfg = igl.IGLConfig(matryoshka=MatryoshkaConfig(cuda_graphs=False, torch_compile=True))
    data = cfg.to_dict()
    back = igl.IGLConfig.from_dict(data)
    assert back.matryoshka.cuda_graphs is False and back.matryoshka.torch_compile is True
    assert igl.IGLConfig().matryoshka.cuda_graphs is True and igl.IGLConfig().matryoshka.torch_compile is False
