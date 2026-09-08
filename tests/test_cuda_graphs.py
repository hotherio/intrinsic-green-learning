"""CUDA graph replay of the batch step: parity with eager training. Runs only where CUDA is available."""

from __future__ import annotations

import pytest
import torch

import igl
from igl import IGLModule, MatryoshkaConfig, MatryoshkaTrainer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a CUDA device")


def _fit(
    *, cuda_graphs: bool, torch_compile: bool = False, n: int = 1024, epochs: int = 3, scheduler: str = "none"
) -> tuple[list[float], object]:
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(n, 8, device=device)
    y = (x[:, 0] * x[:, 1] > 0).long()
    torch.manual_seed(1)
    module = IGLModule(input_dim=8, max_dim=4, output_dim=2, n_anchors=32, n_scales=3).to(device)
    cfg = MatryoshkaConfig(
        epochs=epochs,
        batch_size=128,
        inner_batch_size=1024,
        early_stop_patience=None,
        verbose=False,
        scheduler=scheduler,
        cuda_graphs=cuda_graphs,
        torch_compile=torch_compile,
    )
    trainer = MatryoshkaTrainer(loss=igl.CrossEntropyLoss(n_classes=2), config=cfg)
    torch.manual_seed(2)
    history = trainer.fit(module, x, y)
    return history.train_loss, module


def test_graph_replay_matches_eager_training() -> None:
    eager, _ = _fit(cuda_graphs=False)
    graph, module = _fit(cuda_graphs=True)
    assert len(eager) == len(graph) == 3
    for a, b in zip(eager, graph, strict=True):
        assert abs(a - b) <= 1e-3 * max(1.0, abs(a)), (eager, graph)
    assert all(torch.isfinite(p).all() for p in module.parameters())


def test_graph_replay_captures_once_and_replays_the_full_batches() -> None:
    from igl.core._graph import GraphRunner

    captured: list[GraphRunner] = []
    original = MatryoshkaTrainer._graph_runner  # noqa: SLF001

    def spy(self: MatryoshkaTrainer, *args: object, **kwargs: object) -> GraphRunner | None:
        runner = original(self, *args, **kwargs)  # type: ignore[arg-type]
        if runner is not None:
            captured.append(runner)
        return runner

    MatryoshkaTrainer._graph_runner = spy  # type: ignore[method-assign]  # noqa: SLF001
    try:
        _fit(cuda_graphs=True, n=1000, epochs=2)  # 7 full batches + 1 partial per epoch
    finally:
        MatryoshkaTrainer._graph_runner = original  # type: ignore[method-assign]  # noqa: SLF001
    runner = captured[-1]
    assert runner.captures == 1 and not runner.disabled
    assert runner.eager_runs == 2 and runner.replays == 2 * 7 - 2


def test_compiled_step_matches_eager_training() -> None:
    eager, _ = _fit(cuda_graphs=False, epochs=2)
    compiled, _ = _fit(cuda_graphs=True, torch_compile=True, epochs=2)
    for a, b in zip(eager, compiled, strict=True):
        assert abs(a - b) <= 5e-3 * max(1.0, abs(a)), (eager, compiled)


def test_eigh_airm_loss_trains_eagerly_on_cuda() -> None:
    from igl.data import make_spd_dataset
    from igl.spd import AIRMLoss, LogEigVectorizer

    covs, _ = make_spd_dataset(256, d=4, seed=0)
    vec = torch.as_tensor(LogEigVectorizer().fit(covs.numpy()).transform(covs.numpy()), dtype=torch.float32).cuda()
    torch.manual_seed(0)
    module = IGLModule(input_dim=vec.shape[1], max_dim=3, output_dim=vec.shape[1], n_anchors=16, n_scales=2).cuda()
    cfg = MatryoshkaConfig(epochs=1, batch_size=64, inner_batch_size=256, early_stop_patience=None, verbose=False)
    history = MatryoshkaTrainer(loss=AIRMLoss(latent_dim=4), config=cfg).fit(module, vec, vec)
    assert len(history.train_loss) == 1 and history.train_loss[0] == history.train_loss[0]  # finite, no NaN


def test_graph_replay_follows_a_scheduler_without_recapturing() -> None:
    from igl.core._graph import GraphRunner

    captured: list[GraphRunner] = []
    original = MatryoshkaTrainer._graph_runner  # noqa: SLF001

    def spy(self: MatryoshkaTrainer, *args: object, **kwargs: object) -> GraphRunner | None:
        runner = original(self, *args, **kwargs)  # type: ignore[arg-type]
        if runner is not None:
            captured.append(runner)
        return runner

    MatryoshkaTrainer._graph_runner = spy  # type: ignore[method-assign]  # noqa: SLF001
    try:
        eager, _ = _fit(cuda_graphs=False, epochs=4, scheduler="cosine_warm_restarts")
        graph, _ = _fit(cuda_graphs=True, epochs=4, scheduler="cosine_warm_restarts")
    finally:
        MatryoshkaTrainer._graph_runner = original  # type: ignore[method-assign]  # noqa: SLF001
    assert captured[-1].captures == 1
    for a, b in zip(eager, graph, strict=True):
        assert abs(a - b) <= 1e-3 * max(1.0, abs(a)), (eager, graph)
