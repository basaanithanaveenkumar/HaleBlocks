"""Ray Train backend: config, worker context, launcher and the full Trainer on 2 workers."""

import sys
from pathlib import Path

import pytest

from hale_core.config.sections import RayConfig
from hale_core.distributed import RayLauncher, RayReporter, WorkerContext


def test_config_defaults_and_backend_resolution():
    assert RayConfig().enabled is False
    assert RayConfig().resolved_backend() == "gloo"
    assert RayConfig(use_gpu=True).resolved_backend() == "nccl"
    assert RayConfig(resources_per_worker={"GPU": 0.5}).resolved_backend() == "nccl"
    assert RayConfig(backend="gloo", use_gpu=True).resolved_backend() == "gloo"
    with pytest.raises(ValueError):
        RayConfig(num_workers=0)


def test_worker_context_outside_ray(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    ctx = WorkerContext.current()
    assert (ctx.rank, ctx.world_size, ctx.in_ray, ctx.is_main) == (3, 4, False, False)


def test_reporter_is_a_noop_outside_ray(tmp_path):
    reporter = RayReporter()
    assert not reporter.active
    reporter.report({"loss": 1.0}, checkpoint_dir=tmp_path)  # must not raise


def test_storage_path_is_absolute_and_uris_pass_through():
    assert Path(RayLauncher(RayConfig(storage_path="rel/dir")).storage_path).is_absolute()
    assert (
        RayLauncher(RayConfig(storage_path="s3://bucket/runs")).storage_path == "s3://bucket/runs"
    )


# ---------------------------------------------------------------------------- real Ray runs
ray = pytest.importorskip("ray")

SUPPORT_DIR = Path(__file__).resolve().parents[1] / "ray_support"
sys.path.insert(0, str(SUPPORT_DIR.parent))
from ray_support import mock_training  # noqa: E402


@pytest.fixture(scope="module")
def local_ray():
    # Workers import the helpers from ray_support (pytest test modules are not importable there).
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"py_modules": [str(SUPPORT_DIR)]},
    )
    yield
    ray.shutdown()


@pytest.mark.integration
def test_launcher_runs_two_workers_with_torch_distributed(local_ray, tmp_path):
    result = RayLauncher(RayConfig(enabled=True, num_workers=2, storage_path=str(tmp_path))).launch(
        mock_training.report_rank, {"expect": 2}
    )
    assert result.ok and result.metrics["rank_sum"] == 1.0 and result.metrics["world"] == 2


@pytest.mark.integration
def test_hale_trainer_trains_on_ray(local_ray, tmp_path):
    """hale_core.training.train() with train.ray.enabled: 2 DDP workers, checkpoint reported."""
    from hale_core.config.run import RunConfig
    from hale_core.training import train

    mock_training.register_mock_variant()
    cfg = RunConfig(variant=mock_training.VARIANT, device="cpu")
    cfg.train.steps = 4
    cfg.train.batch_size = 2
    cfg.train.resume = False
    cfg.train.checkpoint_path = str(tmp_path / "ckpt" / "last.pt")
    cfg.logging.backend = "noop"
    cfg.logging.log_file = None
    cfg.experiment.enabled = False
    cfg.eval.every_n_epochs = None
    cfg.viz.enabled = False
    cfg.train.ray = RayConfig(enabled=True, num_workers=2, storage_path=str(tmp_path / "ray"))
    result = train(
        cfg,
        mock_training.MockTokenizer(),
        data_module_factory=mock_training.make_data_module,
        evaluator_factory=mock_training.MockEvaluator,
    )
    assert result.ok and result.metrics["epoch"] >= 0
    assert result.checkpoint_path is not None and Path(result.checkpoint_path, "last.pt").is_file()
