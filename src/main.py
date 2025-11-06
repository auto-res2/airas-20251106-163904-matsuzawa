import os
import subprocess
import sys
import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    """Root orchestrator: spawns src.train with correct overrides."""
    orig_cwd = hydra.utils.get_original_cwd()

    run_cfg_path = os.path.join(orig_cwd, "config", "runs", f"{cfg.run}.yaml")
    if not os.path.exists(run_cfg_path):
        raise FileNotFoundError(f"Run config not found: {run_cfg_path}")

    # Mode-specific overrides
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training = cfg.get("training", {})
        cfg.training.epochs = 1
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError("mode must be 'trial' or 'full'")

    # Serialize cfg for debugging (optional env var)
    os.environ["HYDRA_FULL_CONFIG"] = OmegaConf.to_yaml(cfg)

    # Build command to spawn subprocess
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    print("Launching training subprocess:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
