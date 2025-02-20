import yaml
from dataclasses import dataclass, field, asdict
from typing import Literal, Dict, Any


@dataclass
class DensificationStrategyConfig:
    strategy: Literal["default", "mcmc", "custom"] = "custom"

    # General Parameters
    max_gaussians: int = 1_000_000
    max_steps: int = 30_000

    # Refining Interval
    # start_refine_steps: int = 500
    # end_refine_steps: int = 15_000
    # refine_interval: int = 100

    end_post_backward_steps: int = 15_000

    # Cloning Interval
    start_cloning_steps: int = 0
    end_cloning_steps: int = 0
    cloning_interval: int = 0

    # Splitting Interval
    start_splitting_steps: int = 0
    end_splitting_steps: int = 0
    splitting_interval: int = 0

    # Relocation Interval
    start_relocation_steps: int = 0
    end_relocation_steps: int = 0
    relocation_interval: int = 0

    # Adding samples Interval
    start_add_samples_steps: int = 0
    end_add_samples_steps: int = 0
    add_samples_interval: int = 0

    # Pruning Interval
    start_pruning_steps: int = 0
    end_pruning_steps: int = 0
    pruning_interval: int = 0

    # State Management
    initialized_state_ids: list[
        Literal["grad2d", "count", "radii", "scene_scale", "binoms"]
    ] = field(default_factory=list)
    updatable_state_ids: list[
        Literal["grad2d", "count", "radii", "scene_scale", "binoms"]
    ] = field(default_factory=list)
    resetable_state_ids: list[
        Literal["grad2d", "count", "radii", "scene_scale", "binoms"]
    ] = field(default_factory=list)

    verbose: bool = False

    # Strategy-Specific Flags
    absgrad: bool = False
    sqrgrad: bool = False

    # Gradient Configuration
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"

    # Pruning & Splitting Configuration
    min_opa_prune: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    reset_every: int = 3000
    pause_refine_after_reset: int = 0
    revised_opacity: bool = False
    num_splitted_gaussians: int = 2
    num_cloned_gaussians: int = 2

    # MCMC-Specific Parameters
    can_inject_noise: bool = False
    noise_lr: float = 5e5
    n_max_binoms: int = 51
    min_opa_relocate: float = 0.005

    def apply_strategy_defaults(self, overrides: Dict[str, Any] = None):
        """Set default values for the selected strategy and override with user-defined values."""

        if self.strategy is None:
            raise ValueError("Strategy must be set before applying defaults.")

        strategy_defaults = {
            "default": {
                "min_opa_prune": 0.005,
                "grow_grad2d": 0.0002,
                "grow_scale3d": 0.01,
                "grow_scale2d": 0.05,
                "prune_scale3d": 0.1,
                "prune_scale2d": 0.15,
                "refine_scale2d_stop_iter": 0,
                "end_post_backward_steps": 15_000,
                "start_cloning_steps": 500,
                "end_cloning_steps": 15_000,
                "cloning_interval": 100,
                "start_splitting_steps": 500,
                "end_splitting_steps": 15_000,
                "splitting_interval": 100,
                "start_pruning_steps": 500,
                "end_pruning_steps": 15_000,
                "pruning_interval": 100,
                "reset_every": 3000,
                "pause_refine_after_reset": 0,
                "absgrad": False,
                "sqrgrad": False,
                "revised_opacity": False,
                "verbose": False,
                "key_for_gradient": "means2d",
                "initialized_state_ids": ["grad2d", "count", "radii", "scene_scale"],
                "updatable_state_ids": ["grad2d", "count", "radii"],
                "resetable_state_ids": ["grad2d", "count", "radii"],
                "num_splitted_gaussians": 2,
                "num_cloned_gaussians": 2,
            },
            "mcmc": {
                "max_gaussians": 1_000_000,
                "noise_lr": 5e5,
                "end_post_backward_steps": 15_000,
                "start_relocation_steps": 500,
                "end_relocation_steps": 25_000,
                "relocation_interval": 100,
                "start_add_samples_steps": 500,
                "end_add_samples_steps": 25_000,
                "add_samples_interval": 100,
                "min_opa_relocate": 0.005,
                "n_max_binoms": 51,
                "initialized_state_ids": ["binoms"],
                "can_inject_noise": True,
                "updatable_state_ids": [],
                "resetable_state_ids": [],
            },
        }

        strategy = self.strategy

        if strategy in strategy_defaults and strategy != "custom":
            for field_name in asdict(self).keys():
                if field_name != "strategy":
                    setattr(self, field_name, None)

            for key, value in strategy_defaults[strategy].items():
                setattr(self, key, value)

        if overrides:
            for key, value in overrides.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def save_to_yaml(self, file_path: str):
        """Save the current configuration to a YAML file."""
        with open(file_path, "w") as file:
            yaml.dump(asdict(self), file, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_from_yaml(cls, file_path: str):
        """Load configuration from a YAML file and create an instance of the class."""
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        return cls(**data)


# config = DensificationStrategyConfig(strategy="custom")
# config.apply_strategy_defaults(overrides={"updatable_state_ids": ["radii"]})
# config.save_to_yaml("config.yaml")
