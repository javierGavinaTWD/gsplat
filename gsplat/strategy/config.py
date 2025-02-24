import yaml
from dataclasses import dataclass, field, asdict
from typing import Literal, Dict, Any, List


@dataclass
class DensificationStrategyConfig:
    """
    Configuration for Gaussian Splatting Densification Strategy.

    Defines parameters controlling cloning, splitting, relocation, pruning,
    and other strategies used for Gaussian splatting.
    """

    # ─────────────────────────────────────────────────────────
    # 🔹 Strategy Selection
    # ─────────────────────────────────────────────────────────
    strategy: Literal["default", "mcmc", "custom"] = (
        "default"  # Preset strategy selection
    )

    # ─────────────────────────────────────────────────────────
    # 🔹 General Parameters
    # ─────────────────────────────────────────────────────────
    f_limit_gaussians: bool = True  # Enforce a maximum limit on Gaussians
    max_gaussians: int = 1_000_000  # Maximum number of Gaussians allowed
    end_post_backward_steps: int = 25_000  # Total training steps before stopping

    # ─────────────────────────────────────────────────────────
    # 🔹 Cloning Configuration
    # ─────────────────────────────────────────────────────────
    f_can_clone: bool = True  # Enable Gaussian cloning
    start_cloning_steps: int = 0  # Step at which cloning starts
    end_cloning_steps: int = 0  # Step at which cloning ends
    cloning_interval: int = 0  # Interval between cloning events

    # ─────────────────────────────────────────────────────────
    # 🔹 Splitting Configuration
    # ─────────────────────────────────────────────────────────
    f_can_split: bool = True  # Enable Gaussian splitting
    start_splitting_steps: int = 0  # Step at which splitting starts
    end_splitting_steps: int = 0  # Step at which splitting ends
    splitting_interval: int = 0  # Interval between splitting events
    f_revised_opacity: bool = False  # Apply revised opacity computation on split

    # ─────────────────────────────────────────────────────────
    # 🔹 Relocation Configuration
    # ─────────────────────────────────────────────────────────
    f_can_relocate: bool = True  # Enable Gaussian relocation
    start_relocation_steps: int = 0  # Step at which relocation starts
    end_relocation_steps: int = 0  # Step at which relocation ends
    relocation_interval: int = 0  # Interval between relocation events

    # ─────────────────────────────────────────────────────────
    # 🔹 Adding New Samples
    # ─────────────────────────────────────────────────────────
    f_can_add_samples: bool = True  # Enable adding new Gaussian samples
    start_add_samples_steps: int = 0  # Step at which sample addition starts
    end_add_samples_steps: int = 0  # Step at which sample addition ends
    add_samples_interval: int = 0  # Interval between sample addition events

    # ─────────────────────────────────────────────────────────
    # 🔹 Pruning Configuration
    # ─────────────────────────────────────────────────────────
    f_can_prune: bool = True  # Enable pruning of Gaussians
    start_pruning_steps: int = 0  # Step at which pruning starts
    end_pruning_steps: int = 0  # Step at which pruning ends
    pruning_interval: int = 0  # Interval between pruning events
    f_can_prune_if_opacity_low: bool = True  # Prune low-opacity Gaussians
    f_can_prune_if_too_big: bool = True  # Prune overly large Gaussians
    f_can_prune_if_too_big2d: bool = False  # Prune overly large Gaussians in 2D
    f_can_prune_if_sqrgrad_low: bool = (
        False  # Prune based on squared gradient threshold
    )
    prune_sqrgrad_interval: int = 10_000  # Interval for squared gradient pruning
    prune_sqrgrad_rate: float = 0.80  # Threshold for squared gradient pruning

    # ─────────────────────────────────────────────────────────
    # 🔹 Opacity Reset Configuration
    # ─────────────────────────────────────────────────────────
    f_can_opacity_reset: bool = True  # Enable opacity reset
    start_opacity_reset_steps: int = 0  # Step at which opacity reset starts
    end_opacity_reset_steps: int = 0  # Step at which opacity reset ends
    opacity_reset_interval: int = 0  # Interval between opacity resets

    # ─────────────────────────────────────────────────────────
    # 🔹 Noise Injection Configuration
    # ─────────────────────────────────────────────────────────
    f_can_inject_noise: bool = True  # Enable noise injection
    start_inject_noise_steps: int = 0  # Step at which noise injection starts
    end_inject_noise_steps: int = 0  # Step at which noise injection ends
    inject_noise_interval: int = 0  # Interval between noise injections

    # ─────────────────────────────────────────────────────────
    # 🔹 State Management
    # ─────────────────────────────────────────────────────────
    initialized_state_ids: List[
        Literal[
            "grad2d_clone",
            "grad2d_split",
            "count_clone",
            "count_split",
            "radii",
            "scene_scale",
            "binoms",
            "sqrgrad",
        ]
    ] = field(default_factory=list)
    updatable_state_ids: List[
        Literal[
            "grad2d_clone",
            "grad2d_split",
            "count_clone",
            "count_split",
            "radii",
            "scene_scale",
            "binoms",
            "sqrgrad",
        ]
    ] = field(default_factory=list)
    resetable_state_ids: List[
        Literal[
            "grad2d_clone",
            "grad2d_split",
            "count_clone",
            "count_split",
            "radii",
            "scene_scale",
            "binoms",
            "sqrgrad",
        ]
    ] = field(default_factory=list)

    # ─────────────────────────────────────────────────────────
    # 🔹 Debugging & Logging
    # ─────────────────────────────────────────────────────────
    verbose: bool = False  # Enable verbose logging
    print_number_gs_every: int = 100  # Print Gaussian count every N steps

    # ─────────────────────────────────────────────────────────
    # 🔹 Strategy-Specific Flags
    # ─────────────────────────────────────────────────────────
    absgrad: bool = False  # Use absolute gradient
    sqrgrad: bool = False  # Use squared gradient
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = (
        "means2d"  # Gradient selection
    )

    # ─────────────────────────────────────────────────────────
    # 🔹 Pruning & Splitting Configuration
    # ─────────────────────────────────────────────────────────
    min_opa_prune: float = 0.005  # Minimum opacity threshold for pruning
    grow_grad2d: float = 0.0002  # Growth rate for 2D gradients
    grow_scale3d: float = 0.01  # Growth rate for 3D scaling
    grow_scale2d: float = 0.05  # Growth rate for 2D scaling
    prune_scale3d: float = 0.1  # Scale pruning threshold (3D)
    prune_scale2d: float = 0.15  # Scale pruning threshold (2D)
    refine_scale2d_stop_iter: int = 0  # Stop refinement after this iteration
    prune_warmup_steps: int = (
        3_000  # Number of warmup steps before pruning by scale or score
    )
    pause_refine_after_reset: int = 0  # Pause refinement after a reset
    num_splitted_gaussians: int = 2  # Number of Gaussians created per split
    num_cloned_gaussians: int = 2  # Number of Gaussians created per clone

    # ─────────────────────────────────────────────────────────
    # 🔹 MCMC-Specific Parameters
    # ─────────────────────────────────────────────────────────
    noise_lr: float = 5e5  # Learning rate for noise
    n_max_binoms: int = 51  # Maximum number of binomial samples
    min_opa_relocate: float = 0.005  # Minimum opacity for relocation
    add_sample_rate: float = 0.05  # Sample addition rate

    def apply_strategy_defaults(self, overrides: Dict[str, Any] = None):
        """Set default values for the selected strategy and override with user-defined values."""

        if self.strategy is None:
            raise ValueError("Strategy must be set before applying defaults.")

        strategy_defaults = {
            "default": {
                # Feature Flags
                "f_limit_gaussians": False,
                "f_can_clone": True,
                "f_can_split": True,
                "f_revised_opacity": False,
                "f_can_prune": True,
                "f_can_prune_if_opacity_low": True,
                "f_can_prune_if_too_big": True,
                "f_can_opacity_reset": True,
                # Thresholds & Scaling Factors
                "min_opa_prune": 0.005,
                "grow_grad2d": 0.0002,
                "grow_scale3d": 0.01,
                "grow_scale2d": 0.05,
                "prune_scale3d": 0.1,
                "prune_scale2d": 0.15,
                "refine_scale2d_stop_iter": 0,
                # Step Control
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
                "start_opacity_reset_steps": 3000,
                "end_opacity_reset_steps": 15_000,
                "opacity_reset_interval": 3000,
                "prune_warmup_steps": 3_000,
                "pause_refine_after_reset": 0,
                # Logging & Debugging
                "verbose": False,
                "print_number_gs_every": 100,
                # Gradient Options
                "absgrad": False,
                "sqrgrad": False,
                "key_for_gradient": "means2d",
                # State Management
                "initialized_state_ids": [
                    "grad2d_clone",
                    "count_clone",
                    "grad2d_split",
                    "count_split",
                    "scene_scale",
                ],
                "updatable_state_ids": [
                    "grad2d_clone",
                    "count_clone",
                    "grad2d_split",
                    "count_split",
                ],
                "resetable_state_ids": [
                    "grad2d_clone",
                    "count_clone",
                    "grad2d_split",
                    "count_split",
                ],
                # Gaussian Properties
                "num_splitted_gaussians": 2,
                "num_cloned_gaussians": 2,
            },
            "mcmc": {
                # Feature Flags
                "f_limit_gaussians": True,
                "f_can_relocate": True,
                "f_can_add_samples": True,
                "f_can_inject_noise": True,
                # Limits & Constraints
                "max_gaussians": 1_000_000,
                "min_opa_relocate": 0.005,
                "n_max_binoms": 51,
                # Learning Rates
                "noise_lr": 5e5,
                # Step Control
                "end_post_backward_steps": 25_000,
                "start_relocation_steps": 500,
                "end_relocation_steps": 25_000,
                "relocation_interval": 100,
                "start_add_samples_steps": 500,
                "end_add_samples_steps": 25_000,
                "add_samples_interval": 100,
                "start_inject_noise_steps": 500,
                "end_inject_noise_steps": 25_000,
                "inject_noise_interval": 100,
                "add_sample_rate": 0.05,
                # State Management
                "initialized_state_ids": ["binoms"],
                "updatable_state_ids": [],
                "resetable_state_ids": [],
                # Logging
                "print_number_gs_every": 100,
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
