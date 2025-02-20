import math
from dataclasses import dataclass
import torch

from .base import Strategy
from .ops import *
from .config import DensificationStrategyConfig


@dataclass
class MultiStrategy(Strategy):
    config: DensificationStrategyConfig = DensificationStrategyConfig(
        strategy="default"
    )

    def __post_init__(self):
        self.config.apply_strategy_defaults()

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, any]:
        state = {state_id: None for state_id in self.config.initialized_state_ids}

        if "binoms" in state:
            n_max = 51
            binoms = torch.zeros((n_max, n_max), dtype=torch.float32)
            for n in range(n_max):
                for k in range(n + 1):
                    binoms[n, k] = math.comb(n, k)

            state["binoms"] = binoms

        if "scene_scale" in state:
            state["scene_scale"] = scene_scale

        if "radii" in state and self.config.refine_scale2d_stop_iter > 0:
            state["radii"] = None

        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, any],
        step: int,
        info: Dict[str, any],
    ):
        if self.config.key_for_gradient is None:
            return

        assert (
            self.config.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.config.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, any],
        step: int,
        info: Dict[str, any],
        packed: bool = False,
    ):
        if step >= self.config.end_post_backward_steps:
            return

        self._update_state(params, state, info, packed=packed)

        # Cloning process
        if self.config.start_cloning_steps is not None:
            if (
                step > self.config.start_cloning_steps
                and step % self.config.cloning_interval
                and step <= self.config.end_cloning_steps
            ):
                n_cloned = self._clone_gs(params, optimizers, state, step)

                if self.config.verbose:
                    print(
                        f"Step {step}: {n_cloned} GSs cloned"
                        f"Now having {len(params['means'])} GSs."
                    )

        # Splitting process
        if self.config.start_splitting_steps is not None:
            if (
                step > self.config.start_splitting_steps
                and step % self.config.splitting_interval
                and step <= self.config.end_splitting_steps
            ):
                n_splitted = self._split_gs(params, optimizers, state, step, n_cloned)

                if self.config.verbose:
                    print(
                        f"Step {step}: {n_splitted} GSs splitted"
                        f"Now having {len(params['means'])} GSs."
                    )

        # Relocation process
        if self.config.start_relocation_steps is not None:
            if (
                step > self.config.start_relocation_steps
                and step % self.config.relocation_interval
                and step <= self.config.end_relocation_steps
            ):
                assert (
                    "binoms" in state
                ), "binoms is required for relocation but missing"
                binoms = state["binoms"]

                n_relocated = self._relocate_gs(params, optimizers, binoms)

                if self.config.verbose:
                    print(
                        f"Step {step}: {n_relocated} GSs relocated"
                        f"Now having {len(params['means'])} GSs."
                    )

        # Add multinomial samples process
        if self.config.start_add_samples_steps is not None:
            if (
                step > self.config.start_add_samples_steps
                and step % self.config.add_samples_interval
                and step <= self.config.end_add_samples_steps
            ):
                assert (
                    "binoms" in state
                ), "binoms is required for relocation but missing"
                binoms = state["binoms"]

                n_new_gs = self._add_new_gs(params, optimizers, binoms)

                if self.config.verbose:
                    print(
                        f"Step {step}: {n_new_gs} GSs added"
                        f"Now having {len(params['means'])} GSs."
                    )

        # Pruning process
        if self.config.start_pruning_steps is not None:
            if (
                step > self.config.start_pruning_steps
                and step % self.config.pruning_interval
                and step <= self.config.end_pruning_steps
            ):

                n_prune = self._prune_gs(params, optimizers, state, step)

                if self.config.verbose:
                    print(
                        f"Step {step}: {n_new_gs} GSs added"
                        f"Now having {len(params['means'])} GSs."
                    )

        self._reset_state()

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, any],
        info: Dict[str, any],
        packed: bool = False,
    ):

        # Check for Default
        if any(
            word in ["grad2d", "count", "radii"]
            for word in self.config.updatable_state_ids
        ):
            for key in [
                "width",
                "height",
                "n_cameras",
                "radii",
                "gaussian_ids",
                self.config.key_for_gradient,
            ]:
                assert key in info, f"{key} is required but missing."

            if self.config.absgrad:
                grads = info[self.config.key_for_gradient].absgrad.clone()
            else:
                grads = info[self.config.key_for_gradient].grad.clone()

            grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
            grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

            # initialize state on the first run
            n_gaussian = len(list(params.values())[0])

            if state["grad2d"] is None and "grad2d" in self.config.updatable_state_ids:
                state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)

            if state["count"] is None and "count" in self.config.updatable_state_ids:
                state["count"] = torch.zeros(n_gaussian, device=grads.device)

            if (
                self.config.refine_scale2d_stop_iter > 0
                and state["radii"] is None
                and "radii" in self.config.updatable_state_ids
            ):
                assert "radii" in info, "radii is required but missing."
                state["radii"] = torch.zeros(n_gaussian, device=grads.device)

            # update the running state
            if packed:
                # grads is [nnz, 2]
                gs_ids = info["gaussian_ids"]  # [nnz]
                radii = info["radii"]  # [nnz]
            else:
                # grads is [C, N, 2]
                sel = info["radii"] > 0.0  # [C, N]
                gs_ids = torch.where(sel)[1]  # [nnz]
                grads = grads[sel]  # [nnz, 2]
                radii = info["radii"][sel]  # [nnz]

            if "grad2d" in self.config.updatable_state_ids:
                state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))

            if "count" in self.config.updatable_state_ids:
                state["count"].index_add_(
                    0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
                )

            if (
                self.config.refine_scale2d_stop_iter > 0
                and "radii" in self.config.updatable_state_ids
            ):
                # Should be ideally using scatter max
                state["radii"][gs_ids] = torch.maximum(
                    state["radii"][gs_ids],
                    # normalize radii to [0, 1] screen space
                    radii / float(max(info["width"], info["height"])),
                )

    @torch.no_grad
    def _clone_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, any],
        step: int,
    ) -> int:

        assert "count" in state, "count is required for cloning but missing."
        assert (
            "grad2d" in state
        ), "grad2d is required but missing for cloning but missing."
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.config.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.config.grow_scale3d * state["scene_scale"]
        )

        is_cloned = is_grad_high & is_small
        n_cloned = is_cloned.sum().item()
        if n_cloned > 0:
            clone(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_cloned,
                num_clones=self.config.num_cloned_gaussians,
            )
        return n_cloned

    @torch.no_grad
    def _split_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, any],
        step: int,
        n_cloned: int = 0,
    ) -> int:

        assert "count" in state, "count is required for cloning but missing."
        assert (
            "grad2d" in state
        ), "grad2d is required but missing for cloning but missing."
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.config.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.config.grow_scale3d * state["scene_scale"]
        )
        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.config.refine_scale2d_stop_iter and "radii" in state:
            is_split |= state["radii"] > self.config.grow_scale2d
        n_split = is_split.sum().item()

        # new GSs added by duplication will not be split
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_cloned, dtype=torch.bool, device=device),
            ]
        )

        # then split
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                num_splits=self.config.num_splitted_gaussians,
                revised_opacity=self.config.revised_opacity,
            )

        return n_split

    @torch.no_grad
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        opacities = torch.sigmoid(params["opacities"].flatten())
        dead_mask = opacities <= self.config.min_opa_relocate
        n_relocate = dead_mask.sum().item()
        if n_relocate > 0:
            relocate(
                params=params,
                optimizers=optimizers,
                state={},
                mask=dead_mask,
                binoms=binoms,
                min_opacity=self.config.min_opa_relocate,
            )
        return n_relocate

    @torch.no_grad
    def _add_new_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        current_n_points = len(params["means"])
        n_target = min(self.config.max_gaussians, int(1.05 * current_n_points))
        n_add = max(0, n_target - current_n_points)
        if n_add > 0:
            sample_add(
                params=params,
                optimizers=optimizers,
                state={},
                n=n_add,
                binoms=binoms,
                min_opacity=self.config.min_opa_relocate,
            )
        return n_add

    @torch.no_grad
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, any],
        step: int,
    ) -> int:
        is_prune = (
            torch.sigmoid(params["opacities"].flatten()) < self.config.min_opa_prune
        )
        if step > self.config.reset_every:

            assert (
                "scene_scale" in state
            ), "scene_scale is required for pruning but missing."

            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.config.prune_scale3d * state["scene_scale"]
            )

            if step < self.config.refine_scale2d_stop_iter:
                assert (
                    "radii" in state
                ), "radii is required for screen-size pruning but missing"
                is_too_big |= state["radii"] > self.config.prune_scale2d

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune

    def _reset_state(self, state: Dict[str, any]):
        for state_id in self.config.resetable_state_ids:
            if state_id in state:
                state[state_id].zero_()
