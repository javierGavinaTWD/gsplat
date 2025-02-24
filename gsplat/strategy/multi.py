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

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, any]:
        state = {state_id: None for state_id in self.config.initialized_state_ids}

        if "binoms" in state:
            n_max = self.config.n_max_binoms
            binoms = torch.zeros((n_max, n_max), dtype=torch.float32)
            for n in range(n_max):
                for k in range(n + 1):
                    binoms[n, k] = math.comb(n, k)

            state["binoms"] = binoms.to(torch.device("cuda"))

        if "scene_scale" in state:
            state["scene_scale"] = scene_scale

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
        lr: float,
        packed: bool = False,
    ):
        if step >= self.config.end_post_backward_steps:
            return

        self._update_state(params, state, info, packed=packed)

        # Cloning process
        if self.config.f_can_clone and (
            not self.config.f_limit_gaussians
            or len(params["means"]) < self.config.max_gaussians
        ):
            if (
                step > self.config.start_cloning_steps
                and step % self.config.cloning_interval == 0
                and step <= self.config.end_cloning_steps
            ):
                n_cloned = self._clone_gs(params, optimizers, state, step)

                if self.config.verbose:
                    print(f"Step {step}: {n_cloned} GSs cloned")

        # Splitting process
        if self.config.f_can_split and (
            not self.config.f_limit_gaussians
            or len(params["means"]) < self.config.max_gaussians
        ):
            if (
                step > self.config.start_splitting_steps
                and step % self.config.splitting_interval == 0
                and step <= self.config.end_splitting_steps
            ):
                n_splitted = self._split_gs(params, optimizers, state, step, n_cloned)

                if self.config.verbose:
                    print(f"Step {step}: {n_splitted} GSs splitted")

        # Relocation process
        if self.config.f_can_relocate:
            if (
                step > self.config.start_relocation_steps
                and step % self.config.relocation_interval == 0
                and step <= self.config.end_relocation_steps
            ):
                assert (
                    "binoms" in state
                ), "binoms is required for relocation but missing"

                n_relocated = self._relocate_gs(params, optimizers, state)

                if self.config.verbose:
                    print(f"Step {step}: {n_relocated} GSs relocated")

        # Add multinomial samples process
        if self.config.f_can_add_samples and (
            not self.config.f_limit_gaussians
            or len(params["means"]) < self.config.max_gaussians
        ):
            if (
                step > self.config.start_add_samples_steps
                and step % self.config.add_samples_interval == 0
                and step <= self.config.end_add_samples_steps
            ):
                assert (
                    "binoms" in state
                ), "binoms is required in state for relocation but missing"

                n_new_gs = self._add_new_gs(params, optimizers, state)

                if self.config.verbose:
                    print(f"Step {step}: {n_new_gs} GSs added")

        # Pruning process
        if self.config.f_can_prune:
            if (
                step > self.config.start_pruning_steps
                and step % self.config.pruning_interval == 0
                and step <= self.config.end_pruning_steps
            ):

                n_prune = self._prune_gs(params, optimizers, state, step)

                if self.config.verbose:
                    print(f"Step {step}: {n_prune} GSs removed")

        if self.config.verbose and step % self.config.print_number_gs_every == 0:
            print(f"Now having {len(params['means'])} GSs.")

        if self.config.f_can_inject_noise:
            if (
                step > self.config.start_inject_noise_steps
                and step % self.config.inject_noise_interval == 0
                and step <= self.config.end_inject_noise_steps
            ):
                inject_noise_to_position(
                    params=params,
                    optimizers=optimizers,
                    state={},
                    scaler=lr * self.config.noise_lr,
                )

        if self.config.f_can_opacity_reset:
            if (
                step > self.config.start_opacity_reset_steps
                and step % self.config.opacity_reset_interval == 0
                and step <= self.config.end_opacity_reset_steps
            ):
                reset_opa(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    value=self.config.min_opa_prune * 2.0,
                )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, any],
        info: Dict[str, any],
        packed: bool = False,
    ):

        # Check for Default
        if any(
            word
            in [
                "grad2d_clone",
                "grad2d_split",
                "count_clone",
                "count_split",
                "radii",
                "sqrgrad",
            ]
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

            if self.config.sqrgrad:
                sqrgrads = info["means2d"].sqrgrad.clone()

            grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
            grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

            # initialize state on the first run
            n_gaussian = len(list(params.values())[0])

            if (
                state["grad2d_clone"] is None
                and "grad2d_clone" in self.config.updatable_state_ids
            ):
                state["grad2d_clone"] = torch.zeros(n_gaussian, device=grads.device)

            if (
                state["count_clone"] is None
                and "count_clone" in self.config.updatable_state_ids
            ):
                state["count_clone"] = torch.zeros(n_gaussian, device=grads.device)

            if (
                state["grad2d_split"] is None
                and "grad2d_split" in self.config.updatable_state_ids
            ):
                state["grad2d_split"] = torch.zeros(n_gaussian, device=grads.device)

            if (
                state["count_split"] is None
                and "count_split" in self.config.updatable_state_ids
            ):
                state["count_split"] = torch.zeros(n_gaussian, device=grads.device)

            if (
                self.config.refine_scale2d_stop_iter > 0
                and state["radii"] is None
                and "radii" in self.config.updatable_state_ids
            ):
                assert "radii" in info, "radii is required but missing."
                state["radii"] = torch.zeros(n_gaussian, device=grads.device)

            if (
                self.config.sqrgrad
                and state["sqrgrad"] is None
                and "sqrgrad" in self.config.updatable_state_ids
            ):
                state["sqrgrad"] = torch.zeros(n_gaussian, device=grads.device)

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
                if self.config.sqrgrad:
                    sqrgrads = sqrgrads[sel]

            if "grad2d_clone" in self.config.updatable_state_ids:
                state["grad2d_clone"].index_add_(0, gs_ids, grads.norm(dim=-1))

            if "count_clone" in self.config.updatable_state_ids:
                state["count_clone"].index_add_(
                    0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
                )

            if "grad2d_split" in self.config.updatable_state_ids:
                state["grad2d_split"].index_add_(0, gs_ids, grads.norm(dim=-1))

            if "count_split" in self.config.updatable_state_ids:
                state["count_split"].index_add_(
                    0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
                )

            if "sqrgrad" in self.config.updatable_state_ids:
                state["sqrgrad"].index_add_(0, gs_ids, torch.sum(sqrgrads, dim=-1))

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

    @torch.no_grad()
    def _clone_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, any],
        step: int,
    ) -> int:

        assert (
            "count_clone" in state
        ), "count_clone is required for cloning but missing."
        assert (
            "grad2d_clone" in state
        ), "grad2d_clone is required but missing for cloning but missing."
        count = state["count_clone"]
        grads = state["grad2d_clone"] / count.clamp_min(1)

        is_grad_high = grads > self.config.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.config.grow_scale3d * state["scene_scale"]
        )

        is_clone = is_grad_high & is_small
        n_clone = is_clone.sum().item()
        if n_clone > 0:
            clone(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_clone,
                num_clones=self.config.num_cloned_gaussians,
            )
            self._reset_state("grad2d_clone", state)
            self._reset_state("count_clone", state)
        return n_clone

    @torch.no_grad()
    def _split_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, any],
        step: int,
        n_cloned: int = 0,
    ) -> int:

        assert (
            "count_split" in state
        ), "count_split is required for cloning but missing."
        assert (
            "grad2d_split" in state
        ), "grad2d_split is required but missing for cloning but missing."
        count = state["count_split"]
        grads = state["grad2d_split"] / count.clamp_min(1)

        is_grad_high = grads > self.config.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.config.grow_scale3d * state["scene_scale"]
        )
        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.config.refine_scale2d_stop_iter and "radii" in state:
            is_split |= state["radii"] > self.config.grow_scale2d

        # mark the positions or the cloned gaussians to NOT be splitted
        is_split[-n_cloned:] = False
        n_split = is_split.sum().item()

        # then split
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                num_splits=self.config.num_splitted_gaussians,
                revised_opacity=self.config.f_revised_opacity,
            )
            self._reset_state("count_split", state)
            self._reset_state("grad2d_split", state)

        return n_split

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, any],
    ) -> int:
        opacities = torch.sigmoid(params["opacities"].flatten())
        dead_mask = opacities <= self.config.min_opa_relocate
        n_relocate = dead_mask.sum().item()
        if n_relocate > 0:
            relocate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=dead_mask,
                binoms=state["binoms"],
                min_opacity=self.config.min_opa_relocate,
            )
        return n_relocate

    @torch.no_grad()
    def _add_new_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, any],
    ) -> int:
        current_n_points = len(params["means"])
        n_target = min(
            self.config.max_gaussians,
            int((1 + self.config.add_sample_rate) * current_n_points),
        )
        n_add = max(0, n_target - current_n_points)
        if n_add > 0:
            sample_add(
                params=params,
                optimizers=optimizers,
                state=state,
                n=n_add,
                binoms=state["binoms"],
                min_opacity=self.config.min_opa_relocate,
            )
        return n_add

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, any],
        step: int,
    ) -> int:

        is_prune = torch.zeros_like(
            params["opacities"].flatten(), dtype=torch.bool, device="cuda"
        )
        if self.config.f_can_prune_if_opacity_low:
            is_opa_low = (
                torch.sigmoid(params["opacities"].flatten()) < self.config.min_opa_prune
            )
            is_prune = is_prune | is_opa_low

        if step > self.config.prune_warmup_steps:
            if self.config.f_can_prune_if_too_big:
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
                self._reset_state("radii", state)

            if self.config.f_can_prune_if_sqrgrad_low:
                if step % self.config.prune_sqrgrad_interval == 0:
                    sqrgrad_values = state["sqrgrad"]

                    threshold = torch.quantile(
                        sqrgrad_values, self.config.prune_sqrgrad_rate
                    )
                    not_utilized = sqrgrad_values <= threshold
                    n_u = not_utilized.sum().item()
                    if self.config.verbose:
                        print(f"Step {step}: {n_u} GSs removed by sqrgrad.")
                    is_prune = is_prune | not_utilized
                    self._reset_state("sqrgrad", state)

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune

    def _reset_state(self, state_id: str, state: Dict[str, any]):
        if state_id in self.config.resetable_state_ids and state_id in state:
            state[state_id].zero_()
            torch.cuda.empty_cache()
