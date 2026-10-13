"""Whole-candidate actions and separate joint-only diagnostic controls."""
import math
import re
from dataclasses import dataclass, field

import torch

from .body import observation_points
from .geometry_observation import observation_errors
from .tangent import complete_distances


@dataclass
class Decision:
    method: str
    selected_id: str | None
    selected_index: int | None
    costs: torch.Tensor
    ambiguity: bool
    risk_gap: float | None
    reasons: list
    diagnostic: dict = field(default_factory=dict)

    def human(self, store):
        return None if self.selected_index is None else store.select(self.selected_index)


def stable_min(costs, indices, ids):
    if len(indices) == 0:
        return None
    # Selection is discrete already. Read each tensor once instead of forcing
    # a GPU-to-host scalar synchronization for every candidate and sort key.
    host_costs = costs.detach().cpu().tolist()
    host_indices = indices.detach().cpu().tolist()
    finite = [i for i in host_indices if math.isfinite(host_costs[i])]
    return min(finite, key=lambda i: (host_costs[i], ids[i])) if finite else None


def diagnostic_joint_controls(points, per_joint_errors, mask, *, ids, topk=3, temperature=1.0):
    """GT-free raw-fisheye JPMA/RPEA adaptation, absolute candidate-owned T.

    Mirrors selection/top-k engineering weights in the reviewed author function,
    replacing its GT-root camera projection and root-relative output. COCO17 surface
    points have no artificial zero-root. These outputs are not SMPL-H humans.
    Source: third_party/architecture_reproduction/FMPose3D/fmpose3d/aggregation_methods.py
    """
    if temperature <= 0 or topk < 1:
        raise ValueError("positive RPEA temperature/topk required")
    joints = points.new_full((17, 3), torch.nan)
    weighted = joints.clone()
    selected_ids, weights = [], points.new_zeros(len(points), 17)
    for joint in range(17):
        eligible = [int(i) for i in torch.where(mask)[0] if torch.isfinite(per_joint_errors[i, joint])]
        eligible.sort(key=lambda i: (float(per_joint_errors[i, joint]), ids[i]))
        if not eligible:
            selected_ids.append(None)
            continue
        selected_ids.append(ids[eligible[0]])
        joints[joint] = points[eligible[0], joint]
        indices = eligible[:topk]
        costs = per_joint_errors[indices, joint].clamp(max=temperature*20)
        w = torch.softmax(-costs/temperature, 0)
        weights[indices, joint] = w
        weighted[joint] = (points[indices, joint]*w[:, None]).sum(0)
    return {"jpma_coco17_absolute": joints, "rpea_coco17_absolute": weighted,
            "joint_candidate_ids": selected_ids, "engineering_weights_not_posterior": weights,
            "scope": "common_surface_joint_metrics_only_no_mesh_or_parameter_output"}


def decide(store, method, scale, *, observation=None, cameras=None, noise=None, surface=None,
           lambda_g=0.0):
    if method not in ("A0", "A1", "A2", "A3", "A4", "A5"):
        raise ValueError("unimplemented or unregistered aggregation method")
    if lambda_g < 0:
        raise ValueError("nonnegative fixed geometry coefficient required")
    legal = torch.where(store.numerical)[0]
    actions = torch.where(store.actions)[0]
    costs = store.raw.new_full((len(store.raw),), torch.inf)
    if len(legal) == 0 or len(actions) == 0:
        return Decision(method, None, None, costs, True, None, ["empty_action_set"])
    diagnostic = {}
    needs_geometry = method in ("A2", "A3") or (method == "A5" and lambda_g > 0)
    if needs_geometry and any(v is None for v in (observation, cameras, noise, surface)):
        raise ValueError("fixed original observations, rig, noise and surface operator required")
    if method == "A1" or needs_geometry:
        if surface is None:
            raise ValueError("surface operator required for joint coordinate diagnostic")
        points = store.raw.new_full((len(store.raw), 17, 3), torch.nan)
        points[legal] = observation_points(surface, store.rotations[legal], store.translation[legal], store.beta[legal])
    if needs_geometry:
        errors, per_joint, _, _ = observation_errors(points, observation, cameras, noise)
        if method == "A3":
            diagnostic = diagnostic_joint_controls(points, per_joint, store.actions, ids=store.ids)
        elif method == "A2":
            costs[actions] = errors[actions]
    if method == "A1":
        diagnostic = {"coordinate_mean_coco17_absolute": points[legal].mean(0),
                      "scope": "diagnostic_only_no_human_publication"}
    if method in ("A1", "A3"):
        return Decision(method, None, None, costs, True, None, ["joint_only_diagnostic"], diagnostic)
    if method == "A0":
        def original_order(index):
            tail = re.search(r"(\d+)$", store.ids[index])
            return (int(tail.group(1)) if tail else 0, store.ids[index])
        original_ids = store.provenance.get("original_candidate_ids")
        if original_ids is not None and original_ids[0] not in store.ids:
            return Decision(method, None, None, costs, True, None, ["first_candidate_absent_from_pool"])
        selected = store.ids.index(original_ids[0]) if original_ids else min(range(len(store.raw)), key=original_order)
        if not store.actions[selected]:
            return Decision(method, None, None, costs, True, None, ["first_candidate_not_action"])
        costs[selected] = 0
    elif method in ("A4", "A5"):
        # Expectation over all legal raw candidates, including non-action candidates.
        reduced = store.permuted(legal)
        distance_squared = complete_distances(reduced, scale)
        # Medoid minimizes distances; isotropic quadratic risk minimizes squared distances.
        risk = distance_squared.clamp_min(0).sqrt().mean(1) if method == "A4" else distance_squared.mean(1)
        costs[legal] = risk
        # Avoid 0*inf; projection remains a failed action when geometry is used.
        if method == "A5" and lambda_g > 0:
            costs[legal] += lambda_g*errors[legal]
        selected = stable_min(costs, actions, store.ids)
    else:
        selected = stable_min(costs, actions, store.ids)
    if selected is None:
        return Decision(method, None, None, costs, True, None, ["all_action_costs_nonfinite"])
    ranked = torch.sort(costs[actions][torch.isfinite(costs[actions])]).values
    gap = float(ranked[1]-ranked[0]) if len(ranked) > 1 else None
    diagnostic.update({"decision_near_tie": gap is None or gap <= 1e-8,
                       "physical_ambiguity": "unassessed; original multimodal candidates retained"})
    return Decision(method, store.ids[selected], selected, costs, True,
                    gap, [], diagnostic)
