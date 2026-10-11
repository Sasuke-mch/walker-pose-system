"""Information risk and an isolated causal near-tie time policy, not a posterior."""
from dataclasses import dataclass, field

import torch

from realtime_app.learning_flow_mesh.rotations import validate_rotation_matrices

from .aggregate import Decision, decide, stable_min
from .geometry_observation import observation_errors
from .input_binding import require_information_binding
from .tangent import angles, pairwise_tangent


@dataclass(frozen=True)
class RiskPolicy:
    lambda0: float = 1.0
    lambda_g: float = 0.0
    delta: float = 0.0
    source: str = "analytic_fixture_only_not_validation_locked"
    ambiguity_calibrated: bool = False

    def validate(self):
        if self.lambda0 <= 0 or self.lambda_g < 0 or self.delta < 0 or not self.source:
            raise ValueError("positive fixed information scale and nonnegative validated weights required")
        if not all(torch.isfinite(torch.tensor(v)) for v in (self.lambda0, self.lambda_g, self.delta)):
            raise ValueError("finite risk policy required")
        return self


@dataclass
class DecisionState:
    identity: str
    sequence: str
    timestamp: float
    rotations: torch.Tensor
    translation: torch.Tensor
    beta: torch.Tensor
    candidate_id: str
    reasons: list = field(default_factory=list)

    def to_dict(self):
        return {"identity": self.identity, "sequence": self.sequence, "timestamp": self.timestamp,
                "rotations": self.rotations.cpu().tolist(), "translation": self.translation.cpu().tolist(),
                "beta": self.beta.cpu().tolist(), "candidate_id": self.candidate_id, "reasons": self.reasons}

    @classmethod
    def from_dict(cls, value, *, device="cpu", dtype=torch.float32):
        return cls(value["identity"], value["sequence"], value["timestamp"],
                   *(torch.tensor(value[key], device=device, dtype=dtype) for key in
                     ("rotations", "translation", "beta")), value["candidate_id"], value["reasons"])


def information_metric(information, policy):
    policy.validate()
    matrix = information.matrix
    if matrix is None or not torch.isfinite(matrix).all():
        raise ValueError("invalid_or_unavailable_information")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not torch.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError("information_not_symmetric_square")
    eigenvalues, vectors = torch.linalg.eigh((matrix+matrix.T)/2)
    tolerance = 1e-7*max(1.0, float(eigenvalues.abs().max()))
    if eigenvalues.min() < -tolerance:
        raise ValueError("information_not_positive_semidefinite")
    # Clip roundoff-only negative eigenvalues; no prior inserted into Lambda.
    weights = policy.lambda0 / (eigenvalues.clamp_min(0) + policy.lambda0)
    return (vectors*weights[None]) @ vectors.T


def decide_information(store, information, scale, policy, *, method="A6", observation,
                       cameras=None, noise=None, surface=None, previous=None):
    if method not in ("A6", "A7"):
        raise ValueError("information method must be A6 or independently enabled A7")
    policy.validate()
    observation.validate()
    if previous is not None and (previous.identity, previous.sequence) != (observation.identity, observation.sequence):
        raise ValueError("incompatible decision history; explicit reset required")
    legal, actions = torch.where(store.numerical)[0], torch.where(store.actions)[0]
    if not observation.admitted().any() or len(legal) == 0 or len(actions) == 0:
        return Decision(method, None, None, store.raw.new_full((len(store.raw),), torch.inf),
                        True, None, ["empty_action_set_or_no_current_observation"]), previous
    costs = store.raw.new_full((len(store.raw),), torch.inf)
    reasons = []
    try:
        # Unbound information is only allowed in the explicitly named analytic fixture.
        analytic_unbound = (information.noise_source == "analytic_fixture_only"
                            and observation.source == "analytic_fixture" and not information.binding)
        if not analytic_unbound:
            expected = {"identity": observation.identity, "sequence": observation.sequence,
                        "frame": observation.frame, "timestamp": observation.timestamp,
                        "camera_ids": list(observation.camera_ids), "operator": observation.operator,
                        "source": observation.source}
            if information.provenance != expected:
                raise ValueError("information_current_observation_provenance_mismatch")
            require_information_binding(information, observation, cameras, noise)
        if information.scale.shape != scale.values.shape or not torch.allclose(information.scale.to(scale.values), scale.values):
            raise ValueError("information_and_decision_coordinate_scale_mismatch")
        B = information_metric(information, policy).to(store.raw)
        if B.shape != (store.topology.tangent_dimensions,)*2 or information.anchor_rotation is None:
            raise ValueError("information_anchor_or_topology_invalid")
        validate_rotation_matrices(information.anchor_rotation)
        reduced = store.permuted(legal)
        delta = pairwise_tangent(reduced, information.anchor_rotation, scale)
        risk = torch.einsum("kli,ij,klj->kl", delta, B, delta).mean(1)
        costs[legal] = risk
        if policy.lambda_g > 0:
            if any(value is None for value in (cameras, noise, surface)):
                raise ValueError("geometry_resources_unavailable")
            points = store.raw.new_full((len(store.raw), 17, 3), torch.nan)
            points[legal] = surface(store.rotations[legal], store.translation[legal], store.beta[legal])["coco17"]
            errors = observation_errors(points, observation, cameras, noise)[0]
            costs[legal] += policy.lambda_g*errors[legal]
    except (ValueError, RuntimeError) as error:
        fallback = decide(store, "A4", scale)
        fallback.method = method
        fallback.reasons.extend([f"fallback_isotropic_medoid:{error}", *information.reasons])
        fallback.diagnostic["fallback"] = True
        fallback.ambiguity = True
        fallback.diagnostic["metric_not_posterior_covariance"] = True
        if fallback.selected_index is None:
            return fallback, previous
        # Unsafe charts disable the information/time mechanism together.
        chosen = fallback.human(store)
        return fallback, DecisionState(observation.identity, observation.sequence, observation.timestamp,
                                        chosen["rotations"].detach(), chosen["translation"].detach(),
                                        chosen["beta"].detach(), chosen["id"], fallback.reasons)
    selected = stable_min(costs, actions, store.ids)
    if selected is None:
        return Decision(method, None, None, costs, True, None, ["all_action_costs_nonfinite"]), previous
    minimum = costs[selected]
    ties = actions[torch.isfinite(costs[actions]) & (costs[actions] <= minimum+policy.delta)]
    if method == "A7" and previous is not None:
        dt = observation.timestamp - previous.timestamp
        if dt <= 0:
            raise ValueError("decision timestamps must strictly increase")
        if dt > .5:
            reasons.append("time_preference_disabled_gap_over_0.5s")
        elif len(ties) > 1:
            try:
                validate_rotation_matrices(previous.rotations)
                if not torch.isfinite(previous.translation).all() or not torch.isfinite(previous.beta).all():
                    raise ValueError("nonfinite_previous_human")
                n = store.topology.nodes
                relative = previous.rotations.to(store.raw).transpose(-1, -2) @ store.rotations[ties]
                rotation_cost = (angles(relative).square() / scale.values[:3*n:3].to(store.raw).square()).sum(-1)
                vector = torch.cat((store.translation[ties]-previous.translation.to(store.raw),
                                    store.beta[ties]-previous.beta.to(store.raw)), -1)
                physical_distance = (rotation_cost+(vector/scale.values[3*n:].to(store.raw)).square().sum(-1))/dt**2
                temporal = store.raw.new_full((len(store.raw),), torch.inf)
                temporal[ties] = physical_distance
                temporal_selected = stable_min(temporal, ties, store.ids)
                if temporal_selected is None:
                    raise ValueError("nonfinite_temporal_costs")
                selected = temporal_selected
                reasons.append("time_selected_only_within_current_near_tie")
            except (RuntimeError, ValueError) as error:
                reasons.append(f"time_preference_disabled:{error}")
    ranked = torch.sort(costs[actions][torch.isfinite(costs[actions])]).values
    gap = float(ranked[1]-ranked[0]) if len(ranked) > 1 else None
    decision = Decision(method, store.ids[selected], selected, costs,
                        not policy.ambiguity_calibrated or len(ties) > 1, gap, reasons,
                        {"near_tie_candidate_ids": [store.ids[int(i)] for i in ties],
                         "metric_not_posterior_covariance": True, "fallback": False,
                         "policy_source": policy.source, "unassessed_multimodality_retained": True})
    chosen = decision.human(store)
    return decision, DecisionState(observation.identity, observation.sequence, observation.timestamp,
                                    chosen["rotations"].detach(), chosen["translation"].detach(),
                                    chosen["beta"].detach(), chosen["id"], reasons)
