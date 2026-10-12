"""Opt-in F0 rotation-geometry replacement; no new conditioning or training loop."""
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from realtime_app.learning_flow_mesh.rotations import (
    matrix_to_six,
    validate_rotation_matrices,
)

from .joint_flow import JointFlow
from .tangent import skew


def rotation_exp(vector):
    """Rodrigues with finite small-angle derivatives and no matrix square.

    K^2 = ww^T - |w|^2 I; use the half-angle sinc identity for its coefficient.
    Taylor terms depend on |w|^2 so the derivative at zero needs no norm divide.
    """
    if vector.shape[-1] != 3:
        raise ValueError("rotation exponential requires three angular coordinates")
    square = vector.square().sum(-1)
    angle = square.clamp_min(1e-4).sqrt()
    small = square < 1e-4
    a = torch.where(small, 1-square/6+square.square()/120, torch.sinc(angle/torch.pi))
    b = torch.where(small, .5-square/24+square.square()/720, .5*torch.sinc(angle/(2*torch.pi)).square())
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype)
    outer = vector[..., :, None]*vector[..., None, :]
    return ((1-b*square)[..., None, None]*identity
            + a[..., None, None]*skew(vector)+b[..., None, None]*outer)


def haar_rotations(gaussian):
    """Normalized iid Gaussian4 induces Haar SO3; malformed input is not redrawn."""
    if gaussian.shape[-1] != 4:
        raise ValueError("SO3 base requires four Gaussian coordinates")
    length = gaussian.norm(dim=-1, keepdim=True)
    bad = ~torch.isfinite(gaussian).all(-1) | ~torch.isfinite(length[..., 0]) | (length[..., 0] == 0)
    if bad.any():
        raise ValueError(f"invalid SO3 base at indices {bad.nonzero().tolist()}")
    w, x, y, z = (gaussian/length).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(*gaussian.shape[:-1], 3, 3)


def geodesic_path(source, target, time):
    """Fixed-endpoint target preparation; CPU log has no endpoint-gradient API."""
    if source.shape != target.shape or source.ndim != 4 or source.shape[-2:] != (3, 3):
        raise ValueError("matching batch/node rotation endpoints required")
    if source.requires_grad or target.requires_grad:
        raise ValueError("SO3 training path requires frozen endpoints")
    if time.shape != (len(source),) or not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
        raise ValueError("one shared path time per complete state required")
    validate_rotation_matrices(source)
    validate_rotation_matrices(target)
    relative = source.transpose(-1, -2)@target
    vectors = Rotation.from_matrix(relative.cpu().double().numpy().reshape(-1, 3, 3)).as_rotvec()
    omega = torch.from_numpy(vectors.copy()).to(source).reshape(*source.shape[:-2], 3)
    return source@rotation_exp(time[:, None, None]*omega), omega


class SO3JointFlow(JointFlow):
    """Reuse the existing core; three pose outputs are body angular velocities.

    Internal states use physical rotation columns and standardized T/beta.
    Public outputs use the bound full Normalization so existing CandidateStore
    decoding retains its meaning. Unreachable information layers are omitted
    by default; explicit legacy allocation is a storage-only control.
    """

    rotation_representation = "SO3_body_angular_velocity"
    solver_name = "LieEuler"

    def __init__(self, normalization, config=None, *, rotation_kernel="rodrigues",
                 retain_inactive_information=False):
        if rotation_kernel not in ("rodrigues", "matrix_exp"):
            raise ValueError("unregistered SO3 rotation kernel")
        if not isinstance(retain_inactive_information, bool):
            raise TypeError("explicit boolean legacy information allocation required")
        super().__init__(config)
        if normalization.topology != self.topology:
            raise ValueError("SO3 normalization topology mismatch")
        self.normalization = normalization.validate()
        self.pose_out = nn.Linear(self.config.width, 3)
        self.rotation_kernel = rotation_kernel
        self.retain_inactive_information = retain_inactive_information
        if not retain_inactive_information:
            # Preserve the original construction/RNG order and active tensors.
            # SO3.velocity rejects Lambda before these base-class paths execute.
            for name in ("info_pose", "info_position", "info_shape"):
                delattr(self, name)
            self.pose_probes = self.shape_probes = None

    def _rotation_exp(self, vector):
        return rotation_exp(vector) if self.rotation_kernel == "rodrigues" else torch.matrix_exp(skew(vector))

    def validate_normalization(self, normalization):
        normalization.validate()
        bound = self.normalization
        if (normalization.topology != bound.topology or normalization.source != bound.source
                or normalization.split != bound.split or normalization.source_kind != bound.source_kind
                or not torch.equal(normalization.mean, bound.mean)
                or not torch.equal(normalization.scale, bound.scale)):
            raise ValueError("SO3 output codec normalization mismatch")

    def velocity(self, state, flow_time, context, information=None):
        if information is not None:
            raise ValueError("SO3 F0 does not accept Lambda conditioning")
        return super().velocity(state, flow_time, context)

    def _internal(self, rotations, auxiliary):
        return torch.cat((matrix_to_six(rotations).flatten(1), auxiliary), -1)

    def _export(self, rotations, auxiliary):
        n = self.config.nodes*6
        columns = matrix_to_six(rotations).flatten(1)
        pose = (columns-self.normalization.mean[:n].to(columns))/self.normalization.scale[:n].to(columns)
        return torch.cat((pose, auxiliary), -1)

    def _base(self, initial):
        n = self.config.nodes
        if initial.ndim != 2 or initial.shape[-1] != self.topology.dimensions or not torch.isfinite(initial).all():
            raise ValueError("finite packed Gaussian base required")
        # Reuse existing candidate seeds/noise shape; only four of six iid
        # Gaussian coordinates per node are used by this base measure.
        return haar_rotations(initial[:, :6*n].reshape(-1, n, 6)[..., :4]), initial[:, 6*n:]

    def sample(self, context, initial, *, steps=8, information=None):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("positive Lie Euler step budget required")
        if information is not None:
            raise ValueError("SO3 F0 does not accept Lambda conditioning")
        rotations, auxiliary = self._base(initial)
        context = context.expand(len(initial), -1)
        n = self.config.nodes*3
        for step in range(steps):
            velocity = self.velocity(self._internal(rotations, auxiliary),
                                     initial.new_full((len(initial),), step/steps), context)
            omega = velocity[:, :n].reshape(-1, self.config.nodes, 3)
            rotations = rotations@self._rotation_exp(omega/steps)
            auxiliary = auxiliary+velocity[:, n:]/steps
        return self._export(rotations, auxiliary)

    def regression(self, context, information=None):
        rotations = torch.eye(3, device=context.device, dtype=context.dtype).expand(len(context), self.config.nodes, 3, 3)
        velocity = self.velocity(self._internal(rotations, context.new_zeros(len(context), 13)),
                                 context.new_ones(len(context)), context, information)
        n = self.config.nodes*3
        endpoint = self._rotation_exp(velocity[:, :n].reshape(-1, self.config.nodes, 3))
        return self._export(endpoint, velocity[:, n:])

    def flow_loss(self, context, labels, normalization, *, generator=None, information=None):
        self.validate_normalization(normalization)
        if labels.topology != self.topology:
            raise ValueError("SO3 label topology mismatch")
        if information is not None:
            raise ValueError("SO3 F0 does not accept Lambda conditioning")
        packed, _ = labels.packed(require_complete=True)
        # The Euclidean blocks use the same registered mean/scale as ambient F0.
        n = self.config.nodes
        target = normalization.normalize(packed)[:, 6*n:]
        gaussian = torch.randn(len(packed), self.topology.dimensions, generator=generator,
                               device=packed.device, dtype=packed.dtype)
        base, auxiliary = self._base(gaussian)
        time = torch.rand(len(packed), generator=generator, device=packed.device, dtype=packed.dtype)
        rotations, omega = geodesic_path(base, labels.rotations, time)
        mixed_auxiliary = (1-time[:, None])*auxiliary+time[:, None]*target
        predicted = self.velocity(self._internal(rotations, mixed_auxiliary), time, context)
        error = (predicted-torch.cat((omega.flatten(1), target-auxiliary), -1)).square()
        parts = {"pose": error[:, :3*n].mean(), "translation": error[:, 3*n:3*n+3].mean(),
                 "shape": error[:, -10:].mean()}
        return sum(parts.values())/3, parts
