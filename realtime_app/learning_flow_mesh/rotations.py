"""Column-based 6D rotations with explicit degeneracy reporting."""
import torch
import torch.nn.functional as F


def matrix_to_six(matrix):
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def six_to_matrix(six, eps=1e-6):
    """Return matrices and validity; invalid matrices are NaN, never identity."""
    if six.shape[-1] != 6:
        raise ValueError("rotation representation must have six coordinates")
    finite = torch.isfinite(six).all(dim=-1)
    clean = torch.where(finite[..., None], six, torch.zeros_like(six))
    a, b = clean[..., :3], clean[..., 3:]
    first = F.normalize(a, dim=-1, eps=eps)
    orthogonal = b - (first * b).sum(-1, keepdim=True) * first
    second = F.normalize(orthogonal, dim=-1, eps=eps)
    valid = finite & (a.norm(dim=-1) > eps) & (orthogonal.norm(dim=-1) > eps)
    matrix = torch.stack((first, second, torch.cross(first, second, dim=-1)), dim=-1)
    matrix = torch.where(valid[..., None, None], matrix, torch.full_like(matrix, float("nan")))
    return matrix, valid


def validate_rotation_matrices(matrix, atol=1e-4):
    if matrix.shape[-2:] != (3, 3) or not torch.isfinite(matrix).all():
        raise ValueError("rotations must be finite 3x3 matrices")
    identity = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    if (matrix.transpose(-1, -2) @ matrix - identity).abs().max() > atol:
        raise ValueError("rotation is not orthonormal")
    if (torch.linalg.det(matrix) - 1).abs().max() > atol:
        raise ValueError("rotation determinant must equal +1")
