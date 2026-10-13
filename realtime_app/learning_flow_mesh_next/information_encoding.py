"""Same-width information features; optional lossy signed coupling sketches.

Binary probes follow Achlioptas' random-projection construction. Three/ten
probes are NOT a JL guarantee, an injective representation, or a posterior.
"""
import torch


def fixed_probes(dimensions, columns, *, seed=20261012):
    generator = torch.Generator(device='cpu').manual_seed(seed+columns)
    return (torch.randint(2, (dimensions, columns), generator=generator).double()*2-1)/columns**.5


def signed_log(value):
    return value.sign()*torch.log1p(value.abs())


def _diagonal_pose_position_blocks(matrix, nodes):
    # A strided view of every 3x3 block, including translation. No dense mask,
    # device index allocation, or per-joint CUDA launch is needed.
    stop = 3*(nodes+1)
    return (matrix[:stop, :stop].unfold(0, 3, 3).unfold(1, 3, 3)
            .diagonal(dim1=0, dim2=1).movedim(-1, 0))


def information_features(matrix, nodes, *, mode='diagonal_blocks', pose_probes=None, shape_probes=None):
    dimensions = 3*nodes+13
    if matrix.shape != (dimensions, dimensions) or not torch.isfinite(matrix).all():
        raise ValueError('invalid frozen anchor information')
    if mode not in ('diagonal_blocks', 'cross_block_sketch'):
        raise ValueError('unregistered information encoding')
    info = matrix.detach()
    diagonal = signed_log(_diagonal_pose_position_blocks(info, nodes)).flatten(1)
    shape = signed_log(info[-10:, -10:]).flatten()
    if mode == 'diagonal_blocks':
        return diagonal, shape
    if pose_probes is None or shape_probes is None:
        raise ValueError('fixed explicitly registered coupling probes required')
    if pose_probes.shape != (dimensions, 3) or shape_probes.shape != (dimensions, 10):
        raise ValueError('probe topology mismatch')
    cross = info.clone()
    _diagonal_pose_position_blocks(cross, nodes).zero_()
    cross[-10:, -10:] = 0
    projected = cross @ pose_probes.to(info)
    shape_projected = cross[-10:] @ shape_probes.to(info)
    # Projection before signed-log keeps the blind subspace explicit. No fitted
    # normalization, train statistics, extra MLP, or tunable mixing coefficient.
    mixed = diagonal + signed_log(projected[:3*(nodes+1)]).reshape(nodes+1, 9)
    return mixed, shape+signed_log(shape_projected).flatten()
