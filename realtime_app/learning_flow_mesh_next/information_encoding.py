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


def information_features(matrix, nodes, *, mode='diagonal_blocks', pose_probes=None, shape_probes=None):
    dimensions = 3*nodes+13
    if matrix.shape != (dimensions, dimensions) or not torch.isfinite(matrix).all():
        raise ValueError('invalid frozen anchor information')
    if mode not in ('diagonal_blocks', 'cross_block_sketch'):
        raise ValueError('unregistered information encoding')
    info = matrix.detach()
    slices = [slice(3*i, 3*i+3) for i in range(nodes+1)] + [slice(-10, None)]
    diagonal = [signed_log(info[part, part]).flatten() for part in slices]
    if mode == 'diagonal_blocks':
        return torch.stack(diagonal[:-1]), diagonal[-1]
    if pose_probes is None or shape_probes is None:
        raise ValueError('fixed explicitly registered coupling probes required')
    if pose_probes.shape != (dimensions, 3) or shape_probes.shape != (dimensions, 10):
        raise ValueError('probe topology mismatch')
    cross = info.clone()
    for part in slices:
        cross[part, part] = 0
    projected = cross @ pose_probes.to(info)
    shape_projected = cross[-10:] @ shape_probes.to(info)
    # Projection before signed-log keeps the blind subspace explicit. No fitted
    # normalization, train statistics, extra MLP, or tunable mixing coefficient.
    mixed = [old + signed_log(projected[part]).flatten()
             for old, part in zip(diagonal[:-1], slices[:-1])]
    return torch.stack(mixed), diagonal[-1]+signed_log(shape_projected).flatten()
