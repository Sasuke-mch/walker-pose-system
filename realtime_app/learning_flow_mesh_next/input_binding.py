"""Explicit detached numerical snapshots, no opaque identifiers or input aliases."""
import math

import torch

from .geometry_observation import check_rig
from .schema import Observation


def values(tensor, valid=None):
    value = tensor.detach().cpu()
    if valid is not None:
        mask = valid.detach().cpu()
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        value = torch.where(mask, value.double(), torch.nan)
    def finite(item):
        if isinstance(item, list):
            return [finite(v) for v in item]
        return None if isinstance(item, float) and not math.isfinite(item) else item
    return finite(value.tolist())


def observation_snapshot(obs):
    if not isinstance(obs, Observation):
        raise TypeError('typed observation required for information binding')
    obs.validate()
    admitted = obs.admitted()
    geometry = admitted & obs.geometry
    return {'identity': obs.identity, 'sequence': obs.sequence, 'frame': obs.frame,
            'timestamp': obs.timestamp, 'source': obs.source, 'camera_ids': list(obs.camera_ids),
            'pixel_domain': obs.pixel_domain, 'units': obs.units, 'operator': obs.operator,
            'timestamp_kind': obs.timestamp_kind,
            'view_timestamps': list(obs.view_timestamps) if obs.view_timestamps is not None else None,
            'masks': {key: values(getattr(obs, key)) for key in ('present', 'quality', 'geometry', 'views')},
            'xy': values(obs.xy, admitted), 'confidence': values(obs.confidence, admitted),
            'origins': values(obs.origins, geometry), 'directions': values(obs.directions, geometry),
            'rejection_reasons': [list(row) for row in obs.rejection_reasons]}


def rig_snapshot(cameras):
    result = []
    for camera in cameras:
        camera.validate()
        result.append({'instance_id': camera.instance_id, 'role': camera.role,
                       'image_size': list(camera.image_size),
                       **{key: values(getattr(camera, key)) for key in ('K', 'D', 'R', 'T')}})
    return result


def noise_snapshot(noise):
    noise.validate()
    return {'precision': values(noise.precision), 'source': noise.source, 'calibrated': noise.calibrated,
            'camera_ids': list(noise.camera_ids) if noise.camera_ids is not None else None}


def information_snapshot(obs, cameras, noise):
    check_rig(obs, cameras)
    noise.validate_for(obs)
    return {'observation': observation_snapshot(obs), 'rig': rig_snapshot(cameras), 'noise': noise_snapshot(noise)}


def require_information_binding(information, obs, cameras, noise):
    if not information.binding:
        raise ValueError('information_input_binding_missing')
    if cameras is None or noise is None:
        raise ValueError('information_verification_resources_unavailable')
    current = information_snapshot(obs, cameras, noise)
    for key in ('observation', 'rig', 'noise'):
        if information.binding.get(key) != current[key]:
            raise ValueError(f'information_input_binding_mismatch:{key}')
