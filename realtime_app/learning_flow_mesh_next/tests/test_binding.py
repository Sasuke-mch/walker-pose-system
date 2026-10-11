import json
from dataclasses import replace

import pytest
import torch

from ..candidate_store import infer
from ..data_firewall import Normalization
from ..geometry_information import local_information
from ..geometry_observation import NoiseModel
from ..information_risk import RiskPolicy, decide_information
from ..input_binding import observation_snapshot
from ..joint_flow import JointFlow
from ..sensitivity import reweight_information
from ..tangent import TangentScale
from .test_aggregate import rig, store_fixture
from .test_firewall import observation
from .test_information import rigid_surface
from .test_risk import information as unbound_information


def fixture():
    store, obs, cameras, noise = store_fixture(), observation(), rig(), NoiseModel.proxy()
    scale = TangentScale.fixture(store.topology)
    info = local_information(store, obs, cameras, noise, scale, rigid_surface)
    return store, obs, cameras, noise, scale, info


@pytest.mark.parametrize('case', ['mask', 'pixels', 'confidence', 'rays', 'clock', 'noise', 'noise_status', 'K', 'D', 'R', 'T', 'size'])
def test_same_frame_changed_inputs_force_explicit_fallback(case):
    store, obs, cameras, noise, scale, info = fixture()
    cameras = list(cameras)
    if case == 'mask':
        quality = obs.quality.clone()
        quality[0, 0] = False
        obs = replace(obs, quality=quality)
    elif case == 'pixels':
        obs = replace(obs, xy=obs.xy+5)
    elif case == 'confidence':
        obs = replace(obs, confidence=obs.confidence*.5)
    elif case == 'rays':
        obs = replace(obs, origins=obs.origins+.1)
    elif case == 'clock':
        obs = replace(obs, timestamp_kind='host_read_return')
    elif case == 'noise':
        noise = NoiseModel.proxy(5)
    elif case == 'noise_status':
        noise = replace(noise, calibrated=True)
    elif case == 'size':
        cameras[0] = replace(cameras[0], image_size=(2000, 2000))
    else:
        value = getattr(cameras[0], case).clone()
        if case == 'R':
            value = torch.diag(value.new_tensor([-1., -1., 1.]))
        elif case == 'K':
            value[0, 0] *= 2
        else:
            value[0] += .01
        cameras[0] = replace(cameras[0], **{case: value})
    decision, _ = decide_information(store, info, scale, RiskPolicy(lambda_g=0),
                                    observation=obs, cameras=cameras, noise=noise, surface=rigid_surface)
    assert decision.diagnostic['fallback']
    assert 'information_input_binding_mismatch' in decision.reasons[0]


def test_original_clone_reweight_and_tensor_alias_safety():
    store, obs, cameras, noise, scale, info = fixture()
    original = json.dumps(info.binding, allow_nan=False)
    double_obs = replace(obs, xy=obs.xy.double(), confidence=obs.confidence.double(),
                         origins=obs.origins.double(), directions=obs.directions.double())
    decision, _ = decide_information(store, info, scale, RiskPolicy(), observation=double_obs,
                                    cameras=cameras, noise=noise, surface=rigid_surface)
    assert not decision.diagnostic['fallback']
    new_noise = NoiseModel.proxy(5)
    rebound = reweight_information(info, double_obs, new_noise)
    decision, _ = decide_information(store, rebound, scale, RiskPolicy(), observation=double_obs,
                                    cameras=cameras, noise=new_noise, surface=rigid_surface)
    assert not decision.diagnostic['fallback']
    assert rebound.binding['noise'] != info.binding['noise']
    assert json.dumps(info.binding, allow_nan=False) == original
    rebound.binding['observation']['xy'][0][0][0] += 100
    assert json.dumps(info.binding, allow_nan=False) == original
    with pytest.raises(ValueError, match='exact observation binding'):
        reweight_information(info, replace(obs, xy=obs.xy+1), new_noise)
    obs.xy[0, 0, 0] += 10
    assert json.dumps(info.binding, allow_nan=False) == original
    anchor = info.anchor_rotation.clone()
    store.rotations[store.ids.index(info.anchor_id)] *= -1
    assert torch.equal(info.anchor_rotation, anchor)


def test_new_cache_condition_binding_and_legacy_warning():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = JointFlow().eval()
    obs = observation()
    with torch.no_grad():
        store, _ = infer(model, Normalization.fixture(model.topology), obs, k=2, steps=1)
    assert store.provenance['observation_binding'] == observation_snapshot(obs)
    # Random untrained candidates can be behind the camera; isolate binding
    # from projection admissibility with the known geometric fixture.
    store = store_fixture()
    store.provenance['observation_binding'] = observation_snapshot(obs)
    good = local_information(store, obs, rig(), NoiseModel.proxy(), TangentScale.fixture(store.topology), rigid_surface)
    assert good.matrix is not None and good.diagnostics['candidate_current_observation_binding'] == 'verified'
    changed = replace(obs, xy=obs.xy+1)
    info = local_information(store, changed, rig(), NoiseModel.proxy(), TangentScale.fixture(store.topology), rigid_surface)
    assert info.matrix is None and info.reasons == ['candidate_anchor_observation_binding_mismatch']
    assert fixture()[-1].diagnostics['candidate_current_observation_binding'] == 'legacy_unverified'


def test_named_analytic_bypass_does_not_apply_to_real_source():
    store = store_fixture()
    obs = replace(observation(), source='recorded_pixels')
    info = unbound_information(store)
    info.provenance = {'identity': obs.identity, 'sequence': obs.sequence, 'frame': obs.frame,
                       'timestamp': obs.timestamp, 'camera_ids': list(obs.camera_ids),
                       'operator': obs.operator, 'source': obs.source}
    decision, _ = decide_information(store, info, TangentScale.fixture(store.topology),
                                    RiskPolicy(), observation=obs)
    assert decision.diagnostic['fallback']
    assert 'information_input_binding_missing' in decision.reasons[0]


def test_ignored_nan_slots_and_missing_verification_resources():
    store, obs, _cameras, _noise, scale, info = fixture()
    present = obs.present.clone()
    present[0, 0] = False
    xy = obs.xy.clone()
    xy[0, 0] = torch.nan
    masked = replace(obs, present=present, xy=xy)
    snapshot = observation_snapshot(masked)
    assert snapshot['xy'][0][0] == [None, None]
    assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot
    decision, _ = decide_information(store, info, scale, RiskPolicy(lambda_g=0), observation=obs)
    assert decision.diagnostic['fallback']
    assert 'information_verification_resources_unavailable' in decision.reasons[0]


def test_reweight_rejects_relocated_mask_with_equal_point_count():
    store, obs, cameras, noise, scale, _ = fixture()
    quality = obs.quality.clone()
    quality[0, 0] = False
    first = replace(obs, quality=quality)
    info = local_information(store, first, cameras, noise, scale, rigid_surface)
    other = obs.quality.clone()
    other[0, 1] = False
    second = replace(obs, quality=other)
    assert first.admitted().sum() == second.admitted().sum()
    with pytest.raises(ValueError, match='exact observation binding'):
        reweight_information(info, second, NoiseModel.proxy(5))
