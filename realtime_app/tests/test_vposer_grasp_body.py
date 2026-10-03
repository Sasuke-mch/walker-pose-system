"""Body prior must constrain wrists too, while retaining latent gradients."""
from pathlib import Path
import torch
import pytest
from pose_app.vposer_grasp_body import decode_body_rotations, rotation_anchor_loss, rotation_temporal_loss
from pose_app.smplx_fitting import load_vposer_explicit


def test_real_frozen_vposer_decodes_all21_with_live_gradients():
    root = Path(__file__).resolve().parents[2]
    folder = root/'models/VPoser02_05/V02_05'
    if not list(folder.glob('*.yaml')) or not list((folder/'snapshots').glob('*.ckpt')):
        pytest.skip('licensed VPoser checkpoint not provisioned; run the asset-enabled lane')
    vp, _, _ = load_vposer_explicit(root/'models/VPoser02_05/V02_05', 'cpu')
    z = torch.zeros(3,32,requires_grad=True)
    pose = decode_body_rotations(vp,z)
    assert pose.shape == (3,21,3,3)
    assert torch.allclose(pose.transpose(-1,-2)@pose,torch.eye(3),atol=1e-5)
    target = pose.detach().clone()
    target[:,19,0,1] += .1
    rotation_anchor_loss(pose,target).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in vp.parameters())
    with pytest.raises(ValueError):
        decode_body_rotations(vp,torch.zeros(3,63))
    vp.train()
    with pytest.raises(ValueError):
        decode_body_rotations(vp,z)


def test_rotation_temporal_invalid_pairs_do_not_contribute():
    r=torch.eye(3).expand(3,21,3,3).clone().requires_grad_()
    r.data[1,0,0,1]=.3
    zero=rotation_temporal_loss(r,torch.tensor([True,False,True]))
    zero.backward()
    assert zero == 0 and r.grad.abs().sum()==0
    assert rotation_temporal_loss(r,torch.ones(3,dtype=torch.bool))>0
