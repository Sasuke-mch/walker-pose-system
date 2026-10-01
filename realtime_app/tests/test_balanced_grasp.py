import numpy as np
import torch
from pose_app.balanced_grasp import (grouped_observation_loss, observation_guard,
    build_stage2_anchors, stage2_terms, leg_temporal_terms, guarded_adam_step)


def test_group_normalization_and_guard_cannot_hide_bad_knee():
    p = torch.zeros(3,17,3, requires_grad=True)
    target = torch.zeros_like(p)
    q = torch.ones(3,17); valid = q.bool()
    modified = p + torch.nn.functional.one_hot(torch.tensor(13),17)[None,:,None]*.02
    loss, groups = grouped_observation_loss(modified,target,q,valid)
    assert groups['left_leg'] > 0 and groups['right_leg'] == 0
    loss.backward(); assert p.grad[:,13].abs().sum() > 0
    ok, excess = observation_guard(modified,target,valid,torch.zeros(3,17))
    assert not ok and excess > .01
    valid[:,13] = False
    assert observation_guard(modified,target,valid,torch.zeros(3,17))[0]


def test_stage2_fixed_patches_full_segments_rigid_lift_and_no_bridge():
    v = np.zeros((9,12,3)); v[:,:,2] = -.1
    stages = np.array(['stage1']+['stage2_feet_static_walker_moving']*3+['stage1']+['stage2_feet_static_walker_moving']*3+['stage1'])
    sets = [dict(heel=list(range(3)),ball=list(range(3,6))),dict(heel=list(range(6,9)),ball=list(range(9,12)))]
    ids, targets, segments, records = build_stage2_anchors(v,stages,np.ones(9,bool),sets)
    assert records[0]['rigid_lift_m'] == [.1,.1]
    assert np.allclose(targets[1:4,:,2],0)
    assert (segments == [ -1,0,0,0,-1,1,1,1,-1]).all()
    points = torch.tensor(v,dtype=torch.float32,requires_grad=True)
    points_offset = points + torch.tensor([[[0.,0.,.1]]])
    # Each segment is static, at different XY; no velocity across boundaries.
    offset = torch.zeros_like(points); offset[5:8,:,0] = 2.
    position, velocity = stage2_terms(points_offset+offset,torch.tensor(ids),torch.tensor(targets),torch.tensor(segments))
    assert position > 0 and velocity == 0
    position.backward(); assert points.grad[0].abs().sum() == 0 and points.grad[1].abs().sum() < 1e-10
    assert points.grad[5].abs().sum() > 0


def test_unavailable_short_segments_remain_unavailable():
    v = np.zeros((3,12,3)); sets = [dict(heel=[0,1,2],ball=[3,4,5])]*2
    _,_,segments,records = build_stage2_anchors(v,['stage2_feet_static_walker_moving']*3,[True,False,True],sets)
    assert (segments == -1).all() and not records[0]['available']


def test_temporal_only_leg_rotations_and_invalid_frames():
    body = torch.eye(3).repeat(4,21,1,1).requires_grad_()
    perturb = torch.zeros_like(body); perturb[1,0,0,0] = .1
    coco = torch.randn(4,17,3,requires_grad=True)
    a,b = leg_temporal_terms(body+perturb,coco,torch.ones(4,dtype=torch.bool))
    (a+b).backward()
    assert body.grad[:,15].abs().sum() == 0 and body.grad[:,0].abs().sum() > 0
    assert coco.grad[:,:11].abs().sum() == 0
    a,b = leg_temporal_terms(body,coco,torch.zeros(4,dtype=torch.bool))
    assert a == 0 and b == 0


def test_rejected_adam_restores_moments_and_parameter():
    p = torch.nn.Parameter(torch.tensor([1.]))
    opt = torch.optim.Adam([p],lr=.1)
    loss = p.square().sum(); loss.backward()
    result = guarded_adam_step(opt,[p],lambda:(p.square().sum(),False,{}),loss.detach())
    assert not result['accepted'] and p.item() == 1. and len(opt.state) == 0
    assert len(result['attempts']) == 8


def test_backtracking_accepts_feasible_descent():
    p = torch.nn.Parameter(torch.tensor([1.]))
    opt = torch.optim.Adam([p],lr=1.)
    loss = p.square().sum(); loss.backward()
    result = guarded_adam_step(opt,[p],lambda:(p.square().sum(),bool(p>.7),{}),loss.detach())
    assert result['accepted'] and result['alpha'] == .25 and .7 < p.item() < 1.


def test_rejection_restores_existing_momentum_and_step_count():
    p = torch.nn.Parameter(torch.tensor([1.]))
    opt = torch.optim.Adam([p],lr=.1)
    p.square().sum().backward(); opt.step(); opt.zero_grad()
    value = p.detach().clone()
    previous = {key: item.clone() for key,item in opt.state[p].items()}
    loss = p.square().sum(); loss.backward()
    result = guarded_adam_step(opt,[p],lambda:(p.square().sum(),False,{}),loss.detach())
    assert not result['accepted'] and torch.equal(p,value)
    assert all(torch.equal(opt.state[p][key],item) for key,item in previous.items())
