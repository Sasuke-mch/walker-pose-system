import numpy as np
import pytest
import torch
from pose_app.constrained_grasp import project_halfspaces, frame_local_jacobian, constrained_adam_step, probe_local_goal_descent


def test_projection_moves_along_wall_and_satisfies_intersection():
    value, info = project_halfspaces([1.,-1.], [[1.,0.],[-1.,0.],[0.,-1.]], [0.,0.,2.])
    assert info['passed'] and np.allclose(value,[0.,-1.],atol=1e-7)


def test_inconsistent_constant_constraint_is_not_called_feasible():
    _, info = project_halfspaces([1.], [[0.]], [-1.])
    assert not info['passed']


def test_scaled_jacobian_matches_local_finite_difference():
    p = torch.nn.Parameter(torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64))
    c = torch.stack((p[:,0]**2,p[:,1]+p[:,0]),dim=1)
    jac = frame_local_jacobian(c,[p],[.05])
    assert np.allclose(jac, [[[.1,0.],[.05,.05]],[[.3,0.],[.05,.05]]])


def test_blocked_coordinate_does_not_block_other_frame_or_coordinate():
    p = torch.nn.Parameter(torch.zeros(2,2))
    opt = torch.optim.Adam([p],lr=.1)
    loss = ((p-torch.tensor([[1.,-1.],[1.,-1.]]))**2).sum()
    loss.backward()
    def constraints():
        return torch.stack((p[:,0]*torch.tensor([1.,0.]),),dim=1)
    def evaluate():
        value = ((p-torch.tensor([[1.,-1.],[1.,-1.]]))**2).sum()
        return value, bool(p[0,0]<=1e-6), {}
    result = constrained_adam_step(opt,[p],evaluate,loss.detach(),constraints,[1.])
    assert result['accepted'] and abs(p[0,0])<1e-6 and p[0,1]<0 and p[1,0]>0


@pytest.mark.parametrize('exception',[False,True])
def test_full_state_rollback_for_exact_rejection_or_exception(exception):
    p = torch.nn.Parameter(torch.ones(1,2))
    opt = torch.optim.Adam([p],lr=.1)
    p.square().sum().backward(); opt.step(); opt.zero_grad()
    old = p.detach().clone(); moments = {k:v.clone() for k,v in opt.state[p].items()}
    loss = p.square().sum(); loss.backward()
    def evaluate():
        if exception:
            raise RuntimeError('injected')
        return p.square().sum(), False, {}
    if exception:
        with pytest.raises(RuntimeError,match='injected'):
            constrained_adam_step(opt,[p],evaluate,loss.detach(),lambda:p[:,:1]*0-1,[1.])
    else:
        assert not constrained_adam_step(opt,[p],evaluate,loss.detach(),lambda:p[:,:1]*0-1,[1.])['accepted']
    assert torch.equal(p,old) and all(torch.equal(opt.state[p][k],v) for k,v in moments.items())


def test_linear_feasible_candidate_cannot_bypass_nonlinear_guard():
    p = torch.nn.Parameter(torch.tensor([[0.,0.]]))
    opt = torch.optim.Adam([p],lr=.1)
    loss = ((p-1)**2).sum(); loss.backward()
    result = constrained_adam_step(opt,[p],lambda:(((p-1)**2).sum(),bool(p.square().sum()<=.001),{}),
                                  loss.detach(),lambda:p.square().sum(1,keepdim=True)-.001,[1.])
    assert result['accepted'] and result['alpha']<1 and p.square().sum()<=.001


def test_zero_actual_motion_is_rejected_and_moments_restored():
    p = torch.nn.Parameter(torch.zeros(1,2))
    opt = torch.optim.Adam([p],lr=.1)
    loss = p.square().sum(); loss.backward()
    result = constrained_adam_step(opt,[p],lambda:(p.square().sum(),True,{}),loss.detach(),
                                  lambda:p[:,:1]*0-1,[1.])
    assert not result['accepted'] and not opt.state


def test_normalized_radius_limits_translation_in_meters():
    p = torch.nn.Parameter(torch.zeros(1,2))
    opt = torch.optim.Adam([p],lr=1.)
    loss = ((p-1)**2).sum(); loss.backward()
    result = constrained_adam_step(opt,[p],lambda:(((p-1)**2).sum(),True,{}),loss.detach(),
                                  lambda:p[:,:1]*0-1,[.05],trust_radius=.2)
    assert result['accepted'] and torch.linalg.vector_norm(p)<=.010001


@pytest.mark.parametrize('corrections,accepted',[(0,False),(4,True)])
def test_curved_boundary_recovery_finds_exact_feasible_descent(corrections,accepted):
    p = torch.nn.Parameter(torch.tensor([[1.,0.]],dtype=torch.float64))
    opt = torch.optim.Adam([p],lr=.1)
    loss = (p[:,1]-1).square().sum(); loss.backward()
    constraint = lambda:p.square().sum(dim=1,keepdim=True)-1
    result = constrained_adam_step(opt,[p],lambda:((p[:,1]-1).square().sum(),bool(constraint().max()<=0),{}),
                                  loss.detach(),constraint,[1.],correction_steps=corrections)
    assert result['accepted'] is accepted
    if accepted:
        assert p[0,1]>.09 and constraint().max()<=0 and (p[:,1]-1).square().sum()<loss
        assert torch.linalg.vector_norm(p-torch.tensor([[1.,0.]]))<=.2
    else:
        assert torch.equal(p,torch.tensor([[1.,0.]],dtype=torch.float64)) and not opt.state


def test_recovery_cannot_bypass_other_exact_guard():
    p = torch.nn.Parameter(torch.tensor([[1.,0.]],dtype=torch.float64))
    opt = torch.optim.Adam([p],lr=.1)
    loss = (p[:,1]-1).square().sum(); loss.backward()
    result = constrained_adam_step(opt,[p],lambda:((p[:,1]-1).square().sum(),False,{}),loss.detach(),
                                  lambda:p.square().sum(1,keepdim=True)-1,[1.],correction_steps=4)
    assert not result['accepted'] and not opt.state and p[0,0]==1 and p[0,1]==0


def test_probe_is_read_only_and_reports_local_witness():
    p = torch.nn.Parameter(torch.zeros(2,2,dtype=torch.float64))
    old=p.detach().clone()
    p.grad=torch.ones_like(p)
    report=probe_local_goal_descent([p],[1.],lambda:p[:,0:1]-1,lambda:(p-1).square())
    assert report['status']=='local_finite_witness' and report['global_feasibility']=='unknown'
    assert torch.equal(p,old) and torch.equal(p.grad,torch.ones_like(p))
    assert all(all(v<0 for v in f['linear_wrist_squared_distance_derivatives']) for f in report['frames'])


def test_probe_tangent_failure_does_not_claim_global_infeasibility():
    p = torch.nn.Parameter(torch.tensor([[1.,0.]],dtype=torch.float64))
    report=probe_local_goal_descent([p],[1.],lambda:p.square().sum(1,keepdim=True)-1,
                                  lambda:(p[:,1:2]-1).square())
    assert report['status']=='no_witness_on_this_ray' and report['global_feasibility']=='unknown'
    assert p[0,0]==1 and p[0,1]==0


def test_probe_exception_restores_parameters():
    p = torch.nn.Parameter(torch.zeros(1,1,dtype=torch.float64))
    def goals():
        if p[0,0]>0:
            raise RuntimeError('probe')
        return (p-1).square()
    with pytest.raises(RuntimeError,match='probe'):
        probe_local_goal_descent([p],[1.],lambda:p-1,goals)
    assert p[0,0]==0


def test_recovery_exception_restores_existing_adam_moments():
    p = torch.nn.Parameter(torch.tensor([[1.,0.]],dtype=torch.float64))
    opt = torch.optim.Adam([p],lr=.1)
    (p[:,1]-1).square().sum().backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        p.copy_(torch.tensor([[1.,0.]],dtype=torch.float64))
    old = p.detach().clone(); moments = {k:v.clone() for k,v in opt.state[p].items()}
    loss = (p[:,1]-1).square().sum(); loss.backward()
    calls=0
    def evaluate():
        nonlocal calls
        calls+=1
        if calls>1:
            raise RuntimeError('after recovery')
        return (p[:,1]-1).square().sum(),False,{}
    with pytest.raises(RuntimeError,match='after recovery'):
        constrained_adam_step(opt,[p],evaluate,loss.detach(),lambda:p.square().sum(1,keepdim=True)-1,
                              [1.],correction_steps=4)
    assert torch.equal(p,old) and all(torch.equal(opt.state[p][k],v) for k,v in moments.items())


def test_recovery_radius_includes_correction_not_only_tangent_proposal():
    p = torch.nn.Parameter(torch.zeros(1,1,dtype=torch.float64))
    old = p.detach().clone()
    opt = torch.optim.Adam([p],lr=.1)
    loss = (p-1).square().sum(); loss.backward()
    # At y=.1 this nonconvex constraint has a negative derivative, so local
    # recovery points farther out, beyond the original normalized radius.
    constraint = lambda:torch.sin(20*p).square()
    result = constrained_adam_step(opt,[p],lambda:((p-1).square().sum(),bool(constraint().max()<=0),{}),
                                  loss.detach(),constraint,[1.],correction_steps=4,trust_radius=.1)
    assert result['attempts'][0]['corrections'][-1]['status']=='recovery_exceeds_radius'
    assert torch.linalg.vector_norm(p-old)<=.1


@pytest.mark.parametrize('identities,frames,passed',[
    ([(7,'left'),(7,'right')],[7],True),
    ([(7,'left')],[7],False),
    ([(7,'left'),(7,'left')],[7],False),
    ([(7,'left'),(7,'right'),(8,'left')],[7],False),
    ([(7,'left'),(7,'right')],[7,7],False),
    ([],[],False),
    ([(7,'unknown')],[7],False),
])
def test_hand_audit_requires_exact_all_frame_all_hand_coverage(identities,frames,passed,monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools'))
    from benchmark_constrained_grasp import hand_audit_coverage
    records=[dict(pair_id=p,hand=s,passed_geometry_proxy=True) for p,s in identities]
    assert hand_audit_coverage(records,frames)['passed'] is passed


def test_missing_hand_cannot_pass_main_hand_gate(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools'))
    from benchmark_constrained_grasp import assess
    final=dict(wrist_error_mm={s:dict(p95=0.) for s in ('left','right')},
               sole_minimum_z_mm={s:0. for s in ('left','right')},terms=dict(leg_ground_acceleration=0.))
    (tmp_path/'fit_summary.json').write_text(json.dumps(dict(checkpoints=dict(initial=final,final=final))))
    (tmp_path/'update_transactions.json').write_text(json.dumps(dict(updates=[])))
    (tmp_path/'hand_geometry_audit.json').write_text(json.dumps(dict(records=[dict(pair_id=7,hand='left',passed_geometry_proxy=True)])))
    np.savez(tmp_path/'result.npz',pair_id=[7],predicted_coco=np.zeros((1,17,3)),raw_triangulated_points=np.zeros((1,17,3)))
    np.savez(tmp_path/'observation_guard_reference.npz',baseline_error_m=np.zeros((1,17)),valid=np.ones((1,17),dtype=bool))
    result=assess(tmp_path)
    assert not result['gates']['hands'] and not result['passed']
    assert result['hand_coverage']['missing']==[(7,'right')]


def test_recovery_counts_distinguish_attempts_from_true_forward_evaluations(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'tools'))
    from audit_constrained_grasp_comparison import recovery_counts
    result=recovery_counts([dict(attempts=[dict(corrections=[dict(status=s) for s in
        ('evaluated','linear_recovery_not_certified','recovery_exceeds_radius','evaluated')])])])
    assert result['recovery_attempts']==4 and result['recovery_evaluations']==2
    assert result['recovery_not_certified']==1 and result['recovery_exceeds_radius']==1
