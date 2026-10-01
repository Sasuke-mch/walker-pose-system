import numpy as np
import pytest
import torch
from pose_app.constrained_grasp import project_halfspaces, frame_local_jacobian, constrained_adam_step


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
