"""Coordinate, occlusion and annotation contract checks for local grip fitting."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
import pytest

tools = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0, str(tools))
spec = importlib.util.spec_from_file_location('grip_fit',tools/'fit_annotated_shared_grip.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_upright_pixel_corner_mapping():
    uv = torch.tensor([[0.,0.],[1919.,1079.],[100.,200.]])
    assert torch.equal(mod.upright_projection(uv,'left'),torch.tensor([[0.,1919.],[1079.,0.],[200.,1819.]]))
    assert torch.equal(mod.upright_projection(uv,'right'),torch.tensor([[1079.,0.],[0.,1919.],[879.,100.]]))


def test_ray_cylinder_front_back_and_off_axis():
    a,b = torch.tensor([0.,-.5,1.]),torch.tensor([0.,.5,1.])
    points = torch.tensor([[0.,0.,2.],[0.,0.,.5],[.2,0.,2.],[0.,2.,2.]])
    assert mod.handle_occluded(points,a,b,.05).tolist()==[True,False,False,False]


def test_capsule_gap_gradient():
    p=torch.tensor([[.02,0.,0.]],requires_grad=True)
    g=mod.capsule_gap(p,torch.tensor([0.,-.1,0.]),torch.tensor([0.,.1,0.]))
    assert torch.allclose(g,torch.tensor([.004]),atol=1e-6)
    g.sum().backward()
    assert torch.isfinite(p.grad).all() and p.grad[0,0]>0


def test_explicit_remap_preserves_original_and_unions_components(tmp_path):
    data={'imageWidth':1080,'imageHeight':1920,'shapes':[
        {'label':'right_hand','shape_type':'polygon','points':[[0,10],[10,10],[10,20]]},
        {'label':'right_hand','shape_type':'polygon','points':[[900,10],[910,10],[910,20]]}]}
    path=tmp_path/'label.json'
    path.write_text(json.dumps(data))
    before=path.read_bytes()
    masks,audit=mod.load_masks(path,True)
    assert masks['left'].any() and masks['right'].any()
    assert audit[1]['original_label']=='right_hand' and audit[1]['used_label']=='left_hand'
    assert path.read_bytes()==before


def test_wrist_bound_is_euclidean_and_gradient_remains_active():
    center=torch.tensor([.2,-.1,.9])
    translation=torch.nn.Parameter(center+torch.tensor([.01,.01,.01]))
    mod.project_wrist_ball_(translation,center,.01)
    assert float(torch.linalg.vector_norm(translation.detach()-center))<=.0100001
    translation.square().sum().backward()
    assert translation.grad is not None and torch.isfinite(translation.grad).all()


def test_wrist_bound_preserves_inside_point():
    center=torch.zeros(3)
    point=torch.nn.Parameter(torch.tensor([.002,0.,0.]))
    before=point.detach().clone()
    mod.project_wrist_ball_(point,center,.01)
    assert torch.equal(point,before)


def test_diagnostic_reference_requires_bounded_mode():
    data={'schema':'annotated_wrist_targets_v1','status':'incomplete_or_rejected',
          'hands':{s:{'diagnostic_wrist_walker_m':[.1,.2,.9]} for s in ('left','right')}}
    with pytest.raises(ValueError,match='wrist_targets_not_accepted'):
        mod.wrist_anchor_points(data)
    assert np.array_equal(mod.wrist_anchor_points(data,bounded=True)['left'],[.1,.2,.9])


def test_grasp_requires_every_finger_palm_and_opposing_thumb():
    assert mod.grasp_region_gate([1,2,3,4,5],2,-.3)
    assert not mod.grasp_region_gate([1,2,3,4,6],2,-.3)
    assert not mod.grasp_region_gate([1]*5,6,-.3)
    assert not mod.grasp_region_gate([1]*5,2,.3)
    assert not mod.grasp_region_gate([1]*5,2,float('nan'))
