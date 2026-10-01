"""Coordinate, occlusion and annotation contract checks for local grip fitting."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

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
