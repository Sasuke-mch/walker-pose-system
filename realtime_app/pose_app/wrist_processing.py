"""Single-detector upper-limb views and causal point processing.

All geometric outputs return to original fisheye pixels. No manual labels,
other detector predictions or future frames are used to construct a view.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .fisheye_camera import fisheye_project_numpy


def raw_to_upright(points, camera):
    p = np.asarray(points, float)
    return np.stack((p[..., 1], 1919-p[..., 0]), -1) if camera == 'left' else np.stack((1079-p[..., 1], p[..., 0]), -1)


def upright_to_raw(points, camera):
    p = np.asarray(points, float)
    return np.stack((1919-p[..., 1], p[..., 0]), -1) if camera == 'left' else np.stack((p[..., 1], 1079-p[..., 0]), -1)


def upper_box(previous, image_size=(1080, 1920), threshold=.25):
    """One combined shoulder/elbow/wrist ROI from the previous observation."""
    p = np.asarray(previous, float)[5:11]
    if p.shape != (6, 3) or not np.isfinite(p).all() or np.any(p[:, 2] < threshold):
        return None
    lo, hi = p[:, :2].min(0), p[:, :2].max(0)
    # Keep torso/upper-arm context and the hand beyond the wrist.
    span = np.maximum(hi-lo, [200, 250])
    lo -= span*[.25, .30]
    hi += span*[.25, .45]
    w, h = image_size
    lo, hi = np.maximum(lo, [0, 0]), np.minimum(hi, [w-1, h-1])
    if np.any(hi-lo < 100):
        return None
    return np.r_[lo, hi]


def fisheye_rays(points, K, D):
    q = cv2.fisheye.undistortPoints(np.asarray(points, float).reshape(-1, 1, 2), K, D).reshape(-1, 2)
    return np.column_stack((q, np.ones(len(q))))


def project_raw(rays, K, D):
    rays = np.asarray(rays, float)
    return cv2.fisheye.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)


@dataclass
class PerspectiveView:
    rotation: np.ndarray
    intrinsic: np.ndarray
    K: np.ndarray
    D: np.ndarray
    camera: str
    size: tuple = (768, 1024)

    def to_raw(self, points):
        p = np.asarray(points, float).reshape(-1, 2)
        rays = np.column_stack((p, np.ones(len(p)))) @ np.linalg.inv(self.intrinsic).T
        return project_raw(rays @ self.rotation.T, self.K, self.D)

    def from_raw(self, points):
        rays = fisheye_rays(points, self.K, self.D) @ self.rotation
        if np.any(rays[:, 2] <= 1e-9):
            raise ValueError('raw point lies behind the local virtual camera')
        p = rays @ self.intrinsic.T
        return p[:, :2]/p[:, 2:3]

    def remap(self, raw_image):
        mx, my = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, self.rotation.T, self.intrinsic, self.size, cv2.CV_32FC1)
        return cv2.remap(raw_image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def perspective_view(box, K, D, camera, size=(768, 1024), support=None):
    x0, y0, x1, y1 = np.asarray(box, float)
    xx, yy = np.meshgrid(np.linspace(x0, x1, 5), np.linspace(y0, y1, 5))
    points = np.column_stack((xx.ravel(), yy.ravel())) if support is None else np.asarray(support, float).reshape(-1, 2)
    raw = upright_to_raw(points, camera)
    rays = fisheye_rays(raw, K, D)
    center = fisheye_rays(upright_to_raw([[(x0+x1)/2, (y0+y1)/2]], camera), K, D)[0] if support is None else rays.mean(0)
    ez = center/np.linalg.norm(center)
    right = np.array([0., 1., 0.]) if camera == 'left' else np.array([0., -1., 0.])
    ex = right-right.dot(ez)*ez
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    R = np.column_stack((ex, ey, ez))
    local = rays @ R
    if np.any(local[:, 2] <= .05):
        raise ValueError('perspective view crosses the camera horizon')
    uv = local[:, :2]/local[:, 2:3]
    low, high = uv.min(0), uv.max(0)
    if support is not None:
        mid = (low+high)/2
        low, high = mid-(high-low)*.675, mid+(high-low)*.675
    w, h = size
    focal = min((w-1)/(high[0]-low[0]), (h-1)/(high[1]-low[1]))*.95
    mid = (low+high)/2
    Kv = np.array([[focal, 0, (w-1)/2-focal*mid[0]], [0, focal, (h-1)/2-focal*mid[1]], [0, 0, 1.]])
    return PerspectiveView(R, Kv, K, D, camera, size)


def heatmap_candidates(heatmap, metadata, to_raw, reference_raw, radius_px=12):
    """Retain image-supported candidates close to the decoded raw point."""
    h, w = heatmap.shape
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    grid = np.array([(i, j) for j in range(max(0, y-2), min(h, y+3)) for i in range(max(0, x-2), min(w, x+3))], float)
    scale = np.asarray(metadata['bbox_scale']).reshape(2)
    center = np.asarray(metadata['bbox_center']).reshape(2)
    image_points = grid/[w-1, h-1]*scale + center-scale/2
    raw = to_raw(image_points)
    response = heatmap[grid[:, 1].astype(int), grid[:, 0].astype(int)]
    relative = response/max(float(heatmap.max()), 1e-8)
    keep = (np.linalg.norm(raw-reference_raw, axis=1) <= radius_px) & (relative >= .5)
    # The current UDP result is always available, with peak response assigned.
    return np.vstack((reference_raw, raw[keep])), np.r_[1., relative[keep]]


def epipolar_select(left, right, cal):
    """Local image evidence plus calibrated geometry; never force a point to a line."""
    lp, ls = left
    rp, rs = right
    T = cal.T_cam0_to_cam1_mm
    tx = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
    E = tx @ cal.R_cam0_to_cam1
    a, b = fisheye_rays(lp, cal.K0, cal.D0), fisheye_rays(rp, cal.K1, cal.D1)
    # Pixel-space differential: account for fisheye distortion on both views.
    ja = np.stack([fisheye_rays(lp+v, cal.K0, cal.D0)-a for v in ([1, 0], [0, 1])], -1)
    jb = np.stack([fisheye_rays(rp+v, cal.K1, cal.D1)-b for v in ([1, 0], [0, 1])], -1)
    residual = a @ E.T @ b.T
    ga = np.einsum('ikd,jk->ijd', ja, b @ E)
    gb = np.einsum('jkd,ik->ijd', jb, a @ E.T)
    distance2 = residual**2/np.maximum((ga*ga).sum(-1)+(gb*gb).sum(-1), 1e-12)
    movement = (np.linalg.norm(lp-lp[0], axis=1)[:, None]**2 + np.linalg.norm(rp-rp[0], axis=1)[None, :]**2)/64
    cost = -np.log(np.maximum(ls[:, None]*rs[None, :], 1e-8))+.3*distance2/25+.05*movement
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    return lp[i], rp[j], {'left_shift_px': float(np.linalg.norm(lp[i]-lp[0])), 'right_shift_px': float(np.linalg.norm(rp[j]-rp[0])), 'sampson_px': float(np.sqrt(distance2[i, j]))}


def refine_forearm(initial, observations, target_length, cal):
    """Small 3D elbow/wrist fit around accepted raw image evidence."""
    x0 = np.asarray(initial, float).reshape(2, 3)
    observations = np.asarray(observations, float).reshape(2, 2, 2)
    def project(x):
        a = fisheye_project_numpy(x*1000, cal.K0, cal.D0)
        b = fisheye_project_numpy(x @ cal.R_cam0_to_cam1.T*1000+cal.T_cam0_to_cam1_mm, cal.K1, cal.D1)
        return np.stack((a, b))
    def residual(flat):
        x = flat.reshape(2, 3)
        return np.r_[((project(x)-observations)/3).ravel(), (np.linalg.norm(x[1]-x[0])-target_length)/.015]
    fit = least_squares(residual, x0.ravel(), loss='soft_l1', max_nfev=20, ftol=1e-5, xtol=1e-5, gtol=1e-5)
    x = fit.x.reshape(2, 3)
    raw = project(x)
    shifts = np.linalg.norm(raw-observations, axis=-1)
    right = x @ cal.R_cam0_to_cam1.T+cal.T_cam0_to_cam1_mm/1000
    allowed = fit.success and np.isfinite(raw).all() and np.all(x[:, 2] > 0) and np.all(right[:, 2] > 0) and np.max(shifts) <= 12
    return (raw if allowed else observations.copy()), {'accepted_update': bool(allowed), 'max_shift_px': float(np.max(shifts)), 'initial_length_m': float(np.linalg.norm(x0[1]-x0[0])), 'final_length_m': float(np.linalg.norm(x[1]-x[0])), 'target_length_m': float(target_length)}


class CausalPointFilter:
    """Adaptive low-pass with speed response; missing values remain missing."""
    def __init__(self, min_cutoff=2., beta=.015):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.previous = self.filtered = self.time = None
        self.velocity = None

    def update(self, point, time, available=True):
        point = np.asarray(point, float)
        if not available or not np.isfinite(point).all():
            self.previous = self.filtered = self.time = self.velocity = None
            return np.full_like(point, np.nan), 'missing'
        if self.time is None or time <= self.time or time-self.time > .2:
            self.previous = self.filtered = point.copy()
            self.velocity = np.zeros_like(point)
            self.time = time
            return point.copy(), 'initialized'
        dt = time-self.time
        v = (point-self.previous)/dt
        alpha_v = 1/(1+1/(2*np.pi*dt))
        self.velocity = alpha_v*v+(1-alpha_v)*self.velocity
        cutoff = self.min_cutoff+self.beta*np.linalg.norm(self.velocity)
        alpha = 1/(1+1/(2*np.pi*cutoff*dt))
        self.filtered += alpha*(point-self.filtered)
        self.previous, self.time = point.copy(), time
        return self.filtered.copy(), 'filtered'
