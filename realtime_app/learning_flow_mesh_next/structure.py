"""Budgeted nonadjacent triangle-intersection diagnostic, not anatomy certification."""
import numpy as np


def segment_triangle(start, end, triangle, eps):
    direction = end-start
    edge1, edge2 = triangle[1]-triangle[0], triangle[2]-triangle[0]
    cross = np.cross(direction, edge2)
    determinant = edge1 @ cross
    if abs(determinant) <= eps:
        return False
    offset = start-triangle[0]
    u = (offset @ cross)/determinant
    q = np.cross(offset, edge1)
    v, t = (direction @ q)/determinant, (edge2 @ q)/determinant
    return -eps <= u <= 1+eps and -eps <= v and u+v <= 1+eps and -eps <= t <= 1+eps


def coplanar_overlap(a, b, normal, eps):
    axis = int(np.argmax(np.abs(normal)))
    a, b = np.delete(a, axis, -1), np.delete(b, axis, -1)
    def cross2(x, y):
        return x[0]*y[1]-x[1]*y[0]
    def inside(point, tri):
        values = [cross2(tri[(i+1) % 3]-tri[i], point-tri[i]) for i in range(3)]
        return min(values) >= -eps or max(values) <= eps
    if any(inside(point, b) for point in a) or any(inside(point, a) for point in b):
        return True
    for i in range(3):
        p, q = a[i], a[(i+1) % 3]
        for j in range(3):
            r, s = b[j], b[(j+1) % 3]
            if (max(min(p[0], q[0]), min(r[0], s[0])) > min(max(p[0], q[0]), max(r[0], s[0]))+eps
                or max(min(p[1], q[1]), min(r[1], s[1])) > min(max(p[1], q[1]), max(r[1], s[1]))+eps):
                continue
            if (cross2(q-p, r-p)*cross2(q-p, s-p) <= eps
                and cross2(s-r, p-r)*cross2(s-r, q-r) <= eps):
                return True
    return False


def triangles_intersect(a, b, eps=1e-9):
    normal_a, normal_b = np.cross(a[1]-a[0], a[2]-a[0]), np.cross(b[1]-b[0], b[2]-b[0])
    if np.linalg.norm(normal_a) <= eps or np.linalg.norm(normal_b) <= eps:
        return False  # Separately reported as degenerate, never silently certified.
    if np.max(np.abs((b-a[0])@normal_a)) <= eps*np.linalg.norm(normal_a):
        return coplanar_overlap(a, b, normal_a, eps)
    return any(segment_triangle(a[i], a[(i+1) % 3], b, eps) or
               segment_triangle(b[i], b[(i+1) % 3], a, eps) for i in range(3))


def self_intersection_diagnostic(vertices, faces, *, max_pairs=100000, eps=1e-9):
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
        or not np.issubdtype(faces.dtype, np.integer) or max_pairs < 1 or eps <= 0):
        raise ValueError("V3 vertices, integer F3 triangles and positive diagnostic budget required")
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("invalid mesh coordinates/topology")
    triangles = vertices[faces]
    minimum, maximum = triangles.min(1), triangles.max(1)
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    order = np.argsort(minimum[:, 0], kind="stable")
    intersections, checked, exhausted = [], 0, False
    active = []
    for index in order:
        active = [other for other in active if maximum[other, 0] >= minimum[index, 0]-eps]
        for other in active:
            if np.any(maximum[index]+eps < minimum[other]) or np.any(maximum[other]+eps < minimum[index]):
                continue
            if set(faces[index]) & set(faces[other]):
                continue
            if checked >= max_pairs:
                exhausted = True
                break
            checked += 1
            if triangles_intersect(triangles[index], triangles[other], eps):
                intersections.append([int(index), int(other)])
        if exhausted:
            break
        active.append(index)
    return {"intersection_pair_count_lower_bound": len(intersections), "pairs": intersections,
            "checked_nonadjacent_aabb_pairs": checked, "budget_exhausted": exhausted,
            "complete_for_declared_nonadjacent_test": not exhausted and not (area <= eps).any(),
            "degenerate_triangles": int((area <= eps).sum()), "epsilon_m": eps,
            "scope": "touching counts; shared-vertex triangles excluded; no anatomy/physical-contact claim"}
