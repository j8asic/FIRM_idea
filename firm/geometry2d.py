"""
2D polygon geometry + jittered point-cloud builders for the FIRM tests.

Polygon convention: ``poly`` is an (M, 2) array of vertices in order, implicitly
closed (edge M-1 -> 0). ``nearby_walls`` returns ALL edges within ``h_w`` (needed
for the K=2 wedge that exercises the Gram projector) -- the old firm_hydrostatic
single-nearest-wall query is a bug there.
"""
import numpy as np


# ---------------------------------------------------------------- primitives
def point_in_polygon(p, poly):
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def dist_point_segment(p, a, b):
    p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
    e = b - a
    denom = float(np.dot(e, e))
    t = 0.0 if denom == 0.0 else np.clip(np.dot(p - a, e) / denom, 0.0, 1.0)
    foot = a + t * e
    return float(np.linalg.norm(p - foot)), foot


def _outward_normal(a, b, poly):
    e = np.asarray(b, float) - np.asarray(a, float)
    nrm = np.array([e[1], -e[0]], float)
    nrm /= np.linalg.norm(nrm)
    mid = 0.5 * (np.asarray(a, float) + np.asarray(b, float))
    # orient so the normal points OUT of the polygon
    if point_in_polygon(mid + 1e-4 * nrm, poly):
        nrm = -nrm
    return nrm


def nearest_wall(p, poly):
    """(distance, outward_unit_normal, foot) of the single nearest edge."""
    best = (np.inf, None, None)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        d, foot = dist_point_segment(p, a, b)
        if d < best[0]:
            best = (d, _outward_normal(a, b, poly), foot)
    return best


def nearby_walls(p, poly, h_w):
    """List of (distance, outward_unit_normal, foot) for every edge within h_w.

    Edges whose outward normals are nearly identical (same flat wall split into
    collinear segments) are deduplicated, keeping the closest, so a flat wall
    contributes K=1 and a genuine corner contributes K=2.
    """
    hits = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        d, foot = dist_point_segment(p, a, b)
        if d < h_w:
            hits.append((d, _outward_normal(a, b, poly), foot))
    hits.sort(key=lambda t: t[0])
    unique = []
    for d, nrm, foot in hits:
        if any(np.dot(nrm, u[1]) > 0.999 for u in unique):
            continue
        unique.append((d, nrm, foot))
    return unique


def polygon_segments(poly):
    """Precompute (a, b, outward_unit_normals) for every polygon edge once, so a
    finely-segmented smooth boundary can be queried vectorised (see nearest_segment).
    Avoids the O(M)-per-particle Python loop in nearby_walls for large M."""
    poly = np.asarray(poly, float)
    M = len(poly)
    a = poly
    b = poly[(np.arange(M) + 1) % M]
    e = b - a
    nrm = np.stack([e[:, 1], -e[:, 0]], axis=1)
    nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    mid = 0.5 * (a + b)
    inside = np.array([point_in_polygon(mid[k] + 1e-4 * nrm[k], poly) for k in range(M)])
    nrm[inside] = -nrm[inside]
    return a, b, nrm


def nearest_segment(p, seg_a, seg_b, seg_n, h):
    """Vectorised nearest polygon edge to point ``p`` using precomputed segment arrays.
    Returns (distance, outward_unit_normal, foot) of the nearest edge if within ``h``,
    else ``None``. The perpendicular foot lies along the edge normal (clamped to the
    segment), which the FIRM curved-boundary closures rely on for linear-exactness."""
    p = np.asarray(p, float)
    e = seg_b - seg_a
    denom = (e * e).sum(1)
    t = np.where(denom > 0, ((p - seg_a) * e).sum(1) / np.maximum(denom, 1e-30), 0.0)
    t = np.clip(t, 0.0, 1.0)
    foot = seg_a + t[:, None] * e
    dist = np.linalg.norm(p - foot, axis=1)
    k = int(np.argmin(dist))
    if dist[k] < h:
        return float(dist[k]), seg_n[k], foot[k]
    return None


def raycast_segment_hit(origin, direction, poly, max_dist=np.inf):
    """Nearest distance at which a ray hits any polygon edge, else inf.

    Used to unit-test the Sec 3.4.1 ray-cast wall suppression.
    """
    o = np.asarray(origin, float)
    dvec = np.asarray(direction, float)
    dvec = dvec / np.linalg.norm(dvec)
    best = np.inf
    n = len(poly)
    for i in range(n):
        a = np.asarray(poly[i], float)
        b = np.asarray(poly[(i + 1) % n], float)
        e = b - a
        # solve o + s*d = a + t*e , 0<=t<=1, s>=0
        mat = np.array([[dvec[0], -e[0]], [dvec[1], -e[1]]])
        det = np.linalg.det(mat)
        if abs(det) < 1e-14:
            continue
        s, t = np.linalg.solve(mat, a - o)
        if s > 1e-12 and -1e-9 <= t <= 1 + 1e-9 and s <= max_dist:
            best = min(best, s)
    return best


# ---------------------------------------------------------------- cloud builders
def jittered_box(dx, jitter=0.3, seed=7, lo=0.0, hi=1.0, d=2):
    """Cell-centred lattice in [lo,hi]^d, jittered +/- jitter*dx, clipped to the box.

    Edge particles get one-sided (truncated) support naturally -- exactly the
    boundary rows the linear-exactness claims must hold on.
    """
    m = max(int(round((hi - lo) / dx)), 1)
    xs = lo + (np.arange(m) + 0.5) * dx
    grids = np.meshgrid(*([xs] * d), indexing="ij")
    pos = np.stack([g.ravel() for g in grids], axis=-1)
    rng = np.random.default_rng(seed)
    pos = pos + rng.uniform(-jitter * dx, jitter * dx, pos.shape)
    keep = np.all((pos > lo) & (pos < hi), axis=1)
    return pos[keep]


def box_interior_mask(pos, margin, lo=0.0, hi=1.0):
    """True for particles at least ``margin`` from every box face (full support)."""
    return np.all((pos > lo + margin) & (pos < hi - margin), axis=1)


def half_plane_cloud(dx, jitter=0.3, seed=7, lo=0.0, hi=1.0, axis=1, cut=0.7, keep="below"):
    """Box lattice with everything beyond ``cut`` along ``axis`` removed.

    Synthesises a single flat wall at coordinate ``cut`` with outward normal
    +e_axis (keep='below'). Returns (pos, wall_normal). Particles just under the
    cut have truncated support whose offset o points inward (-e_axis).
    """
    pos = jittered_box(dx, jitter, seed, lo, hi, d=2)
    normal = np.zeros(2)
    normal[axis] = 1.0 if keep == "below" else -1.0
    if keep == "below":
        pos = pos[pos[:, axis] < cut]
    else:
        pos = pos[pos[:, axis] > cut]
    return pos, normal, cut


def wedge_polygon(theta_deg, R=1.5):
    """Triangle whose apex (at origin) has interior angle theta_deg.

    The two edges meeting at the apex are the wedge walls of interest. R is made
    large so apex-region test particles never see the closing (far) edge.
    """
    th = np.deg2rad(theta_deg)
    return np.array([[0.0, 0.0], [R, 0.0], [R * np.cos(th), R * np.sin(th)]])


def wedge_cloud(theta_deg, dx, jitter=0.3, seed=7, R=1.5):
    """Jittered cloud filling a wedge (triangle) of interior angle theta_deg."""
    poly = wedge_polygon(theta_deg, R)
    lo = poly.min(axis=0)
    hi = poly.max(axis=0)
    mx = max(int(round((hi[0] - lo[0]) / dx)), 1)
    my = max(int(round((hi[1] - lo[1]) / dx)), 1)
    xs = lo[0] + (np.arange(mx) + 0.5) * dx
    ys = lo[1] + (np.arange(my) + 0.5) * dx
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], axis=-1)
    rng = np.random.default_rng(seed)
    pos = pos + rng.uniform(-jitter * dx, jitter * dx, pos.shape)
    keep = np.array([point_in_polygon(p, poly) for p in pos])
    return pos[keep], poly


def star_polygon(n=400, r0=0.5, amp=0.2, k=5, center=(0.5, 0.5)):
    """Smooth star/flower boundary r(theta) = r0 + amp*sin(k*theta), polygonised into
    n short straight segments (Gibou et al. 2002 star: r0=0.5, amp=0.2, k=5). The many
    short segments approximate the curved wall; the per-segment reflection plane is
    exact for each straight segment (curved-wall ghosts are noted as future work)."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = r0 + amp * np.sin(k * th)
    cx, cy = center
    return np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], axis=1)


def flower_polygon(n=600, base=1.0, amp=0.16, k=8, center=(0.0, 0.0)):
    """Papac--Gibou-style flower boundary r(theta) = base - amp*cos(k*theta)
    (their level set phi = amp*cos(8 theta) + |x| - base, here normalised to a unit
    scale). Used for the Robin benchmark."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = base - amp * np.cos(k * th)
    cx, cy = center
    return np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], axis=1)


def polygon_cloud(poly, dx, jitter=0.3, seed=7):
    """Jittered interior cloud of an arbitrary simple polygon (no free-surface cap).

    Lattice over the bounding box, kept inside the polygon, then jittered +/- jitter*dx
    and re-clipped so boundary rows keep one-sided (truncated) support. ``jitter`` is the
    half-amplitude as a fraction of dx (chaos c maps to jitter = c/2; see convergence.py).
    """
    poly = np.asarray(poly, float)
    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)
    mx = max(int(round((xmax - xmin) / dx)), 1)
    my = max(int(round((ymax - ymin) / dx)), 1)
    xs = xmin + (np.arange(mx) + 0.5) * dx
    ys = ymin + (np.arange(my) + 0.5) * dx
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], axis=-1)
    rng = np.random.default_rng(seed)
    pos = pos + rng.uniform(-jitter * dx, jitter * dx, pos.shape)
    keep = np.array([point_in_polygon(p, poly) for p in pos])
    return pos[keep]


def tank_cloud(dx, fill_h, tank, jitter=0.3, seed=7):
    """Wetted, jittered cloud inside a polygonal tank, below the free surface.

    Reuses the firm_hydrostatic sampler shape; clips jittered points to stay
    inside the polygon and below fill_h.
    """
    tank = np.asarray(tank, float)
    xmin, ymin = tank.min(axis=0)
    xmax, ymax = tank.max(axis=0)
    rng = np.random.default_rng(seed)
    lattice = []
    x = xmin + 0.5 * dx
    while x < xmax:
        y = ymin + 0.5 * dx
        while y < ymax:
            if y <= fill_h + 1e-9 and point_in_polygon((x, y), tank):
                lattice.append((x, y))
            y += dx
        x += dx
    lattice = np.array(lattice)
    perturbed = lattice + rng.uniform(-jitter * dx, jitter * dx, lattice.shape)
    keep = [k for k in range(len(perturbed))
            if perturbed[k, 1] <= fill_h and point_in_polygon(perturbed[k], tank)]
    return perturbed[keep]
