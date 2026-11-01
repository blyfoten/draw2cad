"""Score a predicted DXF against ground truth, in millimetres.

Representation-agnostic: walls may be drawn as closed polylines, line pairs, hatches or blocks.
What is compared is the wall *faces* (boundary linework on wall layers) against the true wall
faces, after removing an arbitrary translation (the origin of a converted drawing is free, but its
scale and orientation are not).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.ops import unary_union

from bench.dxfio import Linework, densify
from draw2cad.dxf import iter_polygons
from draw2cad.plan import Plan
from draw2cad.layers import WALL_CATEGORIES, Category

TOLERANCES = (25, 50, 100)


@dataclass
class OpeningScore:
    gt: int
    pred: int
    matched: int

    @property
    def recall(self) -> float:
        return self.matched / self.gt if self.gt else 1.0

    @property
    def precision(self) -> float:
        return self.matched / self.pred if self.pred else (1.0 if not self.gt else 0.0)


@dataclass
class CaseScore:
    footprint_err_mm: tuple[float, float]  # predicted minus true, outer width and depth
    scale_err_pct: tuple[float, float]  # x, y; positive = drawn too large
    face_err_median_mm: float  # among predicted wall faces within 200 mm of a true face
    face_err_p95_mm: float
    completeness: dict[int, float]  # share of true wall faces with a prediction within tol
    precision: dict[int, float]  # share of predicted wall linework within tol of a true face
    doors: OpeningScore
    windows: OpeningScore
    ext_int_accuracy: float  # matched wall faces whose exterior/interior layer is right
    translation_mm: tuple[float, float]

    def f1(self, tol: int = 50) -> float:
        c, p = self.completeness[tol], self.precision[tol]
        return 2 * c * p / (c + p) if c + p else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["f1_50"] = self.f1(50)
        for k in ("doors", "windows"):
            o = getattr(self, k)
            d[k].update(recall=o.recall, precision=o.precision)
        return d


def gt_wall_faces(plan: Plan, step: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Sampled true wall-face points and, per point, whether it belongs to an exterior wall."""
    solid = plan.wall_solid()
    pts = []
    for poly in iter_polygons(solid):
        for ring in (poly.exterior, *poly.interiors):
            pts.append(densify(np.asarray(ring.coords)[:, :2], step))
    pts = np.vstack(pts)
    ext = unary_union([w.polygon() for w in plan.walls if w.exterior])
    is_ext = shapely.distance(ext, shapely.points(pts)) < 1.0
    return pts, is_ext


def align_translation(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Translation that best lays `pred` onto `gt` (bbox-centre start, then trimmed ICP)."""
    t = (gt.min(0) + gt.max(0)) / 2 - (pred.min(0) + pred.max(0)) / 2
    tree = cKDTree(gt)
    for thr in (2000, 1000, 500, 250, 120, 60, 30):
        for _ in range(4):
            d, idx = tree.query(pred + t)
            m = d < thr
            if m.sum() < 20:
                break
            t = t + np.median(gt[idx[m]] - (pred[m] + t), axis=0)
    return t


def fit_axis_scale(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-axis scale s such that gt ~ s * pred + c, fitted by trimmed ICP from a bounding-box start.

    Fitted separately from the translation-only alignment so that a large scale error is still
    measured correctly instead of only over the small region that happens to line up.
    """
    lo, hi = np.percentile(pred, 0.2, axis=0), np.percentile(pred, 99.8, axis=0)
    s = np.ptp(gt, axis=0) / np.maximum(hi - lo, 1e-9)
    c = (gt.min(0) + gt.max(0)) / 2 - s * (lo + hi) / 2
    tree = cKDTree(gt)
    for thr in (2000, 1000, 500, 250, 120, 60, 30):
        for _ in range(4):
            d, idx = tree.query(pred * s + c)
            m = d < thr
            if m.sum() < 50:
                break
            for ax in (0, 1):
                if np.ptp(pred[m, ax]) > 1000:
                    s[ax], c[ax] = np.polyfit(pred[m, ax], gt[idx[m], ax], 1)
    return s


def opening_clusters(lw: Linework, cat: Category, link: float = 20.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group linework on a door/window layer into symbols; returns their bounding boxes (lo, hi).

    Polylines join a group when an *endpoint* of one touches the other (leaf to arc, glass line to
    jamb). Joining on any nearby point would merge neighbouring doors whose swing arcs cross.
    """
    lines = lw.lines.get(cat, [])
    if not lines:
        return []
    dense = [densify(pl, 10.0) for pl in lines]
    pts = np.vstack(dense)
    pt_owner = np.concatenate([np.full(len(d), i) for i, d in enumerate(dense)])
    ends = np.vstack([[pl[0], pl[-1]] for pl in lines])
    end_owner = np.repeat(np.arange(len(lines)), 2)
    hits = cKDTree(ends).query_ball_tree(cKDTree(pts), link)
    rows = [end_owner[i] for i, h in enumerate(hits) for _ in h]
    cols = [pt_owner[j] for h in hits for j in h]
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(lines), len(lines)))
    n, labels = connected_components(adj, directed=False)
    boxes = []
    for i in range(n):
        pts = np.vstack([lines[j] for j in np.flatnonzero(labels == i)])
        boxes.append((pts.min(0), pts.max(0)))
    return boxes


def match_openings(gt_centers: np.ndarray, boxes, t: np.ndarray, radius: float = 300.0) -> OpeningScore:
    if len(gt_centers) == 0 or not boxes:
        return OpeningScore(len(gt_centers), len(boxes), 0)
    lo = np.array([b[0] for b in boxes]) + t
    hi = np.array([b[1] for b in boxes]) + t
    # Distance from each true opening centre to each predicted symbol's bounding box.
    gap = np.maximum(0, np.maximum(lo[None] - gt_centers[:, None], gt_centers[:, None] - hi[None]))
    dist = np.hypot(gap[..., 0], gap[..., 1])
    matched, used = 0, set()
    for gi, pi in sorted(np.argwhere(dist < radius).tolist(), key=lambda ij: dist[ij[0], ij[1]]):
        if gi in used or ("p", pi) in used:
            continue
        used.update({gi, ("p", pi)})
        matched += 1
    return OpeningScore(len(gt_centers), len(boxes), matched)


def score_case(pred: Linework, plan: Plan) -> CaseScore:
    gt, gt_ext = gt_wall_faces(plan)
    wall_cats = [Category.WALL_EXT, Category.WALL_INT, Category.WALL]
    per_cat = [(c, pred.points([c])) for c in wall_cats]
    pp = np.vstack([p for _, p in per_cat]) if any(len(p) for _, p in per_cat) else np.zeros((0, 2))
    pcat = np.concatenate([np.full(len(p), c.value, dtype=object) for c, p in per_cat]) if len(pp) else []

    centers = {k: np.array([o.center(plan.wall(o.wall)) for o in plan.openings if o.kind == k]).reshape(-1, 2)
               for k in ("door", "window")}
    if len(pp) == 0:
        zero = {t: 0.0 for t in TOLERANCES}
        return CaseScore((float("nan"),) * 2, (float("nan"),) * 2, float("nan"), float("nan"), zero, zero,
                         OpeningScore(len(centers["door"]), 0, 0), OpeningScore(len(centers["window"]), 0, 0),
                         0.0, (0.0, 0.0))

    t = align_translation(pp, gt)
    pa = pp + t
    d_pg, idx_pg = cKDTree(gt).query(pa)
    d_gp, _ = cKDTree(pa).query(gt)

    near = d_pg < 200
    face_med = float(np.median(d_pg[near])) if near.any() else float("nan")
    face_p95 = float(np.percentile(d_pg[near], 95)) if near.any() else float("nan")

    lo, hi = np.percentile(pa, 0.2, axis=0), np.percentile(pa, 99.8, axis=0)
    glo, ghi = gt.min(0), gt.max(0)
    footprint = tuple(float(v) for v in (hi - lo) - (ghi - glo))

    s = fit_axis_scale(pp, gt)
    scale = [float((1 / v - 1) * 100) if np.isfinite(v) else float("nan") for v in s]

    labelled = (d_pg < 100) & np.isin(pcat, [Category.WALL_EXT.value, Category.WALL_INT.value])
    matched_any = d_pg < 100
    correct = labelled & ((pcat == Category.WALL_EXT.value) == gt_ext[idx_pg])
    ext_int = float(correct.sum() / matched_any.sum()) if matched_any.any() else 0.0

    return CaseScore(
        footprint_err_mm=footprint,
        scale_err_pct=tuple(scale),
        face_err_median_mm=face_med,
        face_err_p95_mm=face_p95,
        completeness={tol: float((d_gp < tol).mean()) for tol in TOLERANCES},
        precision={tol: float((d_pg < tol).mean()) for tol in TOLERANCES},
        doors=match_openings(centers["door"], opening_clusters(pred, Category.DOOR), t),
        windows=match_openings(centers["window"], opening_clusters(pred, Category.WINDOW), t),
        ext_int_accuracy=ext_int,
        translation_mm=(float(t[0]), float(t[1])),
    )


# --- real samples: only known dimensions are available -------------------------------------------


def wall_groups(pred: Linework, link: float = 400.0) -> list[np.ndarray]:
    """Connected groups of wall linework (separate plans on one sheet end up in separate groups)."""
    pts = pred.points(WALL_CATEGORIES, step=50.0)
    if len(pts) == 0:
        return []
    tree = cKDTree(pts)
    n, labels = connected_components(tree.sparse_distance_matrix(tree, link), directed=False)
    return [pts[labels == i] for i in range(n)]


def score_footprints(pred: Linework, footprints: list[dict]) -> list[dict]:
    """For each known outer footprint, the closest-matching wall group's width/depth error in mm."""
    groups = [g for g in wall_groups(pred) if len(g) >= 20]
    sizes = [np.ptp(g, axis=0) for g in groups]
    out = []
    for fp in footprints:
        want = np.array([fp["width"], fp["depth"]], dtype=float)
        if not sizes:
            out.append({**fp, "found": False})
            continue
        errs = [s - want for s in sizes]
        best = min(errs, key=lambda e: abs(e[0]) + abs(e[1]))
        out.append({**fp, "found": True, "width_err_mm": float(best[0]), "depth_err_mm": float(best[1])})
    return out
