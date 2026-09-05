import math
import ezdxf
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix, cKDTree


def normalize_splines_to_unit(spline_points):
    """Normalize all spline points to [0,1] using the global bounding box of all splines.
    
    Args:
        spline_points: (N, num_points, 2) array
    Returns:
        normalized_pts: (N, num_points, 2) in [0,1]
        bbox_params: dict with 'min_xy' (2,) and 'range_xy' (2,) for denormalization
    """
    all_pts = spline_points.reshape(-1, 2)
    min_xy = all_pts.min(axis=0).astype(np.float32)
    max_xy = all_pts.max(axis=0).astype(np.float32)
    range_xy = max_xy - min_xy
    range_xy = np.where(range_xy < 1e-8, 1.0, range_xy).astype(np.float32)
    
    normalized = ((spline_points - min_xy) / range_xy).astype(np.float32)
    bbox_params = {'min_xy': min_xy, 'range_xy': range_xy}
    return normalized, bbox_params

def get_spline_points(spline_entity, num_points=10):
    """Samples num_points evenly along a DXF SPLINE entity."""
    try:
        bspline = spline_entity.construction_tool()
        # approximate(segments=num_points-1) yields exactly num_points vertices
        pts = list(bspline.approximate(segments=num_points - 1))
        # Ensure we have exactly num_points (sometimes numerical issues cause variance)
        if len(pts) > num_points:
            pts = pts[:num_points]
        elif len(pts) < num_points:
            # Pad by repeating the last point
            while len(pts) < num_points:
                pts.append(pts[-1])
        return np.array([[p.x, p.y] for p in pts], dtype=np.float32)
    except Exception as e:
        # Fallback if approximation fails
        return np.zeros((num_points, 2), dtype=np.float32)

def extract_splines_from_dxf(dxf_path, num_points=10):
    """Reads a DXF file and returns an array of sampled points for all SPLINE entities.
    Returns: (num_splines, num_points, 2) array.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    splines = [e for e in msp if e.dxftype() == 'SPLINE']
    if not splines:
        return np.zeros((0, num_points, 2), dtype=np.float32)
        
    all_points = []
    for s in splines:
        pts = get_spline_points(s, num_points)
        all_points.append(pts)
        
    return np.array(all_points, dtype=np.float32)

def compute_node_features(spline_points):
    """
    Given (N, num_points, 2) array of spline points, compute node features.
    Features: 
    - Flattened points (relative to center of bounding box)
    - Center X, Center Y
    - Bounding Box Width, Bounding Box Height
    - Arc Length (approximate)
    """
    N, num_points, _ = spline_points.shape
    features = []
    
    for i in range(N):
        pts = spline_points[i]
        
        # Bounding Box
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        width = max_x - min_x
        height = max_y - min_y
        
        # Approximate length
        diffs = np.diff(pts, axis=0)
        length = np.sum(np.linalg.norm(diffs, axis=1))
        
        # Normalize points relative to center
        rel_pts = pts - np.array([center_x, center_y])
        flat_rel_pts = rel_pts.flatten()
        
        node_feat = np.concatenate([
            flat_rel_pts,
            [center_x, center_y, width, height, length]
        ])
        features.append(node_feat)
        
    return np.array(features, dtype=np.float32)

def build_edges(spline_points, k=8):
    """
    Build K-nearest neighbor edges based on spline center distances.
    Returns edge_index (2, num_edges) and edge_attr (num_edges, num_edge_features).
    """
    N = len(spline_points)
    if N == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4), dtype=torch.float)
        
    # Calculate centers
    centers = []
    for i in range(N):
        min_x, min_y = np.min(spline_points[i], axis=0)
        max_x, max_y = np.max(spline_points[i], axis=0)
        centers.append([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
    centers = np.array(centers, dtype=np.float32)
    
    # KDTree for K-NN
    k_actual = min(k + 1, N) # +1 because query returns the point itself
    tree = cKDTree(centers)
    distances, indices = tree.query(centers, k=k_actual)
    
    edge_list = []
    edge_attrs = []
    
    for i in range(N):
        for j_idx in range(1, k_actual): # Skip 0 (self)
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            
            # Relative position
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            angle = math.atan2(dy, dx)
            
            edge_list.append([i, j])
            edge_attrs.append([dist, dx, dy, angle])
            
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    
    return edge_index, edge_attr

def match_entities(corrupted_pts, perfect_pts):
    """
    Matches corrupted splines to perfect splines using the Hungarian algorithm
    based on the distance between their centroids and bounding box dimensions.
    Returns:
        targets: array where targets[i] is the index of the perfect spline
                 matching the i-th corrupted spline.
        distances: array of match distances for each corrupted spline.
                   Unmatched splines (when N_c > N_p) get distance = inf.
    """
    N_c = len(corrupted_pts)
    N_p = len(perfect_pts)
    
    if N_c == 0 or N_p == 0:
        return np.zeros(N_c, dtype=np.int32), np.full(N_c, np.inf, dtype=np.float32)
        
    def get_features(pts_array):
        feats = []
        for pts in pts_array:
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
            w, h = max_x - min_x, max_y - min_y
            feats.append([cx, cy, w, h])
        return np.array(feats)
        
    feats_c = get_features(corrupted_pts)
    feats_p = get_features(perfect_pts)
    
    # Distance matrix (Euclidean on the [cx, cy, w, h] feature space)
    dist_mat = distance_matrix(feats_c, feats_p)
    
    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(dist_mat)
    
    targets = np.zeros(N_c, dtype=np.int32)
    distances = np.full(N_c, np.inf, dtype=np.float32)
    
    matched_dict = {r: c for r, c in zip(row_ind, col_ind)}
    for i in range(N_c):
        if i in matched_dict:
            targets[i] = matched_dict[i]
            distances[i] = dist_mat[i, matched_dict[i]]
        else:
            # Fallback for unassigned (if N_c > N_p) — mark as unmatched
            targets[i] = np.argmin(dist_mat[i])
            distances[i] = np.inf
            
    return targets, distances


def compute_normalization_stats(all_features, all_targets, all_edge_attrs):
    """Compute dataset-wide mean and std for features, targets, and edge attributes.
    
    Args:
        all_features: list of (N_i, feat_dim) arrays
        all_targets: list of (N_i, tgt_dim) arrays (only matched nodes)
        all_edge_attrs: list of (E_i, edge_dim) arrays
    Returns:
        dict with mean/std for each
    """
    feats = np.concatenate(all_features, axis=0)
    tgts = np.concatenate(all_targets, axis=0) if len(all_targets) > 0 else np.zeros((1, all_features[0].shape[1]))
    edges = np.concatenate(all_edge_attrs, axis=0) if len(all_edge_attrs) > 0 else np.zeros((1, 4))
    
    return {
        'feat_mean': feats.mean(axis=0).astype(np.float32),
        'feat_std': (feats.std(axis=0) + 1e-8).astype(np.float32),
        'tgt_mean': tgts.mean(axis=0).astype(np.float32),
        'tgt_std': (tgts.std(axis=0) + 1e-8).astype(np.float32),
        'edge_mean': edges.mean(axis=0).astype(np.float32),
        'edge_std': (edges.std(axis=0) + 1e-8).astype(np.float32),
    }
