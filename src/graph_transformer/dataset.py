import os
import glob
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from .dxf_graph_utils import (
    extract_splines_from_dxf,
    compute_node_features,
    build_edges,
    match_entities,
    normalize_splines_to_unit,
    compute_normalization_stats
)

class DXFGraphDataset(Dataset):
    def __init__(self, root, split='train', num_points=10, k_neighbors=8, transform=None, pre_transform=None):
        """
        root: The root directory containing 'dataset/' (e.g. /home/kemo/projects/fake/python/DXF-converter)
        split: 'train', 'val', or 'test'
        """
        self.dataset_dir = os.path.join(root, 'dataset')
        self.num_points = num_points
        self.k_neighbors = k_neighbors
        self.split = split
        
        # Get all valid pair directories
        pair_dirs = glob.glob(os.path.join(self.dataset_dir, 'pair_*'))
        # Sort by integer ID
        pair_dirs.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
        
        # Filter pairs that have both DXF files
        valid_pairs = []
        for p in pair_dirs:
            if os.path.exists(os.path.join(p, 'not perfict.dxf')) and os.path.exists(os.path.join(p, 'perfect.dxf')):
                valid_pairs.append(p)
                
        # Split (80% train, 10% val, 10% test)
        num_pairs = len(valid_pairs)
        train_end = int(0.8 * num_pairs)
        val_end = int(0.9 * num_pairs)
        
        if split == 'train':
            self.pair_dirs = valid_pairs[:train_end]
        elif split == 'val':
            self.pair_dirs = valid_pairs[train_end:val_end]
        elif split == 'test':
            self.pair_dirs = valid_pairs[val_end:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
            
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # We don't use this standard mechanism directly because of the dynamic pair folders
        return []

    @property
    def processed_file_names(self):
        # Generate names based on the selected split pairs
        return [f"data_{self.split}_{os.path.basename(p)}.pt" for p in self.pair_dirs]

    def download(self):
        pass
        
    def process(self):
        """Two-pass processing: first collect stats, then normalize and save.
        
        Both corrupted and perfect splines are normalized to [0,1] using their
        own bounding boxes before computing features and deltas. This removes
        the coordinate system / scale mismatch between the two.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        
        norm_stats_path = os.path.join(self.processed_dir, f'norm_stats_{self.split}.pt')
        
        # Check if all files already exist
        all_exist = all(
            os.path.exists(os.path.join(self.processed_dir, f"data_{self.split}_{os.path.basename(p)}.pt"))
            for p in self.pair_dirs
        )
        if all_exist and os.path.exists(norm_stats_path):
            return
        
        # --- Pass 1: Extract raw data, normalize to [0,1], and collect statistics ---
        print(f"[{self.split}] Pass 1: Extracting data and normalizing to [0,1]...")
        
        all_raw_features = []
        all_raw_targets = []    # Only from matched nodes
        all_raw_edge_attrs = []
        all_match_distances = []
        pair_data_cache = []
        skipped = 0
        
        for i, pair_dir in enumerate(self.pair_dirs):
            pair_name = os.path.basename(pair_dir)
            corrupted_path = os.path.join(pair_dir, 'not perfict.dxf')
            perfect_path = os.path.join(pair_dir, 'perfect.dxf')
            
            # Extract raw points
            corrupted_pts_raw = extract_splines_from_dxf(corrupted_path, self.num_points)
            perfect_pts_raw = extract_splines_from_dxf(perfect_path, self.num_points)
            
            if len(corrupted_pts_raw) == 0 or len(perfect_pts_raw) == 0:
                skipped += 1
                pair_data_cache.append(None)
                continue
            
            # Normalize EACH file to [0,1] independently — this removes scale/offset mismatch
            corrupted_norm, _ = normalize_splines_to_unit(corrupted_pts_raw)
            perfect_norm, _ = normalize_splines_to_unit(perfect_pts_raw)
                
            # Compute node features on normalized coordinates
            node_features = compute_node_features(corrupted_norm)
            
            # Build edges on normalized coordinates
            edge_index, edge_attr = build_edges(corrupted_norm, k=self.k_neighbors)
            
            # Match entities using normalized coordinates
            targets_idx, match_dists = match_entities(corrupted_norm, perfect_norm)
            
            # Compute delta in [0,1] space (much smaller and consistent)
            matched_perfect = perfect_norm[targets_idx]
            delta = matched_perfect - corrupted_norm
            delta_flat = delta.reshape(len(corrupted_norm), -1)
            
            # Cache for pass 2
            edge_attr_np = edge_attr.numpy()
            pair_data_cache.append({
                'pair_name': pair_name,
                'node_features': node_features,
                'edge_index': edge_index,
                'edge_attr_np': edge_attr_np,
                'delta_flat': delta_flat,
                'match_dists': match_dists,
            })
            
            # Collect stats from ALL nodes for features (model sees all nodes)
            all_raw_features.append(node_features)
            all_raw_edge_attrs.append(edge_attr_np)
            
            # Collect target stats only from well-matched nodes (finite distance)
            matched_mask = np.isfinite(match_dists)
            if matched_mask.sum() > 0:
                all_raw_targets.append(delta_flat[matched_mask])
                all_match_distances.extend(match_dists[matched_mask].tolist())
            
            if (i + 1) % 200 == 0:
                print(f"  Processed {i+1}/{len(self.pair_dirs)} pairs...")
        
        if len(all_raw_features) == 0:
            print(f"  Warning: No valid pairs found for split '{self.split}'.")
            return
            
        # Compute normalization statistics
        norm_stats = compute_normalization_stats(all_raw_features, all_raw_targets, all_raw_edge_attrs)
        
        # Compute match distance threshold: use 2x median of matched distances
        if len(all_match_distances) > 0:
            match_threshold = float(np.median(all_match_distances)) * 2.0
        else:
            match_threshold = float('inf')
        norm_stats['match_threshold'] = match_threshold
        
        # Save normalization stats
        torch.save(norm_stats, norm_stats_path)
        
        matched_count = 0
        total_count = 0
        
        print(f"[{self.split}] Pass 2: Normalizing features and saving (match_threshold={match_threshold:.4f})...")
        
        # --- Pass 2: Normalize and save ---
        for idx, pair_dir in enumerate(self.pair_dirs):
            pair_name = os.path.basename(pair_dir)
            processed_path = os.path.join(self.processed_dir, f"data_{self.split}_{pair_name}.pt")
            
            cached = pair_data_cache[idx]
            
            if cached is None:
                # Save empty data for skipped pairs
                num_points = self.num_points
                x = torch.zeros((0, num_points * 2 + 5), dtype=torch.float)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 4), dtype=torch.float)
                y = torch.zeros((0, num_points * 2), dtype=torch.float)
                match_mask = torch.zeros(0, dtype=torch.bool)
                empty_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, match_mask=match_mask)
                empty_data.pair_name = pair_name
                torch.save(empty_data, processed_path)
                continue
            
            # Normalize features using dataset-wide stats
            x_norm = (cached['node_features'] - norm_stats['feat_mean']) / norm_stats['feat_std']
            
            # Normalize targets using dataset-wide stats
            y_norm = (cached['delta_flat'] - norm_stats['tgt_mean']) / norm_stats['tgt_std']
            
            # Normalize edge attributes
            edge_attr_norm = (cached['edge_attr_np'] - norm_stats['edge_mean']) / norm_stats['edge_std']
            
            # Match mask: True for nodes with a good match (finite distance below threshold)
            match_mask = np.isfinite(cached['match_dists']) & (cached['match_dists'] < match_threshold)
            
            matched_count += match_mask.sum()
            total_count += len(match_mask)
            
            # Create PyG Data object
            data = Data(
                x=torch.tensor(x_norm, dtype=torch.float),
                edge_index=cached['edge_index'],
                edge_attr=torch.tensor(edge_attr_norm, dtype=torch.float),
                y=torch.tensor(y_norm, dtype=torch.float),
                match_mask=torch.tensor(match_mask, dtype=torch.bool),
            )
            data.pair_name = pair_name
            
            torch.save(data, processed_path)
        
        pct = 100 * matched_count / max(1, total_count)
        print(f"[{self.split}] Done! {len(self.pair_dirs) - skipped} pairs processed, {skipped} skipped.")
        print(f"[{self.split}] Matched nodes: {matched_count}/{total_count} ({pct:.1f}%)")

    def len(self):
        return len(self.pair_dirs)

    def get(self, idx):
        pair_name = os.path.basename(self.pair_dirs[idx])
        processed_path = os.path.join(self.processed_dir, f"data_{self.split}_{pair_name}.pt")
        
        if not os.path.exists(processed_path):
            # If not processed (e.g. empty graph), return correctly shaped empty Data to avoid DataLoader collation errors
            num_points = self.num_points
            x = torch.zeros((0, num_points * 2 + 5), dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4), dtype=torch.float)
            y = torch.zeros((0, num_points * 2), dtype=torch.float)
            match_mask = torch.zeros(0, dtype=torch.bool)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, match_mask=match_mask)
            
        data = torch.load(processed_path, weights_only=False)
        return data
