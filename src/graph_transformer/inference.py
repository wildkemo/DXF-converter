import os
import argparse
import ezdxf
import torch
import numpy as np
from .dxf_graph_utils import extract_splines_from_dxf, compute_node_features, build_edges, normalize_splines_to_unit
from .model import GraphTransformer

def infer_and_correct(corrupted_dxf_path, output_dxf_path, model_path, num_points=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Feature dimensions
    node_in_dim = (num_points * 2) + 5
    edge_in_dim = 4
    out_dim = num_points * 2
    
    # Load model
    model = GraphTransformer(
        node_in_dim=node_in_dim, 
        edge_in_dim=edge_in_dim, 
        hidden_dim=256, 
        out_dim=out_dim
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}, using random weights.")
    
    model.eval()
    
    # Load normalization stats (saved by training dataset processing)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(model_path), '..'))
    norm_stats_path = os.path.join(root_dir, 'processed', 'norm_stats_train.pt')
    
    if not os.path.exists(norm_stats_path):
        print(f"Error: Normalization stats not found at {norm_stats_path}.")
        print("Make sure training has been run first to generate normalization statistics.")
        return
    
    norm_stats = torch.load(norm_stats_path, map_location='cpu', weights_only=False)
    print(f"Loaded normalization stats from {norm_stats_path}")
    
    # Process corrupted graph
    corrupted_pts_raw = extract_splines_from_dxf(corrupted_dxf_path, num_points)
    if len(corrupted_pts_raw) == 0:
        print("No splines found in input DXF.")
        return
    
    print(f"Found {len(corrupted_pts_raw)} splines in input DXF.")
    
    # Normalize input splines to [0,1] — same as training
    corrupted_norm, bbox_params = normalize_splines_to_unit(corrupted_pts_raw)
    
    # Compute features on normalized coordinates
    raw_features = compute_node_features(corrupted_norm)
    edge_index, edge_attr_raw = build_edges(corrupted_norm, k=8)
    
    # Apply dataset-wide normalization (z-score) to features and edges
    feat_mean = torch.tensor(norm_stats['feat_mean'], dtype=torch.float, device=device)
    feat_std = torch.tensor(norm_stats['feat_std'], dtype=torch.float, device=device)
    edge_mean = torch.tensor(norm_stats['edge_mean'], dtype=torch.float, device=device)
    edge_std = torch.tensor(norm_stats['edge_std'], dtype=torch.float, device=device)
    
    x = torch.tensor(raw_features, dtype=torch.float, device=device)
    x = (x - feat_mean) / feat_std
    edge_index = edge_index.to(device)
    edge_attr = edge_attr_raw.to(device)
    edge_attr = (edge_attr - edge_mean) / edge_std
    
    # Forward pass
    with torch.no_grad():
        delta_pred_norm = model(x, edge_index, edge_attr).cpu().numpy()
    
    # Denormalize predicted deltas from z-score space → [0,1] space
    tgt_mean = norm_stats['tgt_mean']
    tgt_std = norm_stats['tgt_std']
    delta_pred_unit = delta_pred_norm * tgt_std + tgt_mean
        
    # Apply corrections in [0,1] space
    delta_pred_unit = delta_pred_unit.reshape(-1, num_points, 2)
    corrected_norm = corrupted_norm + delta_pred_unit
    
    # Denormalize back to original coordinate space using the input's bounding box
    corrected_pts = corrected_norm * bbox_params['range_xy'] + bbox_params['min_xy']
    
    # Write to new DXF
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    for i in range(len(corrected_pts)):
        fit_points = [(float(p[0]), float(p[1])) for p in corrected_pts[i]]
        msp.add_spline(fit_points=fit_points)
        
    doc.saveas(output_dxf_path)
    print(f"Saved corrected DXF to {output_dxf_path} ({len(corrected_pts)} splines)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Transformer DXF Correction Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to corrupted DXF")
    parser.add_argument("--output", type=str, required=True, help="Path to save corrected DXF")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    infer_and_correct(args.input, args.output, args.model)
