import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from .dataset import DXFGraphDataset
from .model import GraphTransformer

def train(num_epochs=100, batch_size=8, learning_rate=1e-3, num_points=10, patience=15):
    """Train the Graph Transformer model.
    
    Args:
        num_epochs: Maximum number of epochs to train.
        batch_size: Batch size for DataLoader.
        learning_rate: Initial learning rate.
        num_points: Number of sampled points per spline.
        patience: Early stopping patience (epochs without improvement).
    """
    # Device agnostic logic
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"  Graph Transformer Training")
    print(f"{'='*60}")
    print(f"  Device:         {device}")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Early stopping: {patience} epochs patience")
    print(f"{'='*60}\n")
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Datasets
    print("Initializing datasets...")
    train_dataset = DXFGraphDataset(root=root_dir, split='train', num_points=num_points)
    val_dataset = DXFGraphDataset(root=root_dir, split='val', num_points=num_points)
    print(f"  Train pairs: {len(train_dataset)}")
    print(f"  Val pairs:   {len(val_dataset)}\n")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Feature dimensions
    # x features = (num_points * 2) + 5 (cx, cy, w, h, length)
    node_in_dim = (num_points * 2) + 5
    # edge features = 4 (dist, dx, dy, angle)
    edge_in_dim = 4
    # output dimension = num_points * 2 (delta x, delta y for each point)
    out_dim = num_points * 2
    
    # Model
    model = GraphTransformer(
        node_in_dim=node_in_dim, 
        edge_in_dim=edge_in_dim, 
        hidden_dim=256, 
        out_dim=out_dim
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params:  {trainable_params:,} trainable / {total_params:,} total\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    os.makedirs(os.path.join(root_dir, 'checkpoints'), exist_ok=True)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"{'Epoch':>7} | {'Train Loss':>11} | {'Val Loss':>11} | {'LR':>10} | {'Time':>7} | {'Status'}")
    print(f"{'-'*7}-+-{'-'*11}-+-{'-'*11}-+-{'-'*10}-+-{'-'*7}-+-{'-'*20}")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # --- Training ---
        model.train()
        total_loss = 0
        num_nodes = 0
        
        for data in train_loader:
            if data.x is None or data.x.size(0) == 0:
                continue
            
            # Only train on batches that have matched nodes
            mask = data.match_mask
            if mask.sum() == 0:
                continue
                
            data = data.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            
            # Forward
            out = model(data.x, data.edge_index, data.edge_attr)
            
            # Loss only on well-matched nodes
            loss = F.smooth_l1_loss(out[mask], data.y[mask])
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            matched_count = mask.sum().item()
            total_loss += loss.item() * matched_count
            num_nodes += matched_count
            
        train_loss = total_loss / max(1, num_nodes)
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        val_nodes = 0
        with torch.no_grad():
            for data in val_loader:
                if data.x is None or data.x.size(0) == 0:
                    continue
                mask = data.match_mask
                if mask.sum() == 0:
                    continue
                data = data.to(device)
                mask = mask.to(device)
                out = model(data.x, data.edge_index, data.edge_attr)
                loss = F.smooth_l1_loss(out[mask], data.y[mask])
                matched_count = mask.sum().item()
                val_loss += loss.item() * matched_count
                val_nodes += matched_count
                
        val_loss = val_loss / max(1, val_nodes)
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        # Step the scheduler
        scheduler.step()
        
        # --- Early stopping + checkpointing ---
        status = ""
        if val_loss < best_val_loss and val_nodes > 0:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_path = os.path.join(root_dir, 'checkpoints', 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            if improvement < float('inf'):
                status = f"✓ saved (↓{improvement:.4f})"
            else:
                status = "✓ saved (first)"
        else:
            epochs_without_improvement += 1
            status = f"no improvement ({epochs_without_improvement}/{patience})"
        
        print(f"{epoch:>4}/{num_epochs:<3}| {train_loss:>11.6f} | {val_loss:>11.6f} | {current_lr:>10.6f} | {epoch_time:>5.1f}s | {status}")
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\n{'='*60}")
            print(f"  Early stopping triggered after {epoch} epochs.")
            print(f"  Best val loss: {best_val_loss:.6f}")
            print(f"{'='*60}")
            break
    else:
        print(f"\n{'='*60}")
        print(f"  Training completed ({num_epochs} epochs).")
        print(f"  Best val loss: {best_val_loss:.6f}")
        print(f"{'='*60}")

if __name__ == "__main__":
    train()
