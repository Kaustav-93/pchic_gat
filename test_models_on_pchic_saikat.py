#!/usr/bin/env python3
"""
test_models_on_pchic_saikat.py

Load trained Pangenomic GAT models and test on pchic_saikat data.

This script:
1. Loads a trained model checkpoint from results/model_fold*.pt
2. Reconstructs the Config and model architecture from the checkpoint
3. Loads pchic_saikat loop files and RNA-seq data
4. Builds graphs for a sample of genes
5. Runs inference and saves predictions

Usage:
    conda activate pangenomic-gat
    python test_models_on_pchic_saikat.py --fold 0 --device cuda --n-genes 100
"""

import argparse
from pathlib import Path
import sys
from dataclasses import dataclass, fields
import warnings

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "codes"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.utils import softmax as pyg_softmax
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score


# =============================================================================
# Config dataclass (must match the one used during training)
# =============================================================================

@dataclass
class Config:
    seed: int = 42
    fast_mode: bool = False
    fast_n_tissues: int = 8
    fast_n_genes: int = 2500
    genome_build: str = "hg19"
    promoter_window: int = 16384
    seq_len_fast: int = 4096
    max_neighbours: int = 32
    min_chicago: float = 5.0
    add_backbone_edges: bool = True
    histone_marks: tuple = ("H3K4me1", "H3K4me3", "H3K27ac", "H3K27me3", "H3K36me3", "H3K9me3", "CTCF")
    histone_stat: str = "mean"
    tpm_threshold: float = 1.0
    tau_threshold: float = 0.85
    link_holdout_frac: float = 0.15
    d_hidden: int = 128
    d_embed: int = 128
    gat_layers: int = 3
    gat_heads: int = 4
    dropout: float = 0.2
    attn_dropout: float = 0.1
    hyena_d_model: int = 128
    hyena_blocks: int = 4
    hyena_order: int = 2
    hyena_filter_order: int = 64
    stem_kernel: int = 15
    stem_stride: int = 4
    rc_equivariant: bool = True
    variants: tuple = ("both", "gat", "seq")
    run_hybrid: bool = True
    w_link: float = 1.0
    w_expr: float = 1.0
    w_tau: float = 0.5
    w_tissue: float = 0.5
    w_align: float = 0.3
    temperature: float = 0.1
    n_folds: int = 5
    group_split_by: str = "gene"
    epochs: int = 40
    epochs_fast: int = 12
    batch_size: int = 64
    lr: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    patience: int = 8
    num_workers: int = 0
    use_amp: bool = True

    @property
    def seq_len(self):
        return self.seq_len_fast if self.fast_mode else self.promoter_window

    @property
    def n_epochs(self):
        return self.epochs_fast if self.fast_mode else self.epochs


def config_from_dict(d):
    """Reconstruct Config from checkpoint dict."""
    cfg = Config()
    for field in fields(Config):
        if field.name in d:
            setattr(cfg, field.name, d[field.name])
    return cfg


# =============================================================================
# Simplified model stubs (minimal for inference)
# =============================================================================

class SequenceEncoder(nn.Module):
    """Placeholder - sequence tower not needed for structure-only testing."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return None


class StructureEncoder(nn.Module):
    """Simplified GAT encoder for graph embeddings."""
    def __init__(self, f_in, f_edge, d_hidden, d_out, layers, heads, dropout, attn_dropout):
        super().__init__()
        if d_hidden % heads != 0:
            raise ValueError(f"d_hidden ({d_hidden}) must be divisible by heads ({heads})")
        per_head = d_hidden // heads
        d_edge = max(8, d_hidden // 2)

        self.in_proj = nn.Linear(f_in, d_hidden)
        self.edge_proj = nn.Sequential(nn.Linear(f_edge, d_edge), nn.GELU())

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(layers):
            last = (i == layers - 1)
            self.convs.append(GATv2Conv(
                d_hidden, d_hidden if last else per_head, heads=heads,
                concat=not last, dropout=attn_dropout, edge_dim=d_edge,
                add_self_loops=True))
            self.norms.append(nn.LayerNorm(d_hidden))

        self.drop = nn.Dropout(dropout)
        self.pool_gate = nn.Linear(d_hidden, 1)
        self.out = nn.Sequential(nn.Linear(2 * d_hidden, d_out), nn.GELU(),
                                 nn.Linear(d_out, d_out))

    def forward(self, x, edge_index, edge_attr, batch, anchor_index):
        h = self.in_proj(x)
        e = self.edge_proj(edge_attr)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, e)
            h = self.drop(F.gelu(norm(h_new + h)))

        gate = pyg_softmax(self.pool_gate(h), batch)
        pooled = global_add_pool(gate * h, batch)
        anchor = h[anchor_index]
        return h, self.out(torch.cat([anchor, pooled], dim=-1)), None


class FiLMFusion(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gamma = nn.Linear(d, d)
        self.beta = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, z_seq, z_str):
        return self.norm(self.gamma(z_str) * z_seq + self.beta(z_str))


class PangenomicGAT(nn.Module):
    """Simplified model for inference (structure-only variant for testing)."""
    VARIANTS = ("both", "seq", "gat")

    def __init__(self, cfg, f_node, f_edge, n_tissues, variant="gat"):
        super().__init__()
        assert variant in self.VARIANTS
        self.variant = variant
        d = cfg.d_embed
        self.use_seq = variant in ("both", "seq")
        self.use_gat = variant in ("both", "gat")

        self.seq = SequenceEncoder() if self.use_seq else None
        self.struc = StructureEncoder(f_node, f_edge, cfg.d_hidden, d,
                                      cfg.gat_layers, cfg.gat_heads,
                                      cfg.dropout, cfg.attn_dropout) if self.use_gat else None
        self.fuse = FiLMFusion(d) if variant == "both" else None

        self.head_expr = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(),
                                       nn.Dropout(cfg.dropout), nn.Linear(d // 2, 1))
        self.head_tau = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(),
                                      nn.Dropout(cfg.dropout), nn.Linear(d // 2, 1))
        self.head_tissue = nn.Linear(d, n_tissues)
        d_h = cfg.d_hidden
        self.link_bilinear = nn.Parameter(torch.eye(d_h) + 0.01 * torch.randn(d_h, d_h))

    def encode(self, batch, seq_bank=None):
        dev = batch.x.device
        z_seq = z_str = h = attn = None

        if self.use_gat:
            anchor = (batch.ptr[:-1] if hasattr(batch, "ptr")
                      else torch.zeros(1, dtype=torch.long, device=dev))
            h, z_str, attn = self.struc(batch.x, batch.edge_index,
                                        batch.edge_attr, batch.batch, anchor)

        if self.variant == "both":
            z = self.fuse(z_seq, z_str)
        elif self.variant == "seq":
            z = z_seq
        else:
            z = z_str
        return z_seq, z_str, z, h, attn

    def forward(self, batch, seq_bank=None):
        z_seq, z_str, z, h, attn = self.encode(batch, seq_bank)
        return {
            "z_seq": z_seq, "z_str": z_str, "z": z, "h": h, "attn": attn,
            "logit_expr": self.head_expr(z).squeeze(-1),
            "logit_tau": self.head_tau(z).squeeze(-1),
            "logit_tissue": self.head_tissue(z),
        }


# =============================================================================
# Data loading and graph construction
# =============================================================================

def load_pchic_data(pchic_dir, rna_seq_file, n_genes=None):
    """Load PCHi-C loops and RNA-seq data."""
    print(f"Loading RNA-seq from {rna_seq_file}...")
    rna = pd.read_csv(rna_seq_file, sep='\t')
    print(f"  Loaded {len(rna)} genes, {len(rna.columns) - 2} samples")

    print(f"\nLoading PCHi-C loop files from {pchic_dir}...")
    loop_files = sorted(Path(pchic_dir).glob("*_merged_loop.txt"))
    print(f"  Found {len(loop_files)} tissue files")

    # Parse all loop files
    loops_list = []
    for loop_file in tqdm(loop_files[:5], desc="Parsing loops"):  # Limit to 5 tissues for testing
        tissue_name = '_'.join(loop_file.stem.split('_')[1:-2])
        sample_id = loop_file.stem.split('_')[0]

        df = pd.read_csv(loop_file, sep='\t')
        # Extract bait_gene from column 4 (0-indexed)
        df['gene'] = df.iloc[:, 4]  # bait_gene column
        df['sample_id'] = sample_id
        df['tissue'] = tissue_name
        df = df[df['gene'].notna() & (df['gene'] != '')]
        loops_list.append(df)

    loops = pd.concat(loops_list, ignore_index=True)
    print(f"\nTotal loops: {len(loops):,}")
    print(f"Unique genes: {loops['gene'].nunique()}")

    # Subsample genes if requested
    if n_genes:
        genes_sample = loops['gene'].unique()[:n_genes]
        loops = loops[loops['gene'].isin(genes_sample)]
        print(f"Subsampled to {n_genes} genes: {len(loops):,} loops")

    return rna, loops, loop_files[:5]


def build_simple_graph(gene_loops, gene_name, tissue_name):
    """Build a simple PyG graph from PCHi-C loops for one gene."""
    if len(gene_loops) < 2:
        return None

    # Node 0: promoter/bait (anchor)
    # Nodes 1+: interaction partners
    n_partners = min(len(gene_loops), 32)  # Cap at 32 neighbors
    n_nodes = n_partners + 1

    # Node features: [is_bait, score, n_reads] + 15 zeros for histones/masks
    nodes = []
    nodes.append(np.array([1.0] + [0.0] * 17, dtype=np.float32))  # Bait node

    edges_src, edges_dst, edges_attr = [], [], []

    for i, (_, row) in enumerate(gene_loops.head(n_partners).iterrows()):
        # Partner node features
        score = float(row.iloc[3]) if not pd.isna(row.iloc[3]) else 0.0
        n_reads = float(row.iloc[2]) if not pd.isna(row.iloc[2]) else 0.0
        nodes.append(np.array([0.0, score, n_reads] + [0.0] * 15, dtype=np.float32))

        # Bidirectional edges between bait (0) and partner (i+1)
        node_idx = i + 1
        edge_feat = np.array([np.log1p(score), np.log1p(n_reads), 0.0, 1.0], dtype=np.float32)
        edges_src.extend([0, node_idx])
        edges_dst.extend([node_idx, 0])
        edges_attr.extend([edge_feat, edge_feat])

    if len(edges_src) == 0:
        return None

    data = Data(
        x=torch.from_numpy(np.stack(nodes)),
        edge_index=torch.tensor([edges_src, edges_dst], dtype=torch.long),
        edge_attr=torch.from_numpy(np.stack(edges_attr)),
        num_nodes=n_nodes
    )

    return data


def run_inference(model, data_loader, device):
    """Run model inference and collect predictions."""
    model.eval()
    predictions = {
        'expr_p': [], 'tau_p': [], 'tis_p': [],
        'gene': [], 'tissue': []
    }

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference"):
            batch = batch.to(device)
            out = model(batch)

            predictions['expr_p'].append(torch.sigmoid(out['logit_expr']).cpu().numpy())
            predictions['tau_p'].append(torch.sigmoid(out['logit_tau']).cpu().numpy())
            predictions['tis_p'].append(torch.softmax(out['logit_tissue'], -1).cpu().numpy())

    # Concatenate predictions
    for k in ['expr_p', 'tau_p']:
        predictions[k] = np.concatenate(predictions[k])
    predictions['tis_p'] = np.concatenate(predictions['tis_p'], axis=0)

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0, help='Model fold to test (0-4)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-genes', type=int, default=100, help='Number of genes to test')
    parser.add_argument('--output', type=str, default='results/test_on_pchic_saikat')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Paths
    model_path = PROJECT_ROOT / f"results/model_fold{args.fold}.pt"
    pchic_dir = PROJECT_ROOT / "pchic_saikat" / "Gene_Annotated_PCHIC_Loops_Selected BEST TO USE"
    rna_seq = PROJECT_ROOT / "pchic_saikat" / "rnaseq_results" / "rnaseq.rsem_genes_tpm.txt"
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load checkpoint
    print(f"Loading model from {model_path}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    cfg = config_from_dict(checkpoint['config'])
    n_node_feat = checkpoint['n_node_feat']
    n_edge_feat = checkpoint['n_edge_feat']
    tissues = checkpoint['tissues']

    print(f"  Config: {cfg.genome_build}, window={cfg.promoter_window}, variant=gat (structure-only)")
    print(f"  Node features: {n_node_feat}, Edge features: {n_edge_feat}")
    print(f"  Tissues: {len(tissues)}")

    # Initialize model (structure-only variant for testing without sequence data)
    model = PangenomicGAT(cfg, n_node_feat, n_edge_feat, len(tissues), variant="gat")

    # Load state dict with strict=False to skip sequence tower weights
    missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f"  Loaded structure tower weights")
    print(f"  Skipped {len([k for k in unexpected if k.startswith('seq.')])} sequence tower params")
    print(f"  Skipped {len([k for k in unexpected if k.startswith('fuse.')])} fusion layer params")

    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")

    # Load data
    rna, loops, loop_files = load_pchic_data(pchic_dir, rna_seq, n_genes=args.n_genes)

    # Build graphs
    print("\nBuilding graphs...")
    graphs = []
    metadata = []

    for (gene, tissue), gene_loops in tqdm(loops.groupby(['gene', 'tissue']), desc="Building graphs"):
        graph = build_simple_graph(gene_loops, gene, tissue)
        if graph is not None:
            graphs.append(graph)
            metadata.append({'gene': gene, 'tissue': tissue})

    print(f"Built {len(graphs)} graphs from {len(metadata)} (gene, tissue) pairs")

    if len(graphs) == 0:
        print("ERROR: No graphs built. Check data paths and format.")
        return 1

    # Create data loader
    test_loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    # Run inference
    print("\nRunning inference...")
    predictions = run_inference(model, test_loader, device)

    # Save results
    results_file = output_dir / f'predictions_fold{args.fold}_pchic_saikat.npz'
    np.savez(results_file, **predictions)
    print(f"\nPredictions saved to: {results_file}")

    # Save metadata
    meta_df = pd.DataFrame(metadata)
    meta_df['expr_pred'] = predictions['expr_p']
    meta_df['tau_pred'] = predictions['tau_p']
    meta_df['tissue_pred_class'] = predictions['tis_p'].argmax(axis=1)
    meta_file = output_dir / f'metadata_fold{args.fold}.csv'
    meta_df.to_csv(meta_file, index=False)
    print(f"Metadata saved to: {meta_file}")

    print(f"\n{'='*60}")
    print("INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Graphs processed:     {len(graphs)}")
    print(f"Expression probs:     range [{predictions['expr_p'].min():.3f}, {predictions['expr_p'].max():.3f}]")
    print(f"Tau probs:            range [{predictions['tau_p'].min():.3f}, {predictions['tau_p'].max():.3f}]")
    print(f"Tissue predictions:   {len(np.unique(predictions['tis_p'].argmax(axis=1)))} unique classes")
    print(f"\nTesting complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
