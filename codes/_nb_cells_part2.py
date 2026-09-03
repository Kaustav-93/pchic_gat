# Cells 14-27: model -> training -> 5-fold CV -> full metric suite -> plots -> export
CELLS = []
def MD(s):   CELLS.append(("markdown", s.strip("\n")))
def CODE(s): CELLS.append(("code",     s.strip("\n")))

# =============================================================================
MD(r"""
## 14 · The model

This is panels 2–4 of the slide, in code.

### Tower A — sequence encoder: a HyenaDNA-style long convolution

Reimplemented from [HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna)
with no external dependency (the original pulls in einops, flash-attn and a
config registry). The maths is theirs; the plumbing is ours.

**The problem with the dilated CNN this replaces.** Four dilated conv layers
reach roughly 500 bp. Promoter-proximal regulatory signal routinely sits several
kilobases from the TSS, so the old tower was structurally incapable of seeing
most of what it was being asked about. Stacking more layers reaches further but
costs depth linearly in log(L) and blurs base-resolution detail through repeated
pooling.

**What a Hyena operator does instead.** One layer computes

```
y = out_proj( x_0 * FFTConv(x_1 * v, h) )
```

an `order`-step recurrence alternating element-wise gating with a long
convolution whose filter `h` is **as long as the input**. So the receptive field
of block 1 is the entire 16 kb window. Three properties follow:

1. **The filter is generated, not stored.** A small MLP maps a positional code
   to filter values (`HyenaFilter`), so parameter count is independent of
   sequence length — a 16 kb filter costs the same as a 1 kb one.
2. **`ExponentialModulation` gives channels different effective lengths.**
   Without it, every channel is free to average over the whole window and the
   filters collapse. With it, channels span a spectrum from a few hundred bp to
   the full window, which is exactly the range between a core-promoter motif and
   a distal element.
3. **Cost is O(L log L)**, via FFT, rather than O(L·k) for an explicit kernel or
   O(L²) for attention.

**Reverse-complement equivariance.** A promoter and its reverse complement are
the same physical locus, so we embed both strands with shared weights and
average. This halves the effective hypothesis space and removes an arbitrary
strand convention from the features. The model cell asserts that
`z(x) == z(rc(x))` — if that assertion ever fails, the features depend on which
strand the annotation happened to pick.

Modernised plumbing throughout: RMSNorm instead of BatchNorm, SwiGLU instead of
a plain MLP, pre-norm residuals.

### Tower B — structure encoder (`StructureEncoder`)
`L` × `GATv2Conv` with `K` heads and `edge_dim=N_EDGE_FEAT`, so the attention
logits see the CHiCAGO score and distance, not just node states. GATv2 rather
than the original GAT because GAT's attention is *static* — its ranking of
neighbours is the same for every query node, which is exactly the wrong
inductive bias when the whole question is "which partners matter for *this*
promoter". Each layer is followed by residual + `LayerNorm` + dropout. Layers
1..L−1 concatenate heads; layer L averages them.

Readout follows the slide: **gather the anchor node** (index 0 of each graph)
and concatenate it with an attention-pooled summary of all nodes, then project
to `z_str ∈ R^d`. **One vector per (gene, tissue).**

### Fusion — FiLM
`z(g,t) = γ(z_str) ⊙ z_seq + β(z_str)`. The tissue-varying structure embedding
*modulates* the tissue-invariant sequence embedding. The direction matters: this
way tissue identity gates promoter identity, rather than being a weak additive
nudge.

### Heads
- **H1** inner-product edge decoder over the node states `H_L` (not the fused
  graph vector — link prediction needs per-node representations).
- **H2**, **H3** MLPs on `z(g,t)` → logit.
- **H4** linear on `z(g,t)` → `T` logits.
- **Alignment** InfoNCE between `z_seq` and `z_str` within the batch.

Masked BCE is used for H2/H3 so that graphs with unknown labels (`-1`)
contribute zero gradient instead of being dropped from the batch.
""")

CODE(r"""
# =============================================================================
#  TOWER A - Hyena-style long-convolution sequence encoder
#
#  Reimplemented from the HyenaDNA operator (Nguyen et al. 2023,
#  github.com/HazyResearch/hyena-dna) with no external dependency: the original
#  needs einops, flash-attn and a config registry we do not want to drag in.
#  The maths is the same; the plumbing is ours.
# =============================================================================

def fftconv(u, k, D):
    # Circular-safe long convolution via FFT: O(L log L) instead of O(L * k).
    # u: (B, C, L)   k: (C, L)   D: (C,) skip-connection gain
    L = u.shape[-1]
    fft_size = 2 * L
    k_f = torch.fft.rfft(k.float(), n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.float(), n=fft_size)
    if u.dim() > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :L]
    return (y + u.float() * D.unsqueeze(-1)).to(u.dtype)


class Sin(nn.Module):
    # Periodic activation. A ReLU MLP cannot represent an oscillating filter
    # without enormous width; sine gives the implicit filter its frequency
    # content almost for free.
    def __init__(self, dim, w=10.0):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim))

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    # Complex-exponential position code, the input to the implicit filter MLP.
    def __init__(self, emb_dim, seq_len):
        super().__init__()
        assert emb_dim % 2 == 1 and emb_dim >= 3, "emb_dim must be odd and >= 3"
        t = torch.linspace(0, 1, seq_len)[None, :, None]              # (1, L, 1)
        bands = (emb_dim - 1) // 2
        t_res = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_res / seq_len
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        self.register_buffer("z", torch.cat([t, z.real, z.imag], dim=-1))
        self.register_buffer("t", t)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    # Multiplies the implicit filter by a learned per-channel exponential decay.
    # Without it every channel is free to attend uniformly over the whole
    # window and the filters collapse to a global average. With it, channels
    # span a spectrum of effective lengths - a few hundred bp to the full
    # window - which is what makes one layer able to model both a core-promoter
    # motif and a distal regulatory signal.
    def __init__(self, d_model, fast_decay_pct=0.3, slow_decay_pct=1.5,
                 target=1e-2, shift=0.0):
        super().__init__()
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        self.register_buffer("deltas",
                             torch.linspace(min_decay, max_decay, d_model)[None, None])

    def forward(self, t, x):
        return x * (torch.exp(-t * self.deltas.abs()) + self.shift)


class HyenaFilter(nn.Module):
    # The filter is not stored, it is GENERATED by a small MLP from the position
    # code. That is what makes the parameter count independent of sequence
    # length: a 16 kb filter and a 1 kb filter cost the same weights.
    def __init__(self, d_model, emb_dim=33, order=64, seq_len=1024,
                 n_inner=2, w=10.0, modulate=True):
        super().__init__()
        self.d_model, self.modulate = d_model, modulate
        self.bias = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len)
        act = Sin(dim=order, w=w)
        layers = [nn.Linear(emb_dim, order), act]
        for _ in range(n_inner):
            layers += [nn.Linear(order, order), Sin(dim=order, w=w)]
        layers += [nn.Linear(order, d_model, bias=False)]
        self.implicit_filter = nn.Sequential(*layers)
        self.modulation = ExponentialModulation(d_model)

    def filter(self, L):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        if self.modulate:
            h = self.modulation(t, h)
        return h                                   # (1, L, d_model)

    def forward(self, x, L, k=None, bias=None):
        if k is None:
            k = self.filter(L)
        k = k.squeeze(0).transpose(0, 1)           # (d_model, L)
        b = self.bias if bias is None else bias
        return fftconv(x, k, b)


class HyenaOperator(nn.Module):
    # y = out_proj( x_0 * FFTConv(x_1 * v, h) )
    #
    # An `order`-step recurrence of element-wise gating and long convolution.
    # The receptive field of ONE operator is the whole sequence, which is the
    # entire reason for using it: a dilated CNN needs log2(L) layers to see
    # 16 kb, and self-attention needs O(L^2) memory to do it.
    def __init__(self, d_model, l_max, order=2, filter_order=64,
                 short_filter_order=3, dropout=0.0, filter_emb_dim=33):
        super().__init__()
        assert order >= 2
        self.d_model, self.l_max, self.order = d_model, l_max, order
        total = d_model * (order + 1)
        self.in_proj = nn.Linear(d_model, total)
        self.out_proj = nn.Linear(d_model, d_model)
        self.short_filter = nn.Conv1d(total, total, short_filter_order,
                                      groups=total, padding=short_filter_order - 1)
        self.filter_fn = HyenaFilter(d_model * (order - 1), emb_dim=filter_emb_dim,
                                     order=filter_order, seq_len=l_max)
        self.drop = nn.Dropout(dropout)

    def forward(self, u):                          # u: (B, L, d_model)
        B, L, _ = u.shape
        lf = min(L, self.l_max)
        x = self.in_proj(u).transpose(1, 2)        # (B, total, L)
        x = self.short_filter(x)[..., :lf]         # local context, depthwise
        parts = x.split(self.d_model, dim=1)       # order+1 tensors of (B, d, L)
        *gates, v = parts

        k = self.filter_fn.filter(lf).squeeze(0).transpose(0, 1)   # (d*(order-1), L)
        k = k.view(self.order - 1, self.d_model, lf)
        bias = self.filter_fn.bias.view(self.order - 1, self.d_model)

        for o, g in enumerate(reversed(gates[1:])):
            v = self.drop(v * g)                   # element-wise gate
            v = fftconv(v, k[o], bias[o])          # global long convolution
        y = (v * gates[0]).transpose(1, 2)         # (B, L, d)
        return self.out_proj(y)


class RMSNorm(nn.Module):
    # Cheaper than LayerNorm and what current long-context stacks use.
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d, mult=2, dropout=0.0):
        super().__init__()
        h = int(d * mult)
        self.w1, self.w2, self.w3 = (nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class HyenaBlock(nn.Module):
    # Pre-norm residual, as in every modern sequence stack.
    def __init__(self, d, l_max, order, filter_order, dropout):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.mixer = HyenaOperator(d, l_max, order, filter_order, dropout=dropout)
        self.mlp = SwiGLU(d, 2, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.mixer(self.n1(x)))
        return x + self.mlp(self.n2(x))


class RMSNorm_1d(nn.Module):
    # RMSNorm over the channel dim of a (B, C, L) tensor, for use inside the stem
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, c, 1)); self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps)


class SequenceEncoder(nn.Module):
    # HyenaDNA-style promoter encoder: one-hot (B, 4, L) -> z_seq (B, d).
    #
    # Three deliberate choices:
    #
    # 1. Single-nucleotide stem, no pooling before the mixer. HyenaDNA's
    #    headline claim is that you do not need k-mer tokenisation; a stride-1
    #    stem keeps base-resolution information available to the long
    #    convolution. We optionally stride the stem when l_max is large purely
    #    to bound FFT cost, and the stride is reported.
    #
    # 2. Global receptive field from layer 1. Each HyenaOperator convolves with
    #    a filter as long as the input, so the first block already sees the
    #    whole promoter window. The dilated CNN this replaces needed log2(L)
    #    layers and still only reached ~500 bp.
    #
    # 3. Reverse-complement equivariance. A promoter and its reverse complement
    #    are the same physical locus. We embed both strands with shared weights
    #    and average, which halves the effective hypothesis space and removes an
    #    arbitrary strand convention from the features. Costs 2x forward
    #    compute; usually worth more than 2x the parameters.
    def __init__(self, d_model, l_max, n_blocks=4, order=2, filter_order=64,
                 stem_kernel=15, stem_stride=1, dropout=0.1, d_out=128,
                 rc_equivariant=True):
        super().__init__()
        self.rc_equivariant = rc_equivariant
        self.stem_stride = stem_stride
        l_eff = l_max // stem_stride
        self.l_eff = l_eff
        self.stem = nn.Sequential(
            nn.Conv1d(4, d_model, stem_kernel, stride=stem_stride,
                      padding=stem_kernel // 2),
            RMSNorm_1d(d_model), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            HyenaBlock(d_model, l_eff, order, filter_order, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = RMSNorm(d_model)
        self.pool_gate = nn.Linear(d_model, 1)
        self.proj = nn.Sequential(nn.Linear(2 * d_model, d_out), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_out, d_out))

    def _embed(self, x):                            # x: (B, 4, L)
        h = self.stem(x).transpose(1, 2)            # (B, L', d)
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        # mean pool ++ attention pool: the mean carries composition, the gate
        # lets a single strong motif dominate when it should
        a = torch.softmax(self.pool_gate(h), dim=1)
        return torch.cat([h.mean(1), (a * h).sum(1)], dim=-1)

    def forward(self, x):
        z = self._embed(x)
        if self.rc_equivariant:
            # reverse complement: flip length, and swap A<->T, C<->G
            xr = x.flip(-1)[:, [3, 2, 1, 0], :]
            z = 0.5 * (z + self._embed(xr))
        return self.proj(z)



class StructureEncoder(nn.Module):
    # promoter subgraph -> node states H_L (d_h each) and graph embedding z_str
    #
    # Width is held constant at d_h through the whole stack: layers 1..L-1 use
    # concat=True with d_h//heads channels per head (heads * d_h//heads == d_h),
    # and layer L averages its heads. That keeps every residual an identity and
    # removes the between-layer projections that a widening stack would need.
    def __init__(self, f_in, f_edge, d_hidden, d_out, layers, heads,
                 dropout, attn_dropout):
        super().__init__()
        if d_hidden % heads != 0:
            raise ValueError(f"d_hidden ({d_hidden}) must be divisible by "
                             f"gat_heads ({heads})")
        per_head = d_hidden // heads
        d_edge = max(8, d_hidden // 2)

        self.in_proj   = nn.Linear(f_in, d_hidden)
        self.edge_proj = nn.Sequential(nn.Linear(f_edge, d_edge), nn.GELU())

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(layers):
            last = (i == layers - 1)
            self.convs.append(GATv2Conv(
                d_hidden,
                d_hidden if last else per_head,
                heads=heads,
                concat=not last,
                dropout=attn_dropout,
                edge_dim=d_edge,
                add_self_loops=True,
            ))
            self.norms.append(nn.LayerNorm(d_hidden))

        self.drop = nn.Dropout(dropout)
        self.pool_gate = nn.Linear(d_hidden, 1)
        self.out = nn.Sequential(nn.Linear(2 * d_hidden, d_out), nn.GELU(),
                                 nn.Linear(d_out, d_out))

    def forward(self, x, edge_index, edge_attr, batch, anchor_index):
        h = self.in_proj(x)
        e = self.edge_proj(edge_attr)
        attn_last = None
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if i == len(self.convs) - 1:
                h_new, (ei, alpha) = conv(h, edge_index, e,
                                          return_attention_weights=True)
                attn_last = (ei.detach(), alpha.detach())
            else:
                h_new = conv(h, edge_index, e)
            h = self.drop(F.gelu(norm(h_new + h)))          # residual + norm

        # readout: anchor node  ++  attention-pooled summary.
        # pyg_softmax normalises the gate WITHIN each graph; a plain
        # torch.softmax would normalise across the whole batch, which silently
        # makes the pooled vector depend on batch composition.
        gate = pyg_softmax(self.pool_gate(h), batch)
        pooled = global_add_pool(gate * h, batch)
        anchor = h[anchor_index]
        return h, self.out(torch.cat([anchor, pooled], dim=-1)), attn_last


class FiLMFusion(nn.Module):
    # z(g,t) = gamma(z_str) * z_seq + beta(z_str)
    def __init__(self, d):
        super().__init__()
        self.gamma = nn.Linear(d, d)
        self.beta  = nn.Linear(d, d)
        self.norm  = nn.LayerNorm(d)

    def forward(self, z_seq, z_str):
        return self.norm(self.gamma(z_str) * z_seq + self.beta(z_str))


class PangenomicGAT(nn.Module):
    # variant selects which towers are live:
    #
    #   "both"   z = FiLM(z_seq, z_str)   full model
    #   "seq"    z = z_seq                sequence only, no graph
    #   "gat"    z = z_str                graph only, no sequence
    #
    # The ablation is a constructor flag rather than three separate classes so
    # that the heads, losses, splits and metric code are provably identical
    # across the three arms. Any difference in the numbers then comes from the
    # towers, which is the whole point of an ablation.
    VARIANTS = ("both", "seq", "gat")

    def __init__(self, cfg, f_node, f_edge, n_tissues, variant="both"):
        super().__init__()
        assert variant in self.VARIANTS, f"variant must be one of {self.VARIANTS}"
        self.variant = variant
        d = cfg.d_embed
        self.use_seq = variant in ("both", "seq")
        self.use_gat = variant in ("both", "gat")

        self.seq = SequenceEncoder(
            d_model=cfg.hyena_d_model, l_max=cfg.seq_len // cfg.stem_stride,
            n_blocks=cfg.hyena_blocks, order=cfg.hyena_order,
            filter_order=cfg.hyena_filter_order, stem_kernel=cfg.stem_kernel,
            stem_stride=cfg.stem_stride, dropout=cfg.dropout, d_out=d,
            rc_equivariant=cfg.rc_equivariant) if self.use_seq else None

        self.struc = StructureEncoder(f_node, f_edge, cfg.d_hidden, d,
                                      cfg.gat_layers, cfg.gat_heads,
                                      cfg.dropout, cfg.attn_dropout) \
            if self.use_gat else None

        self.fuse = FiLMFusion(d) if variant == "both" else None
        self.head_expr   = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(),
                                         nn.Dropout(cfg.dropout), nn.Linear(d // 2, 1))
        self.head_tau    = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(),
                                         nn.Dropout(cfg.dropout), nn.Linear(d // 2, 1))
        self.head_tissue = nn.Linear(d, n_tissues)
        d_h = cfg.d_hidden
        self.link_bilinear = nn.Parameter(torch.eye(d_h) + 0.01 * torch.randn(d_h, d_h))
        self.temperature = cfg.temperature

    def encode(self, batch, seq_bank):
        # batch.ptr[:-1] is the global index of node 0 of every graph, i.e. the
        # promoter anchor. For a single un-batched graph it degenerates to [0].
        dev = batch.x.device
        z_seq = z_str = h = attn = None

        if self.use_seq:
            seq = seq_bank[batch.seq_idx.to(seq_bank.device)].to(dev)
            z_seq = self.seq(seq)
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

    def forward(self, batch, seq_bank):
        z_seq, z_str, z, h, attn = self.encode(batch, seq_bank)
        return {
            "z_seq": z_seq, "z_str": z_str, "z": z, "h": h, "attn": attn,
            "logit_expr":   self.head_expr(z).squeeze(-1),
            "logit_tau":    self.head_tau(z).squeeze(-1),
            "logit_tissue": self.head_tissue(z),
        }

    def score_edges(self, h, pair_index):
        a = h[pair_index[0]]
        b = h[pair_index[1]]
        return ((a @ self.link_bilinear) * b).sum(-1)


def masked_bce(logit, target):
    # target == -1 marks an unknown label; those rows contribute nothing
    m = target >= 0
    if m.sum() == 0:
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit[m], target[m])


def info_nce(z_a, z_b, temperature):
    a = F.normalize(z_a, dim=-1)
    b = F.normalize(z_b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.t(), labels))


rows = []
for v in PangenomicGAT.VARIANTS:
    m = PangenomicGAT(CFG, N_NODE_FEAT, N_EDGE_FEAT, len(TISSUE_LIST), variant=v)
    tot = sum(p.numel() for p in m.parameters())
    nseq = sum(p.numel() for p in m.seq.parameters()) if m.seq else 0
    nstr = sum(p.numel() for p in m.struc.parameters()) if m.struc else 0
    rows.append({"variant": v, "sequence_tower": nseq, "structure_tower": nstr,
                 "fusion+heads": tot - nseq - nstr, "total": tot})
    if v == "both":
        probe = m
print("parameter budget per ablation arm")
print(pd.DataFrame(rows).to_string(index=False))

L_eff = CFG.seq_len // CFG.stem_stride
print()
print("  TOWER A  HyenaDNA-style long convolution")
print(f"    promoter window   {CFG.seq_len:,} bp  "
      f"(stem stride {CFG.stem_stride} -> {L_eff:,} positions)")
print(f"    blocks            {CFG.hyena_blocks} x HyenaOperator(order={CFG.hyena_order})")
print(f"    d_model           {CFG.hyena_d_model}")
print(f"    implicit filter   MLP width {CFG.hyena_filter_order}, sine activations")
print(f"    receptive field   {L_eff:,} positions from block 1 "
      f"(filter length == input length)")
print(f"    rev-comp equiv.   {CFG.rc_equivariant}")
print("  TOWER B  GATv2")
print(f"    layers x heads    {CFG.gat_layers} x {CFG.gat_heads}, d_h={CFG.d_hidden}")
print(f"    node / edge feat  F={N_NODE_FEAT}  F_e={N_EDGE_FEAT}")
print("  SHARED")
print(f"    embedding dim     d={CFG.d_embed}")
print(f"    tissue classes    {len(TISSUE_LIST)}")

# ---- shape + gradient smoke test before we spend hours training -------------
_x = torch.randn(2, 4, CFG.seq_len)
_z = probe.seq(_x)
print(f"\nsmoke test: SequenceEncoder {tuple(_x.shape)} -> {tuple(_z.shape)}")
assert _z.shape == (2, CFG.d_embed), "sequence tower output shape is wrong"

# reverse-complement equivariance must actually hold
if CFG.rc_equivariant:
    probe.seq.eval()
    with torch.no_grad():
        z_f = probe.seq(_x)
        z_r = probe.seq(_x.flip(-1)[:, [3, 2, 1, 0], :])
    dev = (z_f - z_r).abs().max().item()
    print(f"  reverse-complement equivariance: max |z(x) - z(rc(x))| = {dev:.2e}")
    assert dev < 1e-4, "RC averaging is not making the encoder strand-symmetric"
    probe.seq.train()

_z.sum().backward()
g = [p.grad.abs().sum().item() for p in probe.seq.parameters() if p.grad is not None]
print(f"  backward OK: {len(g)} tensors received gradient, "
      f"total |grad| = {sum(g):.3e}")
assert sum(g) > 0, "no gradient reached the sequence tower"
del probe, _x, _z
""")

# =============================================================================
MD(r"""
## 15 · Training and evaluation loops

A single `run_epoch` handles both phases; the only differences are
`model.train()/eval()` and whether we step the optimiser.

**Anchor indexing.** PyG's `DataLoader` concatenates graphs into one big
disconnected graph and gives us `batch.ptr`, the node offset of each graph.
`batch.ptr[:-1]` is therefore exactly the global index of every graph's node 0 —
its promoter anchor. The same offsets shift the held-out edge pairs into the
batched coordinate frame.

**Loss.** Weighted sum of the five terms from cell 1. H2 and H3 use masked BCE;
H4 uses cross-entropy; H1 uses BCE over the concatenated positive and negative
edge scores; alignment uses InfoNCE between the two embedding spaces.

**Early stopping** on validation loss with `patience` epochs, restoring the best
weights. Within each CV fold we carve a 10% validation slice out of the training
graphs — *also grouped by gene* — so the stopping decision never touches the
test fold.
""")

CODE(r"""
# ---------------------------------------------------------------------------
# A note on batched edge-pair indices.
#
# PyG's Data.__inc__ returns num_nodes for any attribute whose *name contains
# "index"*, and __cat_dim__ returns -1 for those. Because we named our held-out
# pairs `pos_edge_label_index` / `neg_edge_label_index`, the DataLoader already
# offsets them into the batched coordinate frame and concatenates them on dim 1.
# So we can use batch.pos_edge_label_index directly -- adding batch.ptr
# ourselves would double-shift and silently score the wrong node pairs.
#
# `seq_idx` deliberately does NOT contain "index", so it is concatenated on
# dim 0 with no offset, giving one gene index per graph.
# ---------------------------------------------------------------------------

def run_epoch(model, loader, seq_bank, cfg, optim=None, device="cpu"):
    train = optim is not None
    model.train() if train else model.eval()
    tot = {k: 0.0 for k in ("loss", "link", "expr", "tau", "tissue", "align")}
    n_batches = 0
    store = {"expr_p": [], "expr_y": [], "tau_p": [], "tau_y": [],
             "tis_p": [], "tis_y": [], "link_p": [], "link_y": [],
             "z": [], "z_seq": [], "z_str": [], "idx": []}

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            z_seq, z_str, z, h, attn = model.encode(batch, seq_bank)

            logit_expr   = model.head_expr(z).squeeze(-1)
            logit_tau    = model.head_tau(z).squeeze(-1)
            logit_tissue = model.head_tissue(z)

            pos = getattr(batch, "pos_edge_label_index", None)
            neg = getattr(batch, "neg_edge_label_index", None)

            # H1 scores node pairs, so it is only defined when the graph tower
            # is present. The seq-only arm therefore reports no link metrics --
            # stated rather than silently zero-filled.
            l_link = torch.tensor(0.0, device=device)
            if (h is not None and pos is not None and neg is not None
                    and pos.numel() and neg.numel()):
                s_pos = model.score_edges(h, pos)
                s_neg = model.score_edges(h, neg)
                s = torch.cat([s_pos, s_neg])
                y = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)])
                l_link = F.binary_cross_entropy_with_logits(s, y)
                store["link_p"].append(torch.sigmoid(s).detach().cpu())
                store["link_y"].append(y.detach().cpu())

            l_expr   = masked_bce(logit_expr, batch.y_expr)
            l_tau    = masked_bce(logit_tau,  batch.y_tau)
            l_tissue = F.cross_entropy(logit_tissue, batch.y_tissue)
            l_align = (info_nce(z_seq, z_str, cfg.temperature)
                       if (z_seq is not None and z_str is not None and len(z_seq) > 1)
                       else torch.tensor(0.0, device=device))

            loss = (cfg.w_link * l_link + cfg.w_expr * l_expr + cfg.w_tau * l_tau
                    + cfg.w_tissue * l_tissue + cfg.w_align * l_align)

            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()

            for k, v in zip(("loss", "link", "expr", "tau", "tissue", "align"),
                            (loss, l_link, l_expr, l_tau, l_tissue, l_align)):
                tot[k] += float(v)
            n_batches += 1

            if not train:
                store["expr_p"].append(torch.sigmoid(logit_expr).cpu())
                store["expr_y"].append(batch.y_expr.cpu())
                store["tau_p"].append(torch.sigmoid(logit_tau).cpu())
                store["tau_y"].append(batch.y_tau.cpu())
                store["tis_p"].append(torch.softmax(logit_tissue, -1).cpu())
                store["tis_y"].append(batch.y_tissue.cpu())
                store["z"].append(z.cpu())
                if z_seq is not None: store["z_seq"].append(z_seq.cpu())
                if z_str is not None: store["z_str"].append(z_str.cpu())

    avg = {k: v / max(n_batches, 1) for k, v in tot.items()}
    cat = {k: (torch.cat(v).numpy() if v else np.array([])) for k, v in store.items()
           if k != "idx"}
    return avg, cat


def make_loader(graphs, indices, bs, shuffle):
    return DataLoader([graphs[i] for i in indices], batch_size=bs,
                      shuffle=shuffle, num_workers=CFG.num_workers)


# ---- verify the batching assumption rather than trusting it -----------------
_probe = DataLoader([GRAPHS[0], GRAPHS[1]], batch_size=2, shuffle=False)
_b = next(iter(_probe))
_n0 = int(GRAPHS[0].num_nodes)
_expected = int(GRAPHS[1].pos_edge_label_index[0, 0]) + _n0
_got = int(_b.pos_edge_label_index[0, GRAPHS[0].pos_edge_label_index.shape[1]])
print("batching self-check")
print(f"  graph 0 has {_n0} nodes; graph 1's first held-out pair should be "
      f"offset by {_n0}")
print(f"  expected {_expected}, got {_got}")
assert _got == _expected, (
    "PyG did not offset pos_edge_label_index as expected. Rename the attribute "
    "so it contains 'index', or offset manually by batch.ptr.")
assert int(_b.ptr[1]) == _n0, "batch.ptr does not mark graph boundaries"
assert _b.seq_idx.shape[0] == 2, "seq_idx should be one entry per graph"
print("  OK: edge pairs offset correctly, ptr marks anchors, seq_idx per graph")

print("\ntraining utilities defined")
print("  run_epoch(...) -> (mean losses, stacked predictions)")
print("  anchors  = batch.ptr[:-1]  (node 0 of every graph)")
print("  H2/H3 use masked BCE, so y = -1 rows contribute no gradient")
""")

# =============================================================================
MD(r"""
## 16 · The metric suite

One function, every metric, computed identically for every head so the results
table is comparable across tasks.

**Binary heads (H1, H2, H3)** — threshold-free first, then thresholded:

- *Ranking*: AUROC, AUPRC (= average precision). AUPRC is the one to trust when
  classes are imbalanced; AUROC flatters a model on skewed data because the
  huge negative pool makes the false-positive rate look small.
- *At threshold 0.5 and at the F1-optimal threshold*: accuracy, balanced
  accuracy, precision, recall (sensitivity), specificity, NPV, F1, F2, MCC,
  Cohen's kappa, Jaccard.
- *Probability quality*: Brier score and log loss. A model can have a great
  AUROC and useless probabilities; these two catch that.
- The **baseline AUPRC** (= positive prevalence) is reported alongside, because
  an AUPRC of 0.4 is excellent at 5% prevalence and terrible at 60%.

**Multiclass head (H4)** — accuracy, balanced accuracy, macro/micro/weighted
precision/recall/F1, macro one-vs-rest AUROC and AUPRC, and Cohen's kappa.

MCC is included deliberately: it is the metric that stays honest under class
imbalance, where accuracy and even F1 can be gamed by predicting the majority.
""")

CODE(r"""
def binary_metrics(y_true, y_prob, prefix=""):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    m = y_true >= 0                                     # drop unknown labels
    y_true, y_prob = y_true[m].astype(int), y_prob[m]
    out = {f"{prefix}n": len(y_true), f"{prefix}pos_rate": float(y_true.mean())
           if len(y_true) else np.nan}
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {**out, **{f"{prefix}{k}": np.nan for k in
                          ("auroc", "auprc", "auprc_baseline", "accuracy",
                           "balanced_accuracy", "precision", "recall",
                           "specificity", "npv", "f1", "f2", "mcc", "kappa",
                           "jaccard", "brier", "logloss", "best_threshold",
                           "f1_at_best", "precision_at_best", "recall_at_best")}}

    out[f"{prefix}auroc"] = roc_auc_score(y_true, y_prob)
    out[f"{prefix}auprc"] = average_precision_score(y_true, y_prob)
    out[f"{prefix}auprc_baseline"] = float(y_true.mean())

    def at(th):
        yp = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
        pre, rec, f1, _ = precision_recall_fscore_support(
            y_true, yp, average="binary", zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        npv  = tn / (tn + fn) if (tn + fn) else np.nan
        f2 = (5 * pre * rec / (4 * pre + rec)) if (4 * pre + rec) else 0.0
        return dict(accuracy=accuracy_score(y_true, yp),
                    balanced_accuracy=balanced_accuracy_score(y_true, yp),
                    precision=pre, recall=rec, specificity=spec, npv=npv,
                    f1=f1, f2=f2, mcc=matthews_corrcoef(y_true, yp),
                    kappa=cohen_kappa_score(y_true, yp),
                    jaccard=jaccard_score(y_true, yp, zero_division=0),
                    tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

    for k, v in at(0.5).items():
        out[f"{prefix}{k}"] = v

    pr, rc, th = precision_recall_curve(y_true, y_prob)
    f1s = np.divide(2 * pr * rc, pr + rc, out=np.zeros_like(pr), where=(pr + rc) > 0)
    bi = int(np.nanargmax(f1s[:-1])) if len(th) else 0
    best_th = float(th[bi]) if len(th) else 0.5
    out[f"{prefix}best_threshold"]    = best_th
    out[f"{prefix}f1_at_best"]        = float(f1s[bi])
    out[f"{prefix}precision_at_best"] = float(pr[bi])
    out[f"{prefix}recall_at_best"]    = float(rc[bi])

    out[f"{prefix}brier"]   = brier_score_loss(y_true, y_prob)
    out[f"{prefix}logloss"] = log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7),
                                       labels=[0, 1])
    return out


def multiclass_metrics(y_true, y_prob, n_classes, prefix="tissue_"):
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0:
        return {}
    y_pred = y_prob.argmax(1)
    present = np.unique(y_true)
    out = {
        f"{prefix}n": len(y_true),
        f"{prefix}n_classes_present": len(present),
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}kappa": cohen_kappa_score(y_true, y_pred),
    }
    for avg in ("macro", "micro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0)
        out[f"{prefix}precision_{avg}"] = p
        out[f"{prefix}recall_{avg}"] = r
        out[f"{prefix}f1_{avg}"] = f
    try:
        out[f"{prefix}auroc_macro_ovr"] = roc_auc_score(
            y_true, y_prob[:, present], multi_class="ovr",
            average="macro", labels=present)
    except Exception:
        out[f"{prefix}auroc_macro_ovr"] = np.nan
    aps = []
    for c in present:
        try:
            aps.append(average_precision_score((y_true == c).astype(int), y_prob[:, c]))
        except Exception:
            pass
    out[f"{prefix}auprc_macro_ovr"] = float(np.mean(aps)) if aps else np.nan
    return out


def all_metrics(pred, n_classes):
    m = {}
    m.update(binary_metrics(pred["link_y"], pred["link_p"], "link_"))
    m.update(binary_metrics(pred["expr_y"], pred["expr_p"], "expr_"))
    m.update(binary_metrics(pred["tau_y"],  pred["tau_p"],  "tau_"))
    m.update(multiclass_metrics(pred["tis_y"], pred["tis_p"], n_classes))
    return m


# ---- self-test on synthetic data so a silent metric bug shows up here -------
_rng = np.random.default_rng(0)
_y = _rng.integers(0, 2, 800)
_p = np.clip(_y * 0.6 + _rng.normal(0.2, 0.25, 800), 0, 1)
_t = binary_metrics(_y, _p, "")
print("metric self-test on synthetic data (should be well above chance):")
for k in ("n", "pos_rate", "auroc", "auprc", "auprc_baseline", "accuracy",
          "precision", "recall", "specificity", "f1", "mcc", "kappa",
          "brier", "logloss", "best_threshold", "f1_at_best"):
    print(f"  {k:<18} {_t[k]:.4f}")
assert _t["auroc"] > 0.7 and 0 <= _t["brier"] <= 1
print("\n  OK: metric suite behaves sensibly")
""")

# =============================================================================
MD(r"""
## 17 · 5-fold cross-validation

For each of the five folds:

1. Split the training graphs again into 90/10 train/validation, **grouped by
   gene** so the early-stopping signal is not contaminated.
2. Fresh model, AdamW, cosine LR schedule.
3. Train for up to `n_epochs` with early stopping on validation loss.
4. Restore the best weights, score the untouched test fold, store predictions.

Per-fold predictions are kept in full so every curve later is computed from raw
scores rather than from averaged summary numbers — averaging AUCs and averaging
curves are different operations and only the latter lets us show variance
bands.

The loop prints a compact per-epoch line so you can watch the individual heads
converge. If the tissue head hits ~1.0 within two epochs while expression is
still moving, that is the saturation problem flagged earlier: the graph gives
the tissue away, and `w_tissue` should probably be reduced.
""")

CODE(r"""
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_fold(fold, train_idx, test_idx, graphs, seq_bank, cfg, device,
                   variant="both", quiet=False):
    # inner validation split, still grouped by gene
    g_tr = META.iloc[train_idx]["gene"].values
    inner = GroupKFold(n_splits=10)
    sub_tr, sub_va = next(inner.split(np.arange(len(train_idx)), groups=g_tr))
    tr_idx = train_idx[sub_tr]; va_idx = train_idx[sub_va]

    model = PangenomicGAT(cfg, N_NODE_FEAT, N_EDGE_FEAT, len(TISSUE_LIST),
                          variant=variant).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=cfg.n_epochs)

    dl_tr = make_loader(graphs, tr_idx, cfg.batch_size, True)
    dl_va = make_loader(graphs, va_idx, cfg.batch_size, False)
    dl_te = make_loader(graphs, test_idx, cfg.batch_size, False)

    best = {"loss": np.inf, "state": None, "epoch": -1}
    history = []
    bad = 0
    for ep in range(cfg.n_epochs):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, dl_tr, seq_bank, cfg, optim, device)
        va_loss, _ = run_epoch(model, dl_va, seq_bank, cfg, None, device)
        sched.step()
        history.append({"variant": variant, "fold": fold, "epoch": ep,
                        **{f"train_{k}": v for k, v in tr_loss.items()},
                        **{f"val_{k}": v for k, v in va_loss.items()},
                        "lr": sched.get_last_lr()[0], "sec": time.time() - t0})
        flag = ""
        if va_loss["loss"] < best["loss"] - 1e-4:
            best = {"loss": va_loss["loss"],
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()},
                    "epoch": ep}
            bad = 0; flag = "  <- best"
        else:
            bad += 1
        if not quiet:
            print(f"  f{fold} ep{ep:>3}  train {tr_loss['loss']:.4f}  "
              f"val {va_loss['loss']:.4f}  "
              f"[link {va_loss['link']:.3f} expr {va_loss['expr']:.3f} "
              f"tau {va_loss['tau']:.3f} tis {va_loss['tissue']:.3f} "
                  f"align {va_loss['align']:.3f}]  {time.time()-t0:.0f}s{flag}")
        if bad >= cfg.patience:
            print(f"  f{fold} early stop at epoch {ep} "
                  f"(best was {best['epoch']})")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    _, pred = run_epoch(model, dl_te, seq_bank, cfg, None, device)
    return model, pred, pd.DataFrame(history), best["epoch"]


SEQ_BANK = SEQ_TENSOR                      # (n_genes, 4, L) on CPU; sliced per batch

VARIANT_LABEL = {"both": "GAT + sequence", "gat": "GAT only", "seq": "sequence only"}
RESULTS = {}                               # variant -> dict(preds, metrics, hist, models)

t_start = time.time()
for variant in CFG.variants:
    print(f"\n\n{'#'*74}\n#  ARM: {variant.upper():<10} ({VARIANT_LABEL[variant]})\n{'#'*74}")
    v_preds, v_metrics, v_hist, v_models = [], [], [], []
    for k, (tr, te) in enumerate(FOLDS):
        print(f"\n{'='*74}\n{variant} | FOLD {k+1}/{CFG.n_folds}   "
              f"train={len(tr):,}  test={len(te):,}\n{'='*74}")
        model, pred, hist, best_ep = train_one_fold(
            k, tr, te, GRAPHS, SEQ_BANK, CFG, DEVICE, variant=variant)
        met = all_metrics(pred, len(TISSUE_LIST))
        met["fold"] = k; met["best_epoch"] = best_ep; met["variant"] = variant
        v_preds.append(pred); v_metrics.append(met)
        v_hist.append(hist); v_models.append(model)
        print(f"\n  {variant} fold {k} test:  "
              f"link AUPRC {met.get('link_auprc', np.nan):.3f} | "
              f"expr AUPRC {met.get('expr_auprc', np.nan):.3f} | "
              f"tau AUPRC {met.get('tau_auprc', np.nan):.3f} | "
              f"tissue acc {met.get('tissue_accuracy', np.nan):.3f}")
    RESULTS[variant] = {"preds": v_preds, "metrics": v_metrics,
                        "hist": pd.concat(v_hist, ignore_index=True),
                        "models": v_models}

print(f"\n{'='*74}\nall {len(CFG.variants)} arms x {CFG.n_folds} folds done in "
      f"{(time.time()-t_start)/60:.1f} min")

# the detailed downstream cells report on the full model
PRIMARY = "both" if "both" in RESULTS else CFG.variants[0]
fold_preds  = RESULTS[PRIMARY]["preds"]
fold_metrics = RESULTS[PRIMARY]["metrics"]
fold_models = RESULTS[PRIMARY]["models"]
HIST = RESULTS[PRIMARY]["hist"]
print(f"cells below report the '{PRIMARY}' arm ({VARIANT_LABEL[PRIMARY]}); "
      f"the ablation cells compare all {len(RESULTS)}")
""")

# =============================================================================
MD(r"""
## 18 · Results table

Mean ± standard deviation across the five folds, for every metric and every
head. The standard deviation is the number to look at hardest: a head with high
mean AUPRC and a large spread across folds has probably latched onto something
fold-specific, and the mean alone is misleading.

Read the binary heads against their **AUPRC baseline** column. A head whose
AUPRC barely exceeds its baseline has learned nothing, no matter how good its
AUROC looks.
""")

CODE(r"""
MET = pd.DataFrame(fold_metrics).drop(columns=["variant"]).set_index("fold")
MET.to_csv(RESULTS_DIR / "metrics_per_fold.csv")

def summarise(prefix, keys):
    rows = []
    for k in keys:
        col = f"{prefix}{k}"
        if col in MET.columns:
            v = MET[col].astype(float)
            rows.append({"metric": k, "mean": v.mean(), "std": v.std(),
                         "min": v.min(), "max": v.max()})
    return pd.DataFrame(rows).set_index("metric")

BIN_KEYS = ["n", "pos_rate", "auroc", "auprc", "auprc_baseline",
            "accuracy", "balanced_accuracy", "precision", "recall",
            "specificity", "npv", "f1", "f2", "mcc", "kappa", "jaccard",
            "brier", "logloss", "best_threshold", "f1_at_best",
            "precision_at_best", "recall_at_best"]
MC_KEYS = ["n", "n_classes_present", "accuracy", "balanced_accuracy", "kappa",
           "precision_macro", "recall_macro", "f1_macro",
           "precision_micro", "recall_micro", "f1_micro",
           "precision_weighted", "recall_weighted", "f1_weighted",
           "auroc_macro_ovr", "auprc_macro_ovr"]

HEADS = [("H1  link prediction",          "link_",   BIN_KEYS),
         ("H2  expressed in tissue",      "expr_",   BIN_KEYS),
         ("H3  tissue-restricted (tau)",  "tau_",    BIN_KEYS),
         ("H4  tissue classification",    "tissue_", MC_KEYS)]

SUMMARIES = {}
for title, pref, keys in HEADS:
    s = summarise(pref, keys)
    SUMMARIES[pref] = s
    print(f"\n{'='*74}\n{title}   (mean +/- sd over {CFG.n_folds} folds)\n{'='*74}")
    print(s.round(4).to_string())
    s.to_csv(RESULTS_DIR / f"metrics_summary_{pref.rstrip('_')}.csv")

print(f"\n\n{'='*74}\nHEADLINE\n{'='*74}")
head_rows = []
for title, pref, _ in HEADS:
    if pref == "tissue_":
        head_rows.append([title,
                          f"{MET['tissue_accuracy'].mean():.3f} +/- {MET['tissue_accuracy'].std():.3f}",
                          f"{MET['tissue_f1_macro'].mean():.3f}",
                          f"{MET['tissue_auroc_macro_ovr'].mean():.3f}",
                          f"{MET['tissue_auprc_macro_ovr'].mean():.3f}", "-"])
    else:
        head_rows.append([title,
                          f"{MET[pref+'accuracy'].mean():.3f} +/- {MET[pref+'accuracy'].std():.3f}",
                          f"{MET[pref+'f1'].mean():.3f}",
                          f"{MET[pref+'auroc'].mean():.3f}",
                          f"{MET[pref+'auprc'].mean():.3f}",
                          f"{MET[pref+'auprc_baseline'].mean():.3f}"])
print(pd.DataFrame(head_rows, columns=["head", "accuracy", "F1", "AUROC",
                                       "AUPRC", "AUPRC baseline"]
                   ).to_string(index=False))
print(f"\nper-fold metrics -> {(RESULTS_DIR/'metrics_per_fold.csv').name}")
""")

# =============================================================================
MD(r"""
## 19 · ROC curves

One panel per binary head. Each fold is drawn as a thin line, and the mean
curve is computed by interpolating every fold onto a common false-positive-rate
grid and averaging the true-positive rates, with a ±1 sd band.

Interpolating then averaging is the right way to do this. Averaging the five
AUC *numbers* and drawing one curve through the pooled predictions are both
common and both hide fold-to-fold variance — which for a biological dataset
split by gene is exactly the quantity you want to see.
""")

CODE(r"""
def roc_panel(ax, key, title):
    grid = np.linspace(0, 1, 200)
    tprs, aucs = [], []
    for k, pred in enumerate(fold_preds):
        y, p = pred[f"{key}_y"], pred[f"{key}_p"]
        m = y >= 0
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[m], p[m])
        auc = roc_auc_score(y[m], p[m]); aucs.append(auc)
        tprs.append(np.interp(grid, fpr, tpr)); tprs[-1][0] = 0.0
        ax.plot(fpr, tpr, lw=.9, alpha=.45, color=PALETTE[k % len(PALETTE)],
                label=f"fold {k} (AUC {auc:.3f})")
    if tprs:
        mt = np.mean(tprs, 0); st = np.std(tprs, 0); mt[-1] = 1.0
        ax.plot(grid, mt, lw=2.4, color="k",
                label=f"mean (AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f})")
        ax.fill_between(grid, np.clip(mt-st, 0, 1), np.clip(mt+st, 0, 1),
                        color="k", alpha=.13, label="+/- 1 sd")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="chance")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(title); ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
roc_panel(axes[0], "link", "H1  link prediction")
roc_panel(axes[1], "expr", "H2  expressed in tissue")
roc_panel(axes[2], "tau",  "H3  tissue-restricted")
fig.suptitle("ROC, 5-fold cross-validation (gene-grouped splits)", y=1.02, fontsize=13)
fig.tight_layout(); savefig(fig, "07_roc_curves"); plt.show()
""")

# =============================================================================
MD(r"""
## 20 · Precision–recall curves

The same three heads, but on the axes that actually matter for imbalanced data.

Each panel carries a dashed horizontal line at the **positive prevalence** —
the AUPRC a random classifier achieves. The distance between the curve and that
line is the real signal. For the expression head, where roughly half of
gene×tissue pairs are positive, the baseline sits near 0.5 and a 0.6 AUPRC is
weak; for a head with 5% positives, 0.4 would be strong.
""")

CODE(r"""
def pr_panel(ax, key, title):
    grid = np.linspace(0, 1, 200)
    precs, aps, bases = [], [], []
    for k, pred in enumerate(fold_preds):
        y, p = pred[f"{key}_y"], pred[f"{key}_p"]
        m = y >= 0
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            continue
        pr, rc, _ = precision_recall_curve(y[m], p[m])
        ap = average_precision_score(y[m], p[m]); aps.append(ap)
        bases.append(float(np.mean(y[m])))
        ax.plot(rc, pr, lw=.9, alpha=.45, color=PALETTE[k % len(PALETTE)],
                label=f"fold {k} (AP {ap:.3f})")
        order = np.argsort(rc)
        precs.append(np.interp(grid, rc[order], pr[order]))
    if precs:
        mp = np.mean(precs, 0); sp = np.std(precs, 0)
        ax.plot(grid, mp, lw=2.4, color="k",
                label=f"mean (AP {np.mean(aps):.3f} +/- {np.std(aps):.3f})")
        ax.fill_between(grid, np.clip(mp-sp, 0, 1), np.clip(mp+sp, 0, 1),
                        color="k", alpha=.13, label="+/- 1 sd")
        b = float(np.mean(bases))
        ax.axhline(b, ls="--", c="r", lw=1.2, label=f"baseline = {b:.3f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(title); ax.legend(fontsize=7, loc="lower left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
pr_panel(axes[0], "link", "H1  link prediction")
pr_panel(axes[1], "expr", "H2  expressed in tissue")
pr_panel(axes[2], "tau",  "H3  tissue-restricted")
fig.suptitle("Precision-recall, 5-fold CV  (dashed red = random baseline)",
             y=1.02, fontsize=13)
fig.tight_layout(); savefig(fig, "08_pr_curves"); plt.show()

print("read each curve against its own red baseline, not against the other panels")
""")

# =============================================================================
MD(r"""
## 21 · Confusion matrices and threshold choice

Left three panels: pooled confusion matrices for the binary heads at the
**F1-optimal threshold** rather than at 0.5. With imbalanced data the 0.5 cut is
arbitrary, and reporting only 0.5 numbers understates a model that ranks well
but is poorly calibrated.

Right: the tissue-classification confusion matrix, row-normalised. This is worth
staring at — confusions here are informative rather than embarrassing. Related
biosamples (the various CD4+ subsets, for instance) *should* be confusable; if
they are perfectly separated, the model has probably found a batch signature
rather than biology.

Below, the F1-vs-threshold curve shows how sharply performance depends on where
the cut is placed.
""")

CODE(r"""
def pool(key):
    y = np.concatenate([p[f"{key}_y"] for p in fold_preds])
    s = np.concatenate([p[f"{key}_p"] for p in fold_preds])
    m = y >= 0
    return y[m].astype(int), s[m]

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])

for i, (key, name) in enumerate([("link", "H1 link"), ("expr", "H2 expressed"),
                                 ("tau", "H3 restricted")]):
    ax = fig.add_subplot(gs[0, i])
    y, s = pool(key)
    if len(np.unique(y)) < 2:
        ax.text(.5, .5, "single class", ha="center"); ax.axis("off"); continue
    pr, rc, th = precision_recall_curve(y, s)
    f1 = np.divide(2*pr*rc, pr+rc, out=np.zeros_like(pr), where=(pr+rc) > 0)
    bt = float(th[int(np.nanargmax(f1[:-1]))]) if len(th) else .5
    cm = confusion_matrix(y, (s >= bt).astype(int), labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"])
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f"{name}\nthreshold={bt:.3f}   F1={2*tp/(2*tp+fp+fn):.3f}", fontsize=10)

ax = fig.add_subplot(gs[0, 3])
for i, (key, name) in enumerate([("link", "H1"), ("expr", "H2"), ("tau", "H3")]):
    y, s = pool(key)
    if len(np.unique(y)) < 2:
        continue
    pr, rc, th = precision_recall_curve(y, s)
    f1 = np.divide(2*pr*rc, pr+rc, out=np.zeros_like(pr), where=(pr+rc) > 0)
    if len(th):
        ax.plot(th, f1[:-1], lw=1.6, color=PALETTE[i], label=name)
        ax.axvline(th[int(np.nanargmax(f1[:-1]))], ls=":", lw=1, color=PALETTE[i])
ax.axvline(.5, ls="--", c="k", lw=1, label="default 0.5")
ax.set_xlabel("threshold"); ax.set_ylabel("F1"); ax.set_title("F1 vs threshold")
ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, :])
yt = np.concatenate([p["tis_y"] for p in fold_preds]).astype(int)
yp = np.concatenate([p["tis_p"] for p in fold_preds]).argmax(1)
present = sorted(set(yt) | set(yp))
cm = confusion_matrix(yt, yp, labels=present).astype(float)
cm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
sns.heatmap(cm, cmap="magma", vmin=0, vmax=1, ax=ax,
            xticklabels=[TISSUE_LIST[i] for i in present],
            yticklabels=[TISSUE_LIST[i] for i in present],
            cbar_kws={"label": "row-normalised fraction"})
ax.set_title(f"H4 tissue classification, row-normalised  "
             f"(accuracy {accuracy_score(yt, yp):.3f})")
ax.tick_params(axis="x", rotation=90, labelsize=7)
ax.tick_params(axis="y", rotation=0, labelsize=7)

fig.tight_layout(); savefig(fig, "09_confusion_matrices"); plt.show()

print("\nper-class report for H4 (pooled over folds):")
print(classification_report(yt, yp, labels=present,
                            target_names=[TISSUE_LIST[i] for i in present],
                            zero_division=0))
""")

# =============================================================================
MD(r"""
## 22 · Calibration

Ranking and calibration are different properties. A model can order examples
perfectly (AUROC 1.0) while its probabilities are systematically wrong, which
matters the moment anyone thresholds the output or uses it as a prior.

The reliability diagram bins predictions by confidence and plots observed
frequency against predicted probability; the diagonal is perfect calibration.
Below the diagonal means overconfident. The Brier score and the **expected
calibration error** quantify the gap.

Multi-task training tends to produce mild overconfidence, because the shared
trunk is optimised for four objectives at once and no single head's probability
scale is directly regularised. If a head is badly miscalibrated, Platt scaling
or temperature scaling fitted on the validation split is the standard fix — and
it must be fitted on validation, never on test.
""")

CODE(r"""
def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return e

fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for i, (key, name) in enumerate([("link", "H1 link"), ("expr", "H2 expressed"),
                                 ("tau", "H3 restricted")]):
    ax = axes[i]
    y, p = pool(key)
    if len(np.unique(y)) < 2:
        ax.text(.5, .5, "single class", ha="center"); ax.axis("off"); continue
    for k, pred in enumerate(fold_preds):
        yy, pp = pred[f"{key}_y"], pred[f"{key}_p"]
        m = yy >= 0
        if m.sum() < 50 or len(np.unique(yy[m])) < 2:
            continue
        try:
            ft, mp = calibration_curve(yy[m], pp[m], n_bins=10, strategy="quantile")
            ax.plot(mp, ft, "o-", ms=3, lw=.9, alpha=.5,
                    color=PALETTE[k % len(PALETTE)])
        except Exception:
            pass
    ft, mp = calibration_curve(y, p, n_bins=12, strategy="quantile")
    ax.plot(mp, ft, "o-", color="k", lw=2.2, ms=5, label="pooled")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="perfect")
    ax.set_title(f"{name}\nBrier={brier_score_loss(y, p):.4f}  ECE={ece(y, p):.4f}",
                 fontsize=10)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.legend(fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

ax = axes[3]
for i, (key, name) in enumerate([("link", "H1"), ("expr", "H2"), ("tau", "H3")]):
    y, p = pool(key)
    if len(np.unique(y)) < 2:
        continue
    ax.hist(p, bins=40, histtype="step", lw=1.8, density=True,
            color=PALETTE[i], label=name)
ax.set_title("predicted probability distributions")
ax.set_xlabel("p"); ax.set_ylabel("density"); ax.legend(fontsize=8)

fig.tight_layout(); savefig(fig, "10_calibration"); plt.show()

print("A U-shaped probability histogram means confident predictions; a spike at")
print("the prevalence means the head has collapsed to predicting the base rate.")
""")

# =============================================================================
MD(r"""
## 23 · Training curves

Total loss and each of the five components, train vs validation, averaged over
folds with a min–max band.

What to look for:

- **Train and validation diverging** → overfitting; raise dropout or weight
  decay, or lower `MAX_NEIGHBOURS`.
- **A head whose loss goes flat almost immediately** → it has saturated and is
  contributing no gradient. The tissue head is the usual suspect. Its loss
  weight can then be reduced without hurting anything.
- **Alignment loss not falling** → the two embedding spaces are not becoming
  comparable, which makes the FiLM fusion little more than a learned projection
  of `z_seq`.
""")

CODE(r"""
comps = ["loss", "link", "expr", "tau", "tissue", "align"]
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, c in zip(axes.ravel(), comps):
    for phase, col, style in [("train", PALETTE[0], "-"), ("val", PALETTE[2], "-")]:
        piv = HIST.pivot_table(index="epoch", columns="fold", values=f"{phase}_{c}")
        m = piv.mean(1)
        ax.plot(m.index, m.values, style, color=col, lw=2, label=phase)
        ax.fill_between(piv.index, piv.min(1), piv.max(1), color=col, alpha=.15)
    ax.set_title(f"{c} loss"); ax.set_xlabel("epoch"); ax.legend(fontsize=8)
fig.suptitle("Training dynamics (mean over folds, band = min-max)", y=1.02, fontsize=13)
fig.tight_layout(); savefig(fig, "11_training_curves"); plt.show()

print("best epoch per fold:", MET["best_epoch"].tolist())
print(f"mean seconds/epoch: {HIST['sec'].mean():.1f}")
HIST.to_csv(RESULTS_DIR / "training_history.csv", index=False)
""")

# =============================================================================
MD(r"""
## 24 · The two embedding spaces

The point of the whole architecture is the pair of embedding spaces, so we look
at them directly. Using the final fold's model we embed the held-out graphs and
project each space to 2-D (UMAP when available, PCA otherwise).

Expectations, and what a violation would mean:

- **`z_seq`** should cluster by *gene family / promoter type* and be blind to
  tissue — it never sees tissue information, so visible tissue structure would
  indicate a bug (most likely a leak through the shared trunk).
- **`z_str`** should cluster by *tissue*, since contact profile and histone
  state are tissue-specific.
- **`z` (fused)** should show both: tissue structure inherited from `z_str`,
  modulated by gene identity.

The right-hand panel quantifies it with a silhouette score against tissue
labels for each space — the numeric version of the same claim.
""")

CODE(r"""
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

last_pred = fold_preds[-1]
te_idx = FOLDS[-1][1]
meta_te = META.iloc[te_idx].reset_index(drop=True)

def project(X, seed=0):
    X = np.asarray(X, dtype=np.float32)
    if len(X) > 6000:
        sub = np.random.default_rng(seed).choice(len(X), 6000, replace=False)
    else:
        sub = np.arange(len(X))
    Xs = X[sub]
    if HAS_UMAP:
        try:
            red = umap_mod.UMAP(n_neighbors=25, min_dist=.15,
                                random_state=seed).fit_transform(Xs)
            return red, sub, "UMAP"
        except Exception as e:
            print(f"  UMAP failed ({e}); using PCA")
    return PCA(n_components=2, random_state=seed).fit_transform(Xs), sub, "PCA"

spaces = [("z_seq  (sequence space)", last_pred["z_seq"]),
          ("z_str  (structure space)", last_pred["z_str"]),
          ("z      (fused gene x tissue)", last_pred["z"])]

fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
sils = {}
for ax, (name, X) in zip(axes[:3], spaces):
    emb, sub, how = project(X)
    lab = meta_te["tissue"].values[sub]
    codes = pd.Categorical(lab).codes
    ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap="tab20", s=5, alpha=.75,
               linewidths=0)
    try:
        sils[name] = silhouette_score(X[sub], codes) if len(set(codes)) > 1 else np.nan
    except Exception:
        sils[name] = np.nan
    ax.set_title(f"{name}\n{how}, coloured by tissue\n"
                 f"silhouette={sils[name]:.3f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

ax = axes[3]
names = list(sils.keys()); vals = [sils[n] for n in names]
ax.barh([n.split("(")[0].strip() for n in names], vals, color=PALETTE[:3])
ax.axvline(0, c="k", lw=1)
ax.set_title("Tissue silhouette by space\n(higher = more tissue structure)", fontsize=9)
ax.set_xlabel("silhouette score")

fig.tight_layout(); savefig(fig, "12_embedding_spaces"); plt.show()

print("\nsilhouette against tissue labels:")
for n, v in sils.items():
    print(f"  {n:<32} {v:.4f}")
print("\nExpected ordering: z_str >= z > z_seq.")
print("If z_seq scores as high as z_str, tissue information is leaking into the")
print("sequence tower and the 'two spaces' claim does not hold.")
""")

# =============================================================================
MD(r"""
## 25 · What the attention actually attends to

The GAT's final-layer attention weights are returned by `GATv2Conv(...,
return_attention_weights=True)`. We take a batch of test graphs and ask: for the
edges pointing into the promoter anchor, what predicts a high attention weight?

Three views:

1. **Attention vs CHiCAGO score** — if the model has learned nothing beyond the
   input evidence, this is a tight monotone relationship. Scatter with structure
   *around* the trend is the interesting case: contacts the model up-weights
   beyond what their score alone would justify.
2. **Attention vs genomic distance** — a strong distance dependence would mean
   the attention has rediscovered the distance-decay prior rather than
   regulatory logic.
3. **Attention weight distribution** — concentrated (a few dominant partners)
   or diffuse (near-uniform, i.e. the attention layer is doing nothing useful).

This is the panel that tells you whether the architecture earned its complexity
over a plain GCN.
""")

CODE(r"""
model = fold_models[-1].eval()
dl = make_loader(GRAPHS, FOLDS[-1][1][:2048], CFG.batch_size, False)

rows = []
with torch.no_grad():
    for batch in dl:
        batch = batch.to(DEVICE)
        seq = SEQ_BANK[batch.seq_idx.cpu()].to(DEVICE).float()   # SEQ_BANK is uint8; conv layers need float
        z_seq = model.seq(seq)
        h, z_str, attn = model.struc(batch.x, batch.edge_index, batch.edge_attr,
                                   batch.batch, batch.ptr[:-1])
        if attn is None:
            break
        ei, alpha = attn
        a = alpha.mean(-1).cpu().numpy()          # average over heads
        src, dst = ei[0].cpu().numpy(), ei[1].cpu().numpy()
        anchors = set(batch.ptr[:-1].cpu().numpy().tolist())
        ea = batch.edge_attr.cpu().numpy()
        # map original edges (self-loops were appended, so index defensively)
        n_orig = ea.shape[0]
        for k in range(min(len(a), n_orig)):
            if int(dst[k]) in anchors:            # edges flowing into a promoter
                rows.append({"alpha": float(a[k]),
                             "log_score": float(ea[k, 0]),
                             "log_reads": float(ea[k, 1]),
                             "sgn_log_d": float(ea[k, 2]),
                             "is_backbone": float(ea[k, 0] == 0.0)})
        if len(rows) > 60000:
            break

A = pd.DataFrame(rows)
print(f"collected {len(A):,} attention weights on edges into promoter anchors")

if len(A) > 100:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    real = A[A.is_backbone == 0]

    axes[0].scatter(real["log_score"], real["alpha"], s=3, alpha=.15,
                    color=PALETTE[0], linewidths=0)
    if len(real) > 20:
        b = pd.cut(real["log_score"], 12)
        med = real.groupby(b, observed=True)["alpha"].median()
        cen = [iv.mid for iv in med.index]
        axes[0].plot(cen, med.values, "o-", color="k", lw=2, ms=4, label="median")
        r = sstats.spearmanr(real["log_score"], real["alpha"])
        axes[0].set_title(f"attention vs CHiCAGO score\nSpearman rho={r.correlation:.3f}")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("log1p(CHiCAGO score)"); axes[0].set_ylabel("attention weight")

    cis = real[real["sgn_log_d"].abs() > 0]
    axes[1].scatter(cis["sgn_log_d"].abs(), cis["alpha"], s=3, alpha=.15,
                    color=PALETTE[2], linewidths=0)
    if len(cis) > 20:
        r2 = sstats.spearmanr(cis["sgn_log_d"].abs(), cis["alpha"])
        axes[1].set_title(f"attention vs genomic distance\nSpearman rho={r2.correlation:.3f}")
    axes[1].set_xlabel("|log10 distance to TSS|"); axes[1].set_ylabel("attention weight")

    axes[2].hist(real["alpha"], bins=60, color=PALETTE[3], density=True,
                 label="PCHi-C edges")
    if (A.is_backbone == 1).sum() > 10:
        axes[2].hist(A[A.is_backbone == 1]["alpha"], bins=60, alpha=.6,
                     color=PALETTE[1], density=True, label="backbone edges")
    axes[2].axvline(real["alpha"].mean(), ls="--", c="k", lw=1,
                    label=f"mean {real['alpha'].mean():.3f}")
    axes[2].set_title("attention weight distribution")
    axes[2].set_xlabel("attention weight"); axes[2].legend(fontsize=8)

    fig.suptitle("Final-layer attention on edges into the promoter anchor",
                 y=1.03, fontsize=12)
    fig.tight_layout(); savefig(fig, "13_attention_analysis"); plt.show()

    gini = lambda x: (np.abs(np.subtract.outer(x, x)).mean() / (2 * np.mean(x))
                      if np.mean(x) > 0 else np.nan)
    g = gini(real["alpha"].sample(min(3000, len(real)), random_state=0).values)
    print(f"\nGini of attention weights: {g:.3f}")
    print("  ~0   = uniform attention (the GAT is behaving like a mean aggregator)")
    print("  >0.3 = genuinely selective, a few partners dominate")
else:
    print("Not enough attention records collected to plot.")
""")

# =============================================================================
MD(r"""
## 26 · Export

Everything a reader needs to check or reuse the run:

- `results/metrics_per_fold.csv` and `metrics_summary_*.csv`
- `results/predictions_fold*.npz` — raw scores and labels for every head, so any
  metric or curve can be recomputed without retraining
- `results/embeddings.parquet` — the gene × tissue embedding `z` plus `z_seq`
  and `z_str` for the last fold's test graphs, keyed by gene and tissue. This is
  head H4 from the slide: the embedding as the deliverable.
- `results/model_fold*.pt` — weights plus the config used to build them
- `results/config.json`, `results/run_manifest.json` — the exact settings, data
  inventory and package versions of this run
- `results/figures/*.png`
""")

CODE(r"""
import platform as _pf

for k, pred in enumerate(fold_preds):
    np.savez_compressed(
        RESULTS_DIR / f"predictions_fold{k}.npz",
        **{n: pred[n] for n in ("link_p", "link_y", "expr_p", "expr_y",
                                "tau_p", "tau_y", "tis_p", "tis_y")})
print(f"predictions      -> results/predictions_fold0..{len(fold_preds)-1}.npz")

n_emb = min(len(meta_te), len(last_pred["z"]))
parts = [pd.DataFrame({
    "gene":      meta_te["gene"].values[:n_emb],
    "tissue":    meta_te["tissue"].values[:n_emb],
    "sample_id": meta_te["sample_id"].values[:n_emb],
})]
for name, arr in [("z", last_pred["z"]), ("z_seq", last_pred["z_seq"]),
                  ("z_str", last_pred["z_str"])]:
    a = np.asarray(arr)[:n_emb]
    parts.append(pd.DataFrame(a, columns=[f"{name}_{j}" for j in range(a.shape[1])]))
emb = pd.concat(parts, axis=1)
emb.to_parquet(RESULTS_DIR / "embeddings.parquet", index=False)
print(f"embeddings       -> results/embeddings.parquet  {emb.shape}")

for k, m in enumerate(fold_models):
    torch.save({"state_dict": m.state_dict(), "config": asdict(CFG),
                "n_node_feat": N_NODE_FEAT, "n_edge_feat": N_EDGE_FEAT,
                "tissues": TISSUE_LIST},
               RESULTS_DIR / f"model_fold{k}.pt")
print(f"models           -> results/model_fold0..{len(fold_models)-1}.pt")

(RESULTS_DIR / "config.json").write_text(json.dumps(asdict(CFG), indent=2, default=str))

manifest = {
    "run_finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    "python": _pf.python_version(), "platform": _pf.platform(),
    "device": str(DEVICE),
    "packages": {**HARD, **{k: str(v) for k, v in OPTIONAL.items()}},
    "data": {
        "tissues_used": int(META["tissue"].nunique()),
        "tissues_available": int(len(tissue_df)),
        "genes": int(META["gene"].nunique()),
        "graphs": int(len(GRAPHS)),
        "node_features": int(N_NODE_FEAT),
        "edge_features": int(N_EDGE_FEAT),
        "histone_tracks_present": int(len(hist_files)),
        "histone_coverage_frac": float(avail_df.values.mean()),
        "hic_real_files": int(sum(1 for v in HIC.status.values() if v.startswith("real"))),
    },
    "cv": {"folds": CFG.n_folds, "grouped_by": CFG.group_split_by},
    "headline": {
        "link_auprc":  float(MET["link_auprc"].mean()),
        "expr_auprc":  float(MET["expr_auprc"].mean()),
        "tau_auprc":   float(MET["tau_auprc"].mean()),
        "tissue_acc":  float(MET["tissue_accuracy"].mean()),
    },
}
(RESULTS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
print("manifest         -> results/run_manifest.json")
print(json.dumps(manifest["headline"], indent=2))

print(f"\nfigures ({len(list(FIG_DIR.glob('*.png')))}):")
for p in sorted(FIG_DIR.glob("*.png")):
    print(f"  {p.name}")
HIST.to_csv(RESULTS_DIR / "training_history.csv", index=False)
print("\nexport complete")
""")

# =============================================================================
MD(r"""
## 27 · What this run does and does not establish

**What it does.** Every number above comes from graphs whose genes were never
seen in training, with promoter sequence, histone state, contact topology and
four supervised objectives learned jointly. The metrics, curves and embeddings
are reproducible from `results/` without retraining.

**Where to be sceptical.**

1. **Histone coverage is the binding constraint.** Whatever fraction cell 8
   reported as available is the fraction of the node-feature block that carries
   real information; the rest is masked. If coverage is low, the structure tower
   is running mostly on topology, and any claim about chromatin state driving
   the prediction is unsupported. Run `fetch_histone_encode.py` and check
   `histone/DOWNLOAD_manifest.csv` — rows marked `proxy` are a *different*
   biosample and should be treated as noise, not signal.

2. **The `.hic` files are placeholders.** `has_hic = 0` everywhere means that
   edge feature contributed nothing. The Hi-C half of "Hi-C and PC-Hi-C" is not
   actually being tested yet.

3. **The tissue head probably saturates.** Contact profile plus histone state
   nearly determines the tissue, so H4 is close to a free win and its high
   accuracy is not evidence that the embedding is good. Treat it as a sanity
   check, not a result.

4. **Tau is a per-gene label.** H3 is constant across a gene's tissues, so the
   model can score well by recognising the gene rather than reasoning about
   tissue. Gene-grouped folds prevent train/test leakage but do not make the
   task harder than it is.

5. **Distance-matched negatives are a floor, not a guarantee.** They remove the
   easiest shortcut for H1, but promoter–enhancer contacts also correlate with
   local fragment density and mappability. An ablation against a distance-only
   logistic baseline is the honest comparison, and it is not in this notebook.

**Ablations worth running before believing any of it:**

| ablation | what it isolates |
|---|---|
| sequence tower only (`w_link=w_tissue=0`, drop `z_str`) | how much is promoter sequence alone |
| structure tower only (drop `z_seq` from fusion) | how much is contact topology alone |
| shuffle histone features within tissue | whether marks contribute beyond topology |
| replace GATv2 with mean aggregation | whether attention earns its parameters |
| `group_split_by = "chrom"` | robustness to regulatory-neighbourhood linkage |

Each is a one-line change in cell 1 plus a re-run.

**To scale up:** set `fast_mode = False` in cell 1. That restores all 46 tissues
and the full gene set, raises the promoter window to 2 kb, and trains for 40
epochs. Expect a few hours on a single GPU and roughly 20–40 GB of RAM for graph
construction; if memory is tight, lower `max_neighbours` before lowering the
gene count, since degree truncation costs less signal than dropping genes.
""")
