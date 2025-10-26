import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .mossformer2_block import ScaledSinuEmbedding, MossformerBlock_GFSMN, MossformerBlock
from .dynamic_slimmable_block import DynamicSlimmableBlock  # Import DSB

EPS = 1e-8

class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x

class CumulativeLayerNorm(nn.LayerNorm):
    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = super().forward(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            x = super().forward(x)
            x = torch.transpose(x, 1, 2)
        return x

def select_norm(norm, dim, shape):
    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)

class Encoder(nn.Module):
    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        x = self.conv1d(x)
        x = F.relu(x)
        return x

class Decoder(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x

class IdentityBlock:
    def _init__(self, **kwargs):
        pass

    def __call__(self, x):
        return x

class MossFormerM(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model=None,
        causal=False,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.,
        attn_dropout=0.1
    ):
        super().__init__()
        self.mossformerM = MossformerBlock_GFSMN(
            dim=d_model,
            depth=num_blocks,
            group_size=group_size,
            query_key_dim=query_key_dim,
            expansion_factor=expansion_factor,
            causal=causal,
            attn_dropout=attn_dropout
        )
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src):
        output = self.mossformerM(src)
        output = self.norm(output)
        return output

class MossFormerM2(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model=None,
        causal=False,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.,
        attn_dropout=0.1
    ):
        super().__init__()
        self.mossformerM = MossformerBlock(
            dim=d_model,
            depth=num_blocks,
            group_size=group_size,
            query_key_dim=query_key_dim,
            expansion_factor=expansion_factor,
            causal=causal,
            attn_dropout=attn_dropout
        )
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src):
        output = self.mossformerM(src)
        output = self.norm(output)
        return output

class Computation_Block(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        out_channels: int,
        norm: str = "ln",
        skip_around_intra: bool = True,
    ):
        super(Computation_Block, self).__init__()

        # Replace MossFormerM with DynamicSlimmableBlock for intra-chunk processing
        self.intra_mdl = DynamicSlimmableBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            lorder=20,  # Match lorder from original FSMN design
            utilization_factors=[0.125, 1.0],  # Per paper's U = {0.125, 1.0}
            hidden_size=1024  # Align with typical FFN hidden size
        )
        self.skip_around_intra = skip_around_intra

        # Set normalization type
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, S = x.shape

        # Permute to [B, S, N] for DSB processing
        intra = x.permute(0, 2, 1).contiguous()

        # Process through DSB, which returns output and losses
        intra, losses = self.intra_mdl(intra)

        # Permute back to [B, N, S]
        intra = intra.permute(0, 2, 1).contiguous()

        # Apply normalization if specified
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # Add skip connection around the intra layer if enabled
        if self.skip_around_intra:
            intra = intra + x

        out = intra
        return out, losses  # Return losses for potential training use

class MossFormer_MaskNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 24,
        norm: str = "ln",
        num_spks: int = 2,
        skip_around_intra: bool = True,
        use_global_pos_enc: bool = True,
        max_length: int = 20000,
    ):
        super(MossFormer_MaskNet, self).__init__()
        
        self.num_spks = num_spks
        self.num_blocks = num_blocks
        
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d_encoder = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
        self.use_global_pos_enc = use_global_pos_enc
        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        self.mdl = Computation_Block(
            num_blocks,
            out_channels,
            norm,
            skip_around_intra=skip_around_intra,
        )

        self.conv1d_out = nn.Conv1d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.conv1_decoder = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.conv1d_encoder(x)
        
        if self.use_global_pos_enc:
            base = x
            x = x.transpose(1, -1)
            emb = self.pos_enc(x)
            emb = emb.transpose(0, -1)
            x = base + emb

        x = self.mdl(x)[0]  # Take only the output, ignoring losses for now
        x = self.prelu(x)
        x = self.conv1d_out(x)
        B, _, S = x.shape
        x = x.view(B * self.num_spks, -1, S)
        x = self.output(x) * self.output_gate(x)
        x = self.conv1_decoder(x)
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        x = x.transpose(0, 1)
        return x

class MossFormer(nn.Module):
    def __init__(
        self,
        in_channels=512,
        out_channels=512,
        num_blocks=24,
        kernel_size=16,
        norm="ln",
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        super(MossFormer, self).__init__()
        self.num_spks = num_spks
        
        self.enc = Encoder(kernel_size=kernel_size, out_channels=in_channels, in_channels=1)
        self.mask_net = MossFormer_MaskNet(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            norm=norm,
            num_spks=num_spks,
            skip_around_intra=skip_around_intra,
            use_global_pos_enc=use_global_pos_enc,
            max_length=max_length,
        )
        self.dec = Decoder(
            in_channels=out_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False
        )

    def forward(self, input: torch.Tensor) -> list:
        x = self.enc(input)
        mask = self.mask_net(x)
        x = torch.stack([x] * self.num_spks)
        sep_x = x * mask
        est_source = torch.cat(
            [self.dec(sep_x[i]).unsqueeze(-1) for i in range(self.num_spks)],
            dim=-1,
        )
        T_origin = input.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]
        out = []
        for spk in range(self.num_spks):
            out.append(est_source[:, :, spk])
        return out

class MossFormer2_SS_16K(nn.Module):
    def __init__(self, args):
        super(MossFormer2_SS_16K, self).__init__()
        self.model = MossFormer(
            in_channels=args.encoder_embedding_dim,
            out_channels=args.mossformer_sequence_dim,
            num_blocks=args.num_mossformer_layer,
            kernel_size=args.encoder_kernel_size,
            norm="ln",
            num_spks=args.num_spks,
            skip_around_intra=True,
            use_global_pos_enc=True,
            max_length=20000
        )

    def forward(self, x: torch.Tensor) -> list:
        outputs = self.model(x)
        return outputs