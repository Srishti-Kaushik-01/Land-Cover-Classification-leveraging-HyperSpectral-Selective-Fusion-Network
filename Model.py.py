"""
HyperSpectral Selective Fusion Network (HSSFN) for Land Cover Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from ..registry import register_model

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Install with: pip install mamba-ssm")


class DifferentialStructureExtractor(nn.Module):
    """
    Trainable differential operators with parameterized kernel patterns.
    Addresses: Small-sample efficiency via data-efficient structural priors.
    
    Based on morphological operations for hyperspectral data [web:24][web:27].
    """
    def __init__(self, channel_count: int, kernel_dim: int = 3):
        super().__init__()
        self.kernel_dim = kernel_dim
        self.channel_count = channel_count

        # Parameterized structural kernels for edge and blob detection
        self.edge_kernel = nn.Parameter(torch.zeros(channel_count, 1, kernel_dim, kernel_dim))
        self.blob_kernel = nn.Parameter(torch.zeros(channel_count, 1, kernel_dim, kernel_dim))
        
        # Feature refinement after morphological operations
        self.edge_refine = nn.Sequential(
            nn.Conv2d(channel_count, channel_count, 1, groups=channel_count),
            nn.BatchNorm2d(channel_count),
            nn.GELU()
        )
        self.blob_refine = nn.Sequential(
            nn.Conv2d(channel_count, channel_count, 1, groups=channel_count),
            nn.BatchNorm2d(channel_count),
            nn.GELU()
        )
        
        self._initialize_kernels()

    def _initialize_kernels(self):
        """Initialize with edge-detection and blob-detection patterns"""
        k = self.kernel_dim
        center = k // 2

        with torch.no_grad():
            for i in range(self.channel_count):
                # Sobel-like pattern for edge detection (erosion-based)
                self.edge_kernel.data[i, 0, center, :] = 1.0
                self.edge_kernel.data[i, 0, :, center] = 1.0
                self.edge_kernel.data[i, 0, center, center] = 2.0

                # Gaussian-like pattern for blob detection (dilation-based)
                for x in range(k):
                    for y in range(k):
                        dist = abs(x - center) + abs(y - center)
                        if dist <= center:
                            self.blob_kernel.data[i, 0, x, y] = 1.0 / (1.0 + dist)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: x [B, C, H, W]
        Returns: (edge_response, blob_response) each [B, C, H, W]
        """
        padding = self.kernel_dim // 2

        # Edge response via depthwise convolution (erosion operation)
        edge_response = F.conv2d(x, -self.edge_kernel, padding=padding, 
                                 groups=self.channel_count)
        edge_response = -edge_response
        edge_response = self.edge_refine(edge_response)

        # Blob response via depthwise convolution (dilation operation)
        blob_response = F.conv2d(x, self.blob_kernel, padding=padding, 
                                 groups=self.channel_count)
        blob_response = self.blob_refine(blob_response)

        return edge_response, blob_response


class PrototypeCoherenceModule(nn.Module):
    """
    Cross-attention with learned prototypical embeddings for feature consistency.
    Addresses: Boundary confusion via structural memory.
    
    Uses momentum-based prototype updates for stability [web:24].
    """
    def __init__(self, prototype_count, embedding_dim, decay_rate=0.9):
        super().__init__()
        self.prototype_count = prototype_count
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate

        self.register_buffer('prototypes', torch.randn(prototype_count, embedding_dim))
        self.register_buffer('update_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('initialized', torch.tensor(False))

        self.projection_layer = nn.Linear(embedding_dim, embedding_dim)
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )

    def forward(self, feature_seq, enable_update=True):
        B, N, D = feature_seq.shape

        # Detach prototypes before projection to avoid in-place errors
        projected_prototypes = self.projection_layer(self.prototypes.detach())

        # Compute cross-attention scores
        coherence_scores = torch.matmul(feature_seq, projected_prototypes.T) / math.sqrt(D)
        coherence_weights = F.softmax(coherence_scores, dim=-1)
        prototype_context = torch.matmul(coherence_weights, projected_prototypes.unsqueeze(0))

        # Gated fusion for adaptive prototype influence
        gate_weights = self.gate(feature_seq)
        enhanced_features = feature_seq + gate_weights * prototype_context

        # Update prototypes with momentum
        if enable_update and self.training:
            with torch.no_grad():
                batch_centroid = feature_seq.mean(dim=(0, 1))
                ptr = int(self.update_ptr.item())

                # Safe in-place update
                self.prototypes.data[ptr].mul_(self.decay_rate).add_(
                    (1 - self.decay_rate) * batch_centroid
                )

                # Safe pointer update
                self.update_ptr.data.copy_(((self.update_ptr + 1) % self.prototype_count))

        return enhanced_features


class SpatialRelevanceModulator(nn.Module):
    """
    Instance-adaptive Gaussian spatial weighting based on global context.
    Addresses: Spectral-spatial trade-off via per-sample context weighting.
    """
    def __init__(self, embedding_dim: int, spatial_extent: int):
        super().__init__()
        self.spatial_extent = spatial_extent

        # Network to predict spread parameter from global features
        self.spread_estimator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, 1),
            nn.Softplus()  # Ensure positive spread
        )

        # Precompute spatial coordinate grid
        self.register_buffer('coordinate_grid', self._compute_grid(spatial_extent))

    def _compute_grid(self, extent: int):
        """Generate 2D coordinate grid centered at origin"""
        center = extent // 2
        y, x = torch.meshgrid(torch.arange(extent), torch.arange(extent), indexing='ij')
        grid = torch.stack([x - center, y - center], dim=-1).float()  # [H, W, 2]
        return grid

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W]
        Returns:
            modulated: [B, C, H, W]
        """
        B, C, H, W = feature_map.shape

        # Aggregate global context
        global_context = feature_map.mean(dim=(2, 3))  # [B, C]

        # Estimate spread parameter per instance
        spread_param = self.spread_estimator(global_context).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        
        # Clamp spread to reasonable range for stability
        spread_param = torch.clamp(spread_param, min=0.5, max=5.0)

        # Compute Gaussian relevance weights
        distance_sq = (self.coordinate_grid ** 2).sum(dim=-1)  # [H, W]
        relevance_map = torch.exp(-distance_sq / (2 * spread_param ** 2 + 1e-6))  # [B, 1, H, W]

        # Apply spatial modulation
        modulated = feature_map * relevance_map

        return modulated


class SelectiveStateFusion(nn.Module):
    """
    Selective state space fusion using Mamba for multi-stream integration.
    Addresses: Long-range dependencies with linear complexity O(N) vs O(N²) attention.
    
    Based on Mamba SSM architecture [web:1][web:26].
    Uses increased state dimension for better feature adaptation [web:19][web:21].
    """
    def __init__(self, embedding_dim: int, state_dim: int = 16, conv_width: int = 4,
                 expansion_factor: int = 2):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for SelectiveStateFusion. "
                "Install with: pip install mamba-ssm"
            )

        # Separate Mamba blocks for each structural stream
        # Note: Increased state_dim for better information storage [web:19]
        self.mamba_edge = Mamba(
            d_model=embedding_dim,
            d_state=state_dim,
            d_conv=conv_width,
            expand=expansion_factor
        )
        
        self.mamba_blob = Mamba(
            d_model=embedding_dim,
            d_state=state_dim,
            d_conv=conv_width,
            expand=expansion_factor
        )
        
        self.mamba_spectral = Mamba(
            d_model=embedding_dim,
            d_state=state_dim,
            d_conv=conv_width,
            expand=expansion_factor
        )

        # Layer normalization for stability [web:1]
        self.norm_edge = nn.LayerNorm(embedding_dim)
        self.norm_blob = nn.LayerNorm(embedding_dim)
        self.norm_spectral = nn.LayerNorm(embedding_dim)

        # Adaptive stream fusion with learnable temperature
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        self.stream_aggregator = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        self.output_projection = nn.Linear(embedding_dim * 3, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, edge_features: torch.Tensor, blob_features: torch.Tensor, 
                spectral_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features, blob_features, spectral_features: [B, N, D]
        Returns:
            fused_output: [B, N, D]
        """
        B, N, D = edge_features.shape

        # Process each stream through Mamba with residual connections
        edge_processed = edge_features + self.mamba_edge(self.norm_edge(edge_features))
        blob_processed = blob_features + self.mamba_blob(self.norm_blob(blob_features))
        spectral_processed = spectral_features + self.mamba_spectral(self.norm_spectral(spectral_features))

        # Concatenate processed streams
        multi_stream = torch.cat([edge_processed, blob_processed, spectral_processed], dim=-1)

        # Compute adaptive fusion weights with temperature scaling
        fusion_coeffs = self.stream_aggregator(multi_stream.mean(dim=1, keepdim=True))  # [B, 1, 3]
        fusion_coeffs = fusion_coeffs / (self.temperature + 1e-6)
        fusion_coeffs = F.softmax(fusion_coeffs, dim=-1)
        
        # Weighted combination
        weighted_fusion = (fusion_coeffs[:, :, 0:1] * edge_processed + 
                          fusion_coeffs[:, :, 1:2] * blob_processed + 
                          fusion_coeffs[:, :, 2:3] * spectral_processed)

        # Project concatenated features and add weighted fusion
        fused_output = self.output_projection(multi_stream) + weighted_fusion
        fused_output = self.dropout(fused_output)
        
        return fused_output


class HierarchicalFusionUnit(nn.Module):
    """
    Hierarchical processing unit combining selective state fusion and prototype coherence.
    
    Architecture inspired by Transformer blocks but uses Mamba for linear complexity.
    """
    def __init__(self, embedding_dim: int, state_dim: int = 16, mlp_expansion: float = 4.0,
                 prototype_count: int = 64, spatial_extent: int = 11, dropout: float = 0.1):
        super().__init__()
        self.pre_norm1 = nn.LayerNorm(embedding_dim)
        self.pre_norm2 = nn.LayerNorm(embedding_dim)
        self.pre_norm3 = nn.LayerNorm(embedding_dim)

        self.selective_fusion = SelectiveStateFusion(embedding_dim, state_dim)
        self.coherence_module = PrototypeCoherenceModule(prototype_count, embedding_dim)

        # Position-wise feed-forward network with dropout
        ffn_hidden_dim = int(embedding_dim * mlp_expansion)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        
        # Layer scale for better training stability
        self.layer_scale1 = nn.Parameter(torch.ones(embedding_dim) * 1e-4)
        self.layer_scale2 = nn.Parameter(torch.ones(embedding_dim) * 1e-4)
        self.layer_scale3 = nn.Parameter(torch.ones(embedding_dim) * 1e-4)

    def forward(self, edge_seq: torch.Tensor, blob_seq: torch.Tensor,
                spectral_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_seq, blob_seq, spectral_seq: [B, N, D]
        Returns:
            output: [B, N, D]
        """
        # Selective state fusion with pre-normalization and layer scale
        fusion_output = self.selective_fusion(
            self.pre_norm1(edge_seq),
            self.pre_norm1(blob_seq),
            self.pre_norm1(spectral_seq)
        )
        x = spectral_seq + self.layer_scale1 * fusion_output

        # Prototype coherence enhancement
        coherence_output = self.coherence_module(self.pre_norm2(x))
        x = x + self.layer_scale2 * (coherence_output - x)

        # Position-wise FFN with residual
        ffn_output = self.feed_forward(self.pre_norm3(x))
        x = x + self.layer_scale3 * ffn_output

        return x


class HSSFN(nn.Module):
    """
    HyperSpectral Selective Fusion Network for Land Cover Classification.

    Three core innovations:
    1. Differential Structure Extraction - addresses small-sample efficiency
    2. Prototype Coherence Module - addresses boundary confusion
    3. Spatial Relevance Modulation - addresses spectral-spatial trade-off
    
    Architecture: Mamba-based selective state space model for efficient sequence modeling
    References: 
    - Mamba: Linear-Time Sequence Modeling [web:26]
    - Morphological operations for hyperspectral data [web:24][web:27]
    """
    def __init__(self, 
                 spectral_bands: int = 200,      # Number of spectral channels
                 spatial_dim: int = 11,          # Spatial patch dimension
                 num_classes: int = 16,          # Number of target classes
                 feature_dim: int = 128,         # Feature embedding dimension
                 encoder_depth: int = 6,         # Number of hierarchical units
                 state_dim: int = 16,            # Mamba SSM state dimension (increase for more capacity)
                 mlp_expansion: float = 4.0,     # FFN expansion ratio
                 prototype_count: int = 64,      # Number of prototypes
                 structure_kernel: int = 3,      # Structural kernel size
                 dropout: float = 0.1):          # Dropout rate
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for HSSFN. "
                "Install with: pip install mamba-ssm"
            )

        self.spatial_dim = spatial_dim
        self.patch_count = spatial_dim * spatial_dim

        # Spectral-to-feature embedding with batch normalization
        self.spectral_encoder = nn.Sequential(
            nn.Conv2d(spectral_bands, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )

        # Differential structure extraction
        self.structure_extractor = DifferentialStructureExtractor(feature_dim, structure_kernel)

        # Learnable positional embeddings
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.patch_count, feature_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Hierarchical fusion units (Mamba-based)
        self.fusion_units = nn.ModuleList([
            HierarchicalFusionUnit(feature_dim, state_dim, mlp_expansion, 
                                  prototype_count, spatial_dim, dropout)
            for _ in range(encoder_depth)
        ])

        # Spatial relevance modulation
        self.relevance_modulator = SpatialRelevanceModulator(feature_dim, spatial_dim)

        # Classification head with dropout
        self.output_norm = nn.LayerNorm(feature_dim)
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize parameters following best practices"""
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        
        # Initialize classifier with small weights
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] where C=spectral bands, H=W=spatial_dim
        Returns:
            class_logits: [B, num_classes]
        """
        # Handle 5D input if present
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)
        
        B = x.shape[0]

        # Spectral feature encoding
        features = self.spectral_encoder(x)  # [B, D, H, W]

        # Apply spatial relevance modulation
        features = self.relevance_modulator(features)

        # Extract differential structures (morphological operations)
        edge_features, blob_features = self.structure_extractor(features)  # Both [B, D, H, W]

        # Convert spatial features to sequences
        def spatial_to_sequence(tensor):
            return tensor.flatten(2).transpose(1, 2)  # [B, D, H, W] -> [B, N, D]

        edge_seq = spatial_to_sequence(edge_features) + self.positional_encoding
        blob_seq = spatial_to_sequence(blob_features) + self.positional_encoding
        spectral_seq = spatial_to_sequence(features) + self.positional_encoding
        
        # Apply position dropout
        edge_seq = self.pos_dropout(edge_seq)
        blob_seq = self.pos_dropout(blob_seq)
        spectral_seq = self.pos_dropout(spectral_seq)

        # Process through hierarchical fusion units (Mamba-based)
        for fusion_unit in self.fusion_units:
            spectral_seq = fusion_unit(edge_seq, blob_seq, spectral_seq)
            # Update structural sequences (shared representation)
            edge_seq = spectral_seq
            blob_seq = spectral_seq

        # Global feature aggregation
        aggregated = self.output_norm(spectral_seq)
        global_descriptor = aggregated.mean(dim=1)  # [B, D]
        
        # Classification with dropout
        global_descriptor = self.head_dropout(global_descriptor)
        class_logits = self.classifier(global_descriptor)  # [B, num_classes]

        return class_logits
    
    def get_attention_weights(self):
        """Extract fusion weights from each layer for visualization"""
        weights = []
        for unit in self.fusion_units:
            if hasattr(unit.selective_fusion, 'stream_aggregator'):
                weights.append(unit.selective_fusion.temperature.detach())
        return weights


@register_model('HSSFN', expects_4d=True, feature_dim=128, encoder_depth=6, state_dim=16)
def proposed(pretrained: bool = False, **kwargs) -> HSSFN:
    """
    Constructs a HSSFN model for hyperspectral image classification.
    
    Args:
        pretrained: Whether to load pretrained weights (not implemented yet)
        **kwargs: Additional model configuration parameters
    
    Returns:
        HSSFN model instance
    """
    # Map standardized names to model-specific names
    if 'bands' in kwargs:
        kwargs['spectral_bands'] = kwargs.pop('bands')
    if 'patch_size' in kwargs:
        kwargs['spatial_dim'] = kwargs.pop('patch_size')
    
    model_config = dict(
        spectral_bands=200,
        spatial_dim=11,
        num_classes=16,
        feature_dim=128,
        encoder_depth=6,
        state_dim=16,  # Can increase to 32 or 64 for more capacity [web:19]
        mlp_expansion=4.0,
        prototype_count=128,
        structure_kernel=3,
        dropout=0.1
    )
    config = dict(model_config, **kwargs)
    return HSSFN(**config)


@register_model('HSSFN-3', expects_4d=True, feature_dim=128, encoder_depth=3, state_dim=16)
def proposed(pretrained: bool = False, **kwargs) -> HSSFN:
    """
    Constructs a HSSFN model for hyperspectral image classification.
    
    Args:
        pretrained: Whether to load pretrained weights (not implemented yet)
        **kwargs: Additional model configuration parameters
    
    Returns:
        HSSFN model instance
    """
    # Map standardized names to model-specific names
    if 'bands' in kwargs:
        kwargs['spectral_bands'] = kwargs.pop('bands')
    if 'patch_size' in kwargs:
        kwargs['spatial_dim'] = kwargs.pop('patch_size')
    
    model_config = dict(
        spectral_bands=200,
        spatial_dim=11,
        num_classes=16,
        feature_dim=128,
        encoder_depth=6,
        state_dim=16,  # Can increase to 32 or 64 for more capacity [web:19]
        mlp_expansion=4.0,
        prototype_count=128,
        structure_kernel=3,
        dropout=0.1
    )
    config = dict(model_config, **kwargs)
    return HSSFN(**config)



@register_model('HSSFN-9', expects_4d=True, feature_dim=128, encoder_depth=9, state_dim=16)
def proposed(pretrained: bool = False, **kwargs) -> HSSFN:
    """
    Constructs a HSSFN model for hyperspectral image classification.
    
    Args:
        pretrained: Whether to load pretrained weights (not implemented yet)
        **kwargs: Additional model configuration parameters
    
    Returns:
        HSSFN model instance
    """
    # Map standardized names to model-specific names
    if 'bands' in kwargs:
        kwargs['spectral_bands'] = kwargs.pop('bands')
    if 'patch_size' in kwargs:
        kwargs['spatial_dim'] = kwargs.pop('patch_size')
    
    model_config = dict(
        spectral_bands=200,
        spatial_dim=11,
        num_classes=16,
        feature_dim=128,
        encoder_depth=6,
        state_dim=16,  # Can increase to 32 or 64 for more capacity [web:19]
        mlp_expansion=4.0,
        prototype_count=128,
        structure_kernel=3,
        dropout=0.1
    )
    config = dict(model_config, **kwargs)
    return HSSFN(**config)


# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        from torchinfo import summary
        model = proposed().to(device)
        print(summary(
            model,
            input_size=(8, 200, 11, 11),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            device=device,
            dtypes=[torch.float32]
        ))
    except ImportError:
        print("torchinfo not installed. Running basic test...")
        model = proposed().to(device)
        x = torch.randn(8, 200, 11, 11).to(device)
        
        # Test forward pass
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(x)
        
        print(f"✓ Output shape: {logits.shape}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

