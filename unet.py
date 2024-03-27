## U-Net script according to Karras et al. (2023) - https://arxiv.org/pdf/2312.02696.pdf, adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py
## MP stands for "Magnitude Preserving" as the authors of the paper have introduced several measures to perserve the magnitude of weights, activations and gradients

import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import numpy as np

from einops import pack, unpack, rearrange, repeat
from functools import partial
import math
from math import sqrt, ceil


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# Define U-Net by single building blocks


### Basic Functions

class MPSiLU(Module):
    '''
    Magnitude Preserving SiLU activation functio
    '''
    def forward(self, x):
        return F.silu(x) / 0.596


class Gain(Module):
    '''
    Gain function that replaces the Bias in the Model
    '''
    
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return x * self.gain


class MPConcat(Module):
    '''
    Magnitude Preserving Concatenation, by weighting the input tensors with a factor t
    '''
    
    def __init__(self, t = 0.5, dim = -1):
        super().__init__()
        self.t = t
        self.dim = dim

    def forward(self, a, b):
        dim, t = self.dim, self.t
        Na, Nb = a.shape[dim], b.shape[dim]

        C = sqrt((Na + Nb) / ((1. - t) ** 2 + t ** 2))

        a = a * (1. - t) / sqrt(Na)
        b = b * t / sqrt(Nb)

        return C * torch.cat((a, b), dim = dim)


class MPAdd(Module):
    '''
    Magnitude Preserving Addition, by weighting the input tensors with a factor t. Empirically, t=0.3 for encoder / decoder / attention residuals and for embedding, t=0.5 according to the paper
    '''
    
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x, res):
        a, b, t = x, res, self.t
        num = a * (1. - t) + b * t
        den = sqrt((1 - t) ** 2 + t ** 2)
        return num / den
    
    
    
#####################
#### Operations #####
#####################


class PixelNorm(Module):
    '''
    Pixel-wise feature normalization to replace group normalization
    '''
    
    def __init__(self, dim, eps = 1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, dim = self.dim, eps = self.eps) * sqrt(x.shape[self.dim]) # L2Norm with scaling
    
    
    
class MPFourierEmbedding(Module):
    '''
    Embedding layer for degradation level t (and potentially class labels) that uses Fourier features instead of positional encodings
    '''
    
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, 'dimension must be divisible by 2'
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi # Formula according to Tanick et al., (2020) - https://arxiv.org/abs/2006.10739
        return torch.cat((freqs.sin(), freqs.cos()), dim = -1) * sqrt(2) # sqrt(2) for preserving magnitude
    
    

def normalize_weight(x, eps = 1e-4):
    '''
    Forced weight normalization for weights of the model, adapted directly from Karras et al. (2023). To be applied In Conv2d and Linear layers
    '''
    
    dim = list(range (1, x.ndim ))
    n = torch.linalg.vector_norm (x, dim=dim , keepdim =True)
    alpha = np.sqrt(n.numel() / x.numel())
    return x / torch.add(eps , n, alpha=alpha)




class Conv2d(Module):
    '''
    Convolutional Layer with forced weight normalization
    '''
    
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps = 1e-4,
        concat_ones_to_input = False   # they use this in the input block to protect against loss of expressivity due to removal of all biases, even though they claim they observed none
    ):
        super().__init__()
        weight = torch.randn(dim_out, dim_in + int(concat_ones_to_input), kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size ** 2 # This is quivalent to self.weight[0].numel() - we index the first element of the weight tensor and count its number of elements, which is the remaining number of elements in the tensor, i.e., dim_in * kernel_size ** 2
        self.concat_ones_to_input = concat_ones_to_input

    def forward(self, x):

        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps) # Normalizing the weights with the forced normalization function, relating to Equation 46 in the paper
                self.weight.copy_(normed_weight) # Copying the normalized weights into the weight tensor

        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in) # Normalizing the weights with the forced normalization function, relating to Equation 66 in the paper

        if self.concat_ones_to_input:
            x = F.pad(x, (0, 0, 0, 0, 1, 0), value = 1.)

        return F.conv2d(x, weight, padding='same') # Actual convolutional operation with the normalized weights


class Linear(Module):
    
    '''
    Linear Layer with forced weight normalization
    '''
    
    def __init__(self, dim_in, dim_out, eps = 1e-4):
        super().__init__()
        weight = torch.randn(dim_out, dim_in)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = normalize_weight(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = normalize_weight(self.weight, eps = self.eps) / sqrt(self.fan_in)
        return F.linear(x, weight)
    

    
class ConvAttention(Module):
    '''
    Attention Layer to be applied in the Encoder and Decoder, not for the highest resolutions, only in deeper layers.
    Operates with classical multi-head attention, but with magnitude preserving operations. Includes residual connection.
    '''
    
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 64,
        num_mem_kv = 4,
        mp_add_t = 0.3, 
        dropout = 0.1
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.pixel_norm = PixelNorm(dim = -1)
        self.attn_dropout = nn.Dropout(dropout)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = Conv2d(dim, hidden_dim * 3, 1) # 1x1 convolution to get query, key and value from input
        self.to_out = Conv2d(hidden_dim, dim, 1) # 1x1 convolution to get output from attention

        self.mp_add = MPAdd(t = mp_add_t)


    def forward(self, x):
        '''
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        '''
        
        res, b, c, h, w = x, *x.shape

        qkv = self.to_qkv(x).chunk(3, dim = 1) # Split the output of the 1x1 convolution into query, key and value
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv) # Rearrange the query, key and value to fit the attention operation

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v))) # partial here used to create concat function with default dim = -2

        q, k, v = map(self.pixel_norm, (q, k, v)) # Pixel-wise feature normalization
        
        # Calculate attention
        # similarity between q and k - matrix multiply q with k, scaled by square root of dimension
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * q.shape[-1] ** -0.5 # scale by negative square root of dimension

        # attention (softmax over similarity)
        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn) # dropout

        # aggregate values - matrix multiply v with attention
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out) # 1x1 convolution to get output from attention

        return self.mp_add(out, res) # Residual connection of attention output and input (x)
    
    

### Blocks ###

class EncoderBlock(Module):
    '''
    Encoder Block for the U-Net, including options for residual connection and attention.
    '''
    
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        emb_dim = None,
        dropout = 0.1,
        mp_add_t = 0.3,
        has_attn = False,
        attn_dim_head = 64,
        attn_res_mp_add_t = 0.3,
        downsample = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        
        self.downsample = downsample
        self.downsample_conv = None

        curr_dim = dim
        if downsample:
            self.downsample_conv = Conv2d(curr_dim, dim_out, 1)
            curr_dim = dim_out

        self.pixel_norm = PixelNorm(dim = 1)

        # Embedding
        self.to_emb = nn.Sequential(
            Linear(emb_dim, dim_out),
            Gain()
        )

        # Convolutional Blocks
        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv2d(curr_dim, dim_out, 3)
        )

        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv2d(dim_out, dim_out, 3)
        )

        # Magnitude preserving addition
        self.mp_add_res = MPAdd(t = mp_add_t)

        # Attention
        self.attn = None
        if has_attn:
            self.attn = ConvAttention(
                dim = dim_out,
                heads = max(ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
            )


    def forward(
        self,
        x,
        emb = None
    ):
    
        if self.downsample:
            h, w = x.shape[-2:]
            x = F.interpolate(x, (h // 2, w // 2), mode = 'bilinear') # Downsampling is only done by bilinear interpolation and subsequent 1x1 convolution, no pooling
            x = self.downsample_conv(x)

        # Pixel-wise feature normalization
        x = self.pixel_norm(x)

        # Copy input for residual connection
        res = x.clone()

        # Convolutional Blocks with Embedding added after first block
        x = self.block1(x)
        
        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1 1') # Note: Embedding is applied by multiplication with the input tensor, not concatenation!

        x = self.block2(x)

        # Magnitude preserving addition
        x = self.mp_add_res(x, res)
    
        # Attention
        if exists(self.attn):
            x = self.attn(x)

        return x


class DecoderBlock(Module):
    '''
    Decoder Block for the U-Net, including options for residual connection and attention.
    '''
    
    def __init__(
        self,
        dim,
        dim_out = None,
        *,
        emb_dim = None,
        dropout = 0.1,
        mp_add_t = 0.3,
        has_attn = False,
        attn_dim_head = 64,
        attn_res_mp_add_t = 0.3,
        upsample = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.upsample = upsample
        self.needs_skip = not upsample

        # Embedding Encoder
        self.to_emb = None
        if exists(emb_dim):
            self.to_emb = nn.Sequential(
                Linear(emb_dim, dim_out),
                Gain()
            )

        # Convolutional Blocks
        self.block1 = nn.Sequential(
            MPSiLU(),
            Conv2d(dim, dim_out, 3)
        )

        self.block2 = nn.Sequential(
            MPSiLU(),
            nn.Dropout(dropout),
            Conv2d(dim_out, dim_out, 3)
        )

        # Residual Convolution
        self.res_conv = Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        # Magnitude Preserving Addition
        self.mp_add_res = MPAdd(t = mp_add_t)

        # Attention
        self.attn = None
        if has_attn:
            self.attn = ConvAttention(
                dim = dim_out,
                heads = max(ceil(dim_out / attn_dim_head), 2),
                dim_head = attn_dim_head,
                mp_add_t = attn_res_mp_add_t,
            )


    def forward(
        self,
        x,
        emb = None
    ):
        # Upsampling (bilinear interpolation)
        if self.upsample:
            h, w = x.shape[-2:]
            x = F.interpolate(x, (h * 2, w * 2), mode = 'bilinear')

        # Convolve input for residual connection
        res = self.res_conv(x)

        # Convolutional Blocks with Embedding added after first block
        x = self.block1(x)

        if exists(emb):
            scale = self.to_emb(emb) + 1
            x = x * rearrange(scale, 'b c -> b c 1 1')

        x = self.block2(x)

        # Add residual connection
        x = self.mp_add_res(x, res)

        # Attention
        if exists(self.attn):
            x = self.attn(x)

        return x
    
    




###### U-Net Model ######


class UNet(Module):
    """
    going by figure 21. config G
    """

    def __init__(
        self,
        *,
        image_size,
        dim = 32,                # Number of channels - the single most important hyperparameter for number of parameters and memory usage - reduced for testing purposes to 32
        dim_max = 256,            # channels will double every downsample and cap out to this value # Reduced for testing purposes to 256 (32*2^3)
        channels = 3,             # 4 channels in paper for some reason, must be alpha channel? # Changed to 3 for RGB images
        num_downsamples = 3,
        num_blocks_per_stage = 2,
        attn_res = (16, 8),
        fourier_dim = 32,
        attn_dim_head = 64,
        mp_cat_t = 0.5,
        mp_add_emb_t = 0.5,
        attn_res_mp_add_t = 0.3,
        dropout = 0.1,
        self_condition = False
    ):
        super().__init__()

        self.self_condition = self_condition

        # determine dimensions
        self.channels = channels
        self.image_size = image_size
        input_channels = channels * (2 if self_condition else 1)

        # input and output blocks

        self.input_block = Conv2d(input_channels, dim, 3, concat_ones_to_input = True)

        self.output_block = nn.Sequential(
            Conv2d(dim, channels, 3),
            Gain()
        )

        # Time Embedding - timestep t
        emb_dim = dim * 4
        self.to_time_emb = nn.Sequential(
            MPFourierEmbedding(fourier_dim),
            Linear(fourier_dim, emb_dim)
        )
        self.emb_activation = MPSiLU()

        # No. downsamples
        self.num_downsamples = num_downsamples

        # Attention
        # attn_res = set(tuple(attn_res)) Original code, but is inflexible, writing dynamic version below
        attn_res = set([(image_size // 2**x) for x in range(1, num_downsamples)]) # Attention resolutions are every layer except the first one
        
        # Arguments for Encoder and Decoder Blocks
        block_kwargs = dict(
            dropout = dropout,
            emb_dim = emb_dim,
            attn_dim_head = attn_dim_head,
            attn_res_mp_add_t = attn_res_mp_add_t,
        )
        

        # For reference see Figure 21 in https://arxiv.org/pdf/2312.02696.pdf
        # Lists for encoder and decoder blocks
        self.downs = ModuleList([])
        self.ups = ModuleList([])

        curr_dim = dim
        curr_res = image_size

        self.mp_cat = MPConcat(t = mp_cat_t, dim = 1)

        # There is always one extra decoder block in each decoder stage / layer
        self.ups.insert(0, DecoderBlock(dim * 2, dim, **block_kwargs)) # Extra decoder block
        
        assert num_blocks_per_stage >= 1
        for _ in range(num_blocks_per_stage):
            self.downs.append(EncoderBlock(curr_dim, curr_dim, **block_kwargs))
            self.ups.insert(0, DecoderBlock(curr_dim * 2, curr_dim, **block_kwargs)) # dim * 2 because of skip connection

        # Stages
        for _ in range(self.num_downsamples):
            dim_out = min(dim_max, curr_dim * 2)
            self.ups.insert(0, DecoderBlock(dim_out, curr_dim, has_attn = curr_res in attn_res, upsample = True, **block_kwargs)) # dim_out = curr_dim * 2 because it receives the doubled channels from the encoder

            # Halving the resolution only here because corresponding Upsampling and Downsampling operation don't have the same resolution, see figure
            curr_res //= 2 
            has_attn = curr_res in attn_res
            print(f'curr_res: {curr_res}, attn_res: {attn_res}, has_attn: {has_attn}')
            
            self.downs.append(EncoderBlock(curr_dim, dim_out, downsample = True, has_attn = has_attn, **block_kwargs)) # curr_dim is half the size of dim_out, works only because it is the downsampling operation )
            self.ups.insert(0, DecoderBlock(dim_out * 2, dim_out, has_attn = has_attn, **block_kwargs)) # One extra Decoder Block to receive the residual connection from the downsampling encoder, such that upsampler doesn't need to catch it

            for _ in range(num_blocks_per_stage):
                self.downs.append(EncoderBlock(dim_out, dim_out, has_attn = has_attn, **block_kwargs))
                self.ups.insert(0, DecoderBlock(dim_out * 2, dim_out, has_attn = has_attn, **block_kwargs)) # dim_out * 2 for residual connection

            curr_dim = dim_out

        # Middle Decoders - in the paper there's only one but I guess that's not a hard rule
        mid_has_attn = curr_res in attn_res
        self.mids = ModuleList([
            DecoderBlock(curr_dim, curr_dim, has_attn = mid_has_attn, **block_kwargs),
            DecoderBlock(curr_dim, curr_dim, has_attn = mid_has_attn, **block_kwargs),
        ])

        self.out_dim = channels


    def forward(
        self,
        x,
        time,
    ):
        
        # Validate image input
        assert x.shape[1:] == (self.channels, self.image_size, self.image_size), f'input must be of shape (b, {self.channels}, {self.image_size}, {self.image_size}), is {x.shape}'

        # Time Embedding
        time_emb = self.to_time_emb(time)
        emb = self.emb_activation(time_emb)

        # Store Residuals
        residuals = []

        # Input
        x = self.input_block(x)
        residuals.append(x)

        # Downsampling and Encoder Blocks
        for encoder in self.downs:
            x = encoder(x, emb = emb)
            residuals.append(x)

        # Bottleneck Decoder Blocks
        for decoder in self.mids:
            x = decoder(x, emb = emb)

        # Upsampling and Decoder Blocks
        for decoder in self.ups:
            if decoder.needs_skip:
                skip = residuals.pop() # Remove the last element from the list and return it
                x = self.mp_cat(x, skip) # Concatenate output from last decoder block with skip connection

            x = decoder(x, emb = emb)

        # Output
        return self.output_block(x)



## Ideas for noise injections: 
## Concatenate the noise with the temporal embedding and feed it into the model at the same points as the temporal embedding