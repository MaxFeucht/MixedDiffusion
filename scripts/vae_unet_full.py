import math
import torch
import torch.nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(channels,
                                        channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(channels,
                                        channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.channels = channels
        out_channels = channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(channels)
        self.conv1 = torch.nn.Conv2d(channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.norm = Normalize(channels)
        self.q = torch.nn.Conv2d(channels,
                                 channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(channels,
                                 channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(channels,
                                 channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(channels,
                                        channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_



class VAEEncoder(nn.Module):

    def __init__(self, *, dim, dim_mult=(1,2,2,2), latent_dim, image_size, channels, temb_channels, dropout):
        super().__init__()
        self.dim = dim
        self.num_resolutions = len(dim_mult)
        self.image_size = image_size
        self.channels = channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(channels*2, # *2 for concatenation of two images
                                       self.dim,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        
        # Provisoric timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.dim,
                            self.dim*4),
            torch.nn.Linear(self.dim*4,
                            self.dim*4),
        ])

        curr_res = image_size
        in_dim_mult = (1,)+dim_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            downblock = nn.Module()
            downblock.resblock = ResnetBlock(channels=self.dim*in_dim_mult[i_level], 
                                             out_channels=self.dim*dim_mult[i_level],
                                             dropout=dropout, temb_channels=temb_channels)
            if i_level != self.num_resolutions-1:
                downblock.downsample = Downsample(self.dim*dim_mult[i_level], with_conv=True)
                curr_res = curr_res // 2
            self.down.append(downblock)

        # final block
        self.norm_out = Normalize(self.dim*dim_mult[i_level])
        self.conv_out = torch.nn.Conv2d(self.dim*dim_mult[i_level],
                                        self.dim*dim_mult[-1],
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        # dense layers for mean and logvar
        self.dense_mean = torch.nn.Linear(self.dim*dim_mult[-1]*curr_res*curr_res, latent_dim)
        self.dense_logvar = torch.nn.Linear(self.dim*dim_mult[-1]*curr_res*curr_res, latent_dim)
    

    def forward(self, xt, xtm1, temb):
        assert xt.shape[2] == xt.shape[3] == self.image_size

        # Provisoric timestep embedding
        temb = get_timestep_embedding(temb, self.dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # Combine the two images
        x = torch.cat([xt, xtm1], dim=1)

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            h = self.down[i_level].resblock(h, temb)
            if i_level != self.num_resolutions-1:
                h = self.down[i_level].downsample(h)

        # final block
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        # dense layers for mean and logvar
        h = h.view(h.size(0), -1)
        mean = self.dense_mean(h)
        logvar = self.dense_logvar(h)

        return mean, logvar
    

class VAEUNet(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,2,2), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, channels,
                 image_size):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.image_size = image_size
        self.channels = channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])


        
        # downsampling
        self.conv_in = torch.nn.Conv2d(channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = image_size

        # VAE injection
        self.vae_encoder = VAEEncoder(dim=ch, 
                                dim_mult=ch_mult, 
                                latent_dim=channels*image_size*image_size, # Returns flattened image latents
                                image_size=image_size, 
                                channels=channels, 
                                temb_channels=self.temb_ch,
                                dropout=dropout)
        

        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Bottleneck
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, xt, t, xtm1 = None, prior=None):
        assert xt.shape[2] == xt.shape[3] == self.image_size

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # VAE Injection
        
        # In Training mode, we have xtm1
        if xtm1 is not None:

            # VAE Encoder
            mu, logvar = self.vae_encoder(xt, xtm1, temb)

            # Reparameterization trick
            z_sample = torch.randn_like(mu) * torch.exp(0.5*logvar) + mu

            # KL Divergence for VAE Encoder
            self.kl_div = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1).mean()

        # In Generation mode, we don't have xtm1
        else:
            if prior is None:
                bs,channels, res = xt.shape[0], xt.shape[1], xt.shape[2]
                z_sample = torch.randn(bs, channels, res, res).to(xt.device)

            else:
                z_sample = prior

                # If feature map is not of batch size, resize z_sample
                if prior.shape[0] != xt.shape[0]:
                    z_sample = z_sample[:xt.shape[0]]
             
        # Bring latent to the same shape as the last feature map
        bs, depth, res = xt.shape[0], xt.shape[1], xt.shape[2]
        z_sample = z_sample.reshape(bs, depth, res, res) #.expand(-1, depth, -1, -1) # We only expand after we're certain that the VAE injections work on full scale

        xt = xt + 0.01 * z_sample # Delete this line after testing

        # downsampling
        hs = [self.conv_in(xt)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # U-Net Bottleneck
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h