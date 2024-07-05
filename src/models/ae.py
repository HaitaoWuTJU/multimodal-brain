import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Block
import numpy as np

class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data to Patch Embedding
    """
    def __init__(self, size=256, patch_size=16, in_chans=63, embed_dim=768):
        super().__init__()
        num_patches = size // patch_size
        self.patch_shape = patch_size
        self.size = size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_chans = in_chans

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B,C, T = x.shape # batch, channel, time
        x = self.proj(x)
        x = x.transpose(1, 2).contiguous() # put embed_dim at the last dimension
        return x


def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(length, dtype=np.float32)

    grid_l = grid_l.reshape([1, length])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Autoencoder(nn.Module):
    def __init__(self,size = 250 ,patch_size=25 ,in_chans=63,embed_dim=512,depth=24,num_heads=16,mlp_ratio=1.,decoder_embed_dim=512,decoder_depth=8,decoder_num_heads=16,norm_layer=nn.LayerNorm,norm_pix_loss=False):
        super(Autoencoder, self).__init__()
        #patch embed
        self.patch_embed = PatchEmbed1D(size = size,patch_size = patch_size,in_chans=in_chans,embed_dim=embed_dim)

        #pos embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # Encoder
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1,self.patch_embed.num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
                    Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                    for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * in_chans, bias=True) # decoder to patch
        
        self.norm_pix_loss = norm_pix_loss
        
        self.initialize_weights()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
         
    def patchify(self, x):
        """
        x: (N, 63, T)
        x: (N, L, patch_size)
        """
        p = self.patch_embed.patch_size
        assert x.ndim == 3 and x.shape[2] % p == 0

        h = x.shape[2] // p
        
        x = x.reshape(x.shape[0],h,p*63)
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 63, T)
        """

        x = x.reshape(x.shape[0], 63, -1)
        return x
       
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
    

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_pool(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1, keepdim=True)
        x = self.norm(x)
        return x
    
    def forward_cls(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:,0]
    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
        
    def forward_loss(self, x, pred, mask):
        """
        x: [N, 63, T]
        pred: [N, L, p*63]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(x)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, mask_ratio=0.75):
        # x = F.interpolate(x.unsqueeze(1), size=(63, 256), mode='bilinear', align_corners=True).squeeze(1)
        
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*63]
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask
    

class Autoencoder_MLP(nn.Module):
    def __init__(self,in_dim=17*250,h=1024,latend_dim=512):
        super(Autoencoder_MLP, self).__init__()
        self.embedder = nn.Sequential(
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                # nn.Linear(h, h),
                # nn.LayerNorm(h),
                # nn.GELU(),
                nn.Linear(h, latend_dim)
            )
        self.builder = nn.Sequential(
                nn.Linear(latend_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                # nn.Linear(h, h),
                # nn.LayerNorm(h),
                # nn.GELU(),
                nn.Linear(h, in_dim),
            )
    def forward(self, x):
        b,c,t=x.shape
        source_x = x
        x = x.view(x.shape[0],-1)
        x = self.embedder(x)
        x = self.builder(x)
        x = x.view(b,c,t)
        
        loss = (x - source_x) ** 2
        loss = loss.mean()
    
        return loss,x,None
    def unpatchify(self,x):
        return x
    def forward_pool(self,x):
        x = x.view(x.shape[0],-1)
        x = self.embedder(x)
        return x
if __name__=="__main__":
    model = Autoencoder_MLP()
    x = torch.randn(10, 63, 250)
    x = model(x)
    print(x.shape)
    
    