


from functools import partial
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np
import models.configs_Trans as configs
#from monai.networks.blocks import UnetrUpBlock,UnetrUpOnlyBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock,UnetrUpOnlyBlock
import math
import torch.nn.functional as F
from einops import rearrange
from train_model import head

from typing import Sequence, Tuple, Union


class SwinTransformer_Unetr_Mask_In_Seperate(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,):
        super().__init__()
        
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))        
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed_For_SSIM(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            #self.pos_embd = SinPositionalEncoding3D(96).cuda()#SinusoidalPositionEmbedding().cuda()
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
            #self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x,mask):
        """Forward function."""
        #print ('before patch embd x size is ',x.size())  # b,1,96,96,96
        x,x_ful_size = self.patch_embed(x)
        #print ('after patch embd x size is  ',x.size())  # b,48,48,48,48
        #print ('after patch embd x_ful_size size is  ',x_ful_size.size())  # b,48,48,48,48
        assert mask is not None
        #print ('self.mask_token size ',self.mask_token.size())
        B, L, _ = x.shape
        _,_,Wh,Ww,Wt=x_ful_size.shape
        mask_tokens = self.mask_token.expand(B, L, -1)
        #print ('mask size is ',mask.size())
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        


        x = x * (1. - w) + mask_tokens * w

        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        
        for i in range(self.num_layers):
            
            layer = self.layers[i]
            
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            
            
            #print ('after x size is ', x.size())

            if i==2:
                x_feature=x
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                
                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                
                
                outs.append(out)
        #print ('info: bottle net X size is ', x.size())
            
        return x,x_feature,outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_Unetr_Mask_In_Seperate, self).train(mode)
        self._freeze_stages()



class Trans_SMIT_pre_train_Student(nn.Module):
    def __init__(
        self,
        config,
        out_channels: int=1,
        feature_size: int = 48,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = False,
        ibot_head_share: bool = True,

    ) -> None:
       
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        #self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        
        
        self.transformer = SwinTransformer_Unetr_Mask_In_Seperate(patch_size=config.patch_size, 
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        self.norm = cls_norm(384)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        
        self.head=head.iBOTHead_w_Cls_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )
    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, x_feature, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x_feature)  # B L C

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        #print ('after avgpool x size is ',x.size())
        x_cls = torch.flatten(x_cls, 1)
        #print ('x size',x.shape)
        #print ('x_region size',x_region.shape)
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)


        x_token=self.head(x_region_all)
        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec


    def get_token(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        #print ('after avgpool x size is ',x.size())
        x_cls = torch.flatten(x_cls, 1)
        #print ('x size',x.shape)
        #print ('x_region size',x_region.shape)
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)


        x_token=self.head(x_region_all)
        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec



class SwinTransformer_Unetr_Seperate(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            #self.pos_embd = SinPositionalEncoding3D(96).cuda()#SinusoidalPositionEmbedding().cuda()
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
            #self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        #print ('before patch embd x size is ',x.size())
        x = self.patch_embed(x)
        #print ('after patch embd x size is  ',x.size())

        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        #print ('there are steps numbers ',self.num_layers)
        #print ('x size is  ',x.size())
        for i in range(self.num_layers):
            #print ('stage ',i)
            layer = self.layers[i]
            #print ('before X size is ', x.size())
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)

            if i==2:
                x_feature=x
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                
                #print ('after out size is ', out.size())
                outs.append(out)
        #print ('bottle net X size is ', x.size())
            
        return x,x_feature,outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_Unetr_Seperate, self).train(mode)
        self._freeze_stages()



class Trans_SMIT_pre_train_Teacher(nn.Module):
    def __init__(
        self,
        config,
        out_channels: int=1,
        feature_size: int = 48,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = False,
        ibot_head_share: bool =True,

    ) -> None:
       
        
        super().__init__()
        self.hidden_size = hidden_size
       
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        

        self.transformer = SwinTransformer_Unetr_Seperate(patch_size=config.patch_size, 
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        
        cls_norm=partial(nn.LayerNorm, eps=1e-6)

       
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        
        self.norm = cls_norm(384)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        self.head=head.iBOTHead_w_Cls_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )
    def proj_feat(self, x, hidden_size, feat_size):
        
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        

        _,x, out_feats = self.transformer(x_in)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C

        
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        #print ('after avgpool x size is ',x.size())
        x_cls = torch.flatten(x_cls, 1)
        
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)


        x_token=self.head(x_region_all)
        
        

        return x_token




class Trans_SMIT_pre_train_Student_No_Share(nn.Module):
    def __init__(
        self,
        config,
        out_channels: int=1,
        feature_size: int = 48,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = False,
        ibot_head_share: bool = True,

    ) -> None:
       
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
       
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        
        
        self.transformer = SwinTransformer_Unetr_Mask_In_Seperate(patch_size=config.patch_size, 
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)

        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        self.norm = cls_norm(384)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        
        self.head_group=head.iBOTHead_for_one_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )

        self.head_cls=head.iBOTHead_for_one_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )

        

    def proj_feat(self, x, hidden_size, feat_size):
        #print (hidden_size)
        #print (feat_size)
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in,mask):
        

        x, x_feature, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x_feature)  # B L C
        #print ('x_region_token before size',x_region.shape)
        x_region_token=self.head_group(x_region)
        #print ('x_region_token after size',x_region_token.shape)

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls=x_cls.transpose(1,2)
        #print ('after avgpool x size is ',x_cls.size())
        #x_cls = torch.flatten(x_cls, 1)
        
        x_cls_token=self.head_cls(x_cls)
        #print ('x_cls_token size is ',x_cls_token.size())

        #print ('x_region size',x_region.shape)
        #x_token=torch.cat([x_cls_token, x_region_token], dim=1)
        #print ('INFO: x_token size',x_token.shape)

        x_token=x_cls_token,x_region_token


        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec


    def get_token(self, x_in,mask):
        

        x, out_feats = self.transformer(x_in,mask)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C

        #print ('after all transformer x size is ',x_region.size())
        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        #print ('after avgpool x size is ',x.size())
        x_cls = torch.flatten(x_cls, 1)
        #print ('x size',x.shape)
        #print ('x_region size',x_region.shape)
        x_region_all=torch.cat([x_cls.unsqueeze(1), x_region], dim=1)


        x_token=self.head(x_region_all)
        #print ('info: after encoder x size ',x_token[1].size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        #print ('after proj x size ',x.size())
        
        #z = self.encoder(x, mask)
        
        x_rec = self.decoder1(x)
        #print ('x_rec1 size after encoder is ',x_rec1.size())  # 4,256,3,3,3
        #x_rec3 = self.decoder3(x_rec1)
        #print ('x_rec3 size after encoder is ',x_rec3.size())  # 4,256,3,3,3
        x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=16,s2=16,s3=16) 
        #x_rec= rearrange(x_rec, 'b (s1 s2 s3) h w t -> b 1 (h s1) (w s2) (t s3)', s1=32,s2=32,s3=32) 
        
        #print ('info: before head x size is',x.size())
        
        #print ('encoder size after encoder is ',z.size())  # 4,256,3,3,3
        #print ('self.encoder_stride   is ',self.encoder_stride)
        
        """ 
        mask = mask.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3).unsqueeze(1).contiguous()
        
        loss_recon = F.l1_loss(x_in, x_rec, reduction='none')

        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans """

        return x_token,x_rec





class Trans_SMIT_pre_train_Teacher_No_Share(nn.Module):
    def __init__(
        self,
        config,
        out_channels: int=1,
        feature_size: int = 48,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = False,
        ibot_head_share: bool =True,

    ) -> None:
        
        
        
        super().__init__()
        self.hidden_size = hidden_size
       
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        
        self.transformer = SwinTransformer_Unetr_Seperate(patch_size=config.patch_size, 
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        
        cls_norm=partial(nn.LayerNorm, eps=1e-6)

        #below is the decoder from UnetR
        self.encoder_stride=32

        self.decoder1 = nn.Conv3d(768,out_channels=self.encoder_stride ** 3 * 1, kernel_size=1)
        
        self.norm = cls_norm(384)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.in_chans=1
        
        self.head_group=head.iBOTHead_for_one_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )

        self.head_cls=head.iBOTHead_for_one_Token(
            384,
            8192,
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer='True',
            shared_head=ibot_head_share,#'True',
        )
    def proj_feat(self, x, hidden_size, feat_size):
        
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        

        _,x, out_feats = self.transformer(x_in)
        #print ('info: before encoder x size ',x.size())

        x_region = self.norm(x)  # B L C
        x_region_token=self.head_group(x_region)

        x_cls = self.avgpool(x_region.transpose(1, 2))  # B C 1
        x_cls=x_cls.transpose(1,2)
        #print ('after avgpool x size is ',x_cls.size())
        #x_cls = torch.flatten(x_cls, 1)
        
        x_cls_token=self.head_cls(x_cls)
        #print ('x_cls_token size is ',x_cls_token.size())

        #print ('x_region size',x_region.shape)
        #x_token=torch.cat([x_cls_token, x_region_token], dim=1)
        #print ('INFO: x_token size',x_token.shape)

        x_token=x_cls_token,x_region_token




        return x_token




CONFIGS = {
    
    
    'Trans-Small_SMIT_pre_train': configs.get_3DTransSmall_config_Unetr(), 
    'Trans-Small_Unetr': configs.get_3DTransSmall_config_Unetr(), 
    
}





class PatchEmbed_For_SSIM(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, T = x.size()
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        if T % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x_reshape = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x,x_reshape



class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2,):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf)
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, T = x.size()
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        if T % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - T % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww Wt
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8//reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)


    def forward(self, x, H, W, T):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, T % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x



class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}".format(self.shift_size, self.window_size)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.T = None


    def forward(self, x, mask_matrix):
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, T, C)
        #print ('x size is ',x.size())
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_f, pad_h))
        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :T, :].contiguous()

        x = x.view(B, H * W * T, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        B_, N, C = x.shape #(num_windows*B, Wh*Ww*Wt, C)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, H, W, L, C = x.shape
    #print (' in windw_partition x size',x.shape)
    #print ('window size ',window_size)
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B, H, W, L, -1)
    return x

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        B_, N, C = x.shape #(num_windows*B, Wh*Ww*Wt, C)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



# below is the segmentation model 


class Trans_3D_Seg(nn.Module):
    def __init__(
        self,
        config,
        out_channels: int=14,
        feature_size: int = 48,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "batch",
        conv_block: bool = False,
        res_block: bool = True,

    ) -> None:
        '''
        TransMorph Model
        '''
        
        #super(TransMorph_Unetr, self).__init__()
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        

        #below is the decoder from UnetR

        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrUpOnlyBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size//2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size//2, out_channels=out_channels)  
       
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


    def forward(self, x):

        x, out_feats = self.transformer(x)
        #print ('encoder x size is ',x.size())
        x=self.proj_feat(x, self.hidden_size, self.feat_size)
        
        enc4 = out_feats[-1]#self.proj_feat(out_feats[-1], self.hidden_size, self.feat_size)
        enc3 = out_feats[-2]
        enc2 = out_feats[-3]
        enc1 = out_feats[-4]



        dec4 = self.decoder5(x, enc4)
        dec3 = self.decoder4(dec4, enc3)
        dec2 = self.decoder3(dec3, enc2)
        dec1 = self.decoder2(dec2, enc1)
        #print ('dec1 size ',dec1.size())
        dec_upsample = self.decoder1(dec1)
        logits = self.out(dec_upsample)

        return logits




class SwinTransformer_Unetr(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            #self.pos_embd = SinPositionalEncoding3D(96).cuda()#SinusoidalPositionEmbedding().cuda()
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
            #self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers) else None,
                                use_checkpoint=use_checkpoint,
                               pat_merg_rf=pat_merg_rf,)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        #print ('before patch embd x size is ',x.size())
        x = self.patch_embed(x)
        #print ('after patch embd x size is  ',x.size())

        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        #print ('there are steps numbers ',self.num_layers)
        #print ('x size is  ',x.size())
        for i in range(self.num_layers):
            #print ('stage ',i)
            layer = self.layers[i]
            #print ('before X size is ', x.size())
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, T, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                
                #print ('after out size is ', out.size())
                outs.append(out)
        #print ('bottle net X size is ', x.size())
            
        return x,outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_Unetr, self).train(mode)
        self._freeze_stages()





from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer


class UnetrUpBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetrUpOnlyBlock(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super(UnetrUpOnlyBlock, self).__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels,# + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp):
        #print ('inp size ',inp.size())
        out = self.transp_conv(inp)
        #print ('out size ',out.size())
        out = self.conv_block(out)
        return out
		
class UnetrPrUpBlock(nn.Module):
    """
    A projection upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        conv_block: bool = False,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            num_layer: number of upsampling blocks.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )
        if conv_block:
            if res_block:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetResBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
            else:
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            get_conv_layer(
                                spatial_dims,
                                out_channels,
                                out_channels,
                                kernel_size=upsample_kernel_size,
                                stride=upsample_stride,
                                conv_only=True,
                                is_transposed=True,
                            ),
                            UnetBasicBlock(
                                spatial_dims=spatial_dims,
                                in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                norm_name=norm_name,
                            ),
                        )
                        for i in range(num_layer)
                    ]
                )
        else:
            self.blocks = nn.ModuleList(
                [
                    get_conv_layer(
                        spatial_dims,
                        out_channels,
                        out_channels,
                        kernel_size=upsample_kernel_size,
                        stride=upsample_stride,
                        conv_only=True,
                        is_transposed=True,
                    )
                    for i in range(num_layer)
                ]
            )

    def forward(self, x):
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class UnetrBasicBlock(nn.Module):
    """
    A CNN module that can be used for UNETR, based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            stride: convolution stride.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()

        if res_block:
            self.layer = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )
        else:
            self.layer = UnetBasicBlock(  # type: ignore
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                norm_name=norm_name,
            )

    def forward(self, inp):
        return self.layer(inp)




class Trans_Unetr(nn.Module):
    def __init__(
        self,
        config,
        out_channels: int ,
        feature_size: int = 48,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "batch",
        conv_block: bool = False,
        res_block: bool = True,
        spatial_dims: int = 3,
        in_channels: int=1,
        #out_channels: int,
    ) -> None:
       
        super().__init__()
        self.hidden_size = hidden_size
        self.feat_size=(config.img_size[0]//32,config.img_size[1]//32,config.img_size[2]//32)
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = SwinTransformer_Unetr(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           )
        

       

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):

        x, out_feats = self.transformer(x_in)
       
        


        enc44 = out_feats[-1]   # torch.Size([4, 384, 8, 8, 8])  
        enc33 = out_feats[-2]   # torch.Size([4, 192, 16, 16, 16])
        enc22 = out_feats[-3]   # torch.Size([4, 96, 32, 32, 32])   
        enc11 = out_feats[-4]   # torch.Size([4, 48, 64, 64, 64])    
        x=self.proj_feat(x, self.hidden_size, self.feat_size) # torch.Size([4, 768, 4, 4, 4])  


        enc0 = self.encoder1(x_in)
        
        enc1 = self.encoder2(enc11)
        
        enc2 = self.encoder3(enc22)
       
        enc3 = self.encoder4(enc33)
       

        dec4 = self.encoder10(x)

        dec3 = self.decoder5(dec4, enc44)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)


        return logits

