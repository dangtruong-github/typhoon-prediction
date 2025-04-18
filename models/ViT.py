import torch
import timm

import torch.nn as nn

from torchsummary import summary
from timm.models.vision_transformer import PatchEmbed, Block
# from transformers import ViTImageProcessor, ViTForImageClassification

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """Attention Block.

        Args:
            embed_dim: Dimensionality of input and attention feature vectors
            hidden_dim: Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads: Number of heads to use in the Multi-Head Attention block
            dropout: Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
    
class ViT_scratch(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0,
    ):
        """Vision Transformer.

        Args:
            embed_dim: Dimensionality of the input feature vectors to the Transformer
            hidden_dim: Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels: Number of channels of the input (3 for RGB)
            num_heads: Number of heads to use in the Multi-Head Attention block
            num_layers: Number of layers to use in the Transformer
            num_classes: Number of classes to predict
            patch_size: Number of pixels that the patches have per dimension
            num_patches: Maximum number of patches an image can have
            dropout: Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels * (patch_size**2), embed_dim)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), 
                                      nn.Linear(embed_dim, num_classes))
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        # Preprocess input
        x = self.img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)
 
        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : T + 1]

        # Apply Transforrmer
        x = self.dropout(x)
        # x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = torch.mean(x, dim=1)
        cls = x[:, 0]
        out = self.mlp_head(cls)
        return self.sm(out)
    
    def img_to_patch(self, x, patch_size, flatten_channels=True):
        """
        Args:
            x: Tensor representing the image of shape [B, C, H, W]
            patch_size: Number of pixels per dimension of the patches (integer)
            flatten_channels: If True, the patches will be returned in a flattened format
                            as a feature vector instead of a image grid.
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
        x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
        if flatten_channels:
            x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
        return x

class ViT_remaster(nn.Module):
    def __init__(
        self,
        img_size = [61, 81],
        patch_size = 8,
        inp_channels = 9,
        embed_dim = 768,
        num_heads = 8,
        num_layers = 12,
        mlp_ratio = 4.,
        head_dim = 256,
        num_classes = 1,
        dropout=0.1,
    ):   
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, inp_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True)  # fixed sin-cos embedding
        
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm) 
                                     for i in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.num_classes = num_classes
        
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), 
                                      nn.Linear(embed_dim, head_dim),
                                      nn.Linear(head_dim, num_classes))
        self.sm = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Preprocess input
        x = self.patch_embed(x)
        B, T, _ = x.shape
        
        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed[:, : T + 1]

        # print(x.shape)
        # Apply Transforrmer
        x = self.dropout(x)
        # print(x.shape)
        # x = x.transpose(0, 1)
        # print(x.shape)
        x = self.blocks(x)

        # Perform classification prediction
        cls = torch.mean(x, dim=1)
        cls = x[:, 0]
        out = self.mlp_head(cls)
        if self.num_classes == 1:
            return out
        return self.sm(out)

class ViT_timm_pretrained(nn.Module):
    def __init__(
        self, 
        model_name = 'vit_base_patch16_224',
        pretrained = True,  # default size for the pretrained model
        inp_channels = 228,  # default number of input channels for the pretrained model
        num_classes = 2,  # default number of classes for the pretrained model
    ):
        
        super().__init__()
        # Load the pretrained model from timm
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes,
            in_chans=inp_channels,
        )
    
    def forward(self, x):
        return self.model(x)
    
    def hihi(self):
        return self.model
    
    
def create_model(out_dir = None,
                 model = ViT_remaster,
                 **kwargs):
    # if out_dir is not None:
    #     Path(os.path.join(out_dir, 'config')).mkdir(parents=True, exist_ok=True)
    #     save_kwargs = dict(locals())
    #     save_kwargs['img_size'] = 'hihi'
    #     save_kwargs['model'] = model.__name__
    #     json.dump(save_kwargs, open(os.path.join(out_dir, 'config', 'Model.json'), 'w'))
    #     print('[INFO]: Model config saved!')
        
    return model(**kwargs)

if __name__ == '__main__':
    # model = ViT_final(
    #     img_size = [33, 33],
    #     patch_size = 8,
    #     in_chans = 228,
    #     num_classes = 2,
    #     embed_dim = 1024,
    #     depth = 4,
    #     num_heads = 8,
    #     head_dim = 128, # 1024//
    #     mlp_ratio = 4.,).to('cuda')
    
    model = ViT_remaster(
        img_size = [33, 33],
        patch_size = 8,
        inp_channels = 228,
        embed_dim = 1024,
        num_heads = 16,
        num_layers = 8,
        mlp_ratio = 4.,
        head_dim = 1024,
        num_classes = 2,
        dropout=0.0,
    ).to('cuda')
    
    # model = create_model(
    #     model = ViT_timm_pretrained,
    #     model_name = 'vit_base_patch16_224',
    #     pretrained = True,
    #     inp_channels = 228,
    #     num_classes = 100,
    # ).to('cuda')
    
    
    # print(model)
    
    print(summary(model, (228, 33, 33)))