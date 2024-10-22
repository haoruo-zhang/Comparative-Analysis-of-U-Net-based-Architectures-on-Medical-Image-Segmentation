import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import copy
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from ResNetV2 import ResNetV2
import numpy as np


class Attention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super(Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(0.0)
        self.proj_dropout = Dropout(0.0)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # (B, 1024, 768)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # (B,12,1024,64)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # (B, 12, 1024, 1024)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # (B,12,1024,64)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (B,1024,12,64)
        # (B,1024,768)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, hidden_size=768, mlp_dim=3072):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = F.gelu
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.dropout(self.act_fn(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x  # (B, 1024, 768)


class Embeddings(nn.Module):
    def __init__(self, img_size, n_in, hidden_size=768, width_factor=1, grid_size=14):
        super(Embeddings, self).__init__()
        n_grid = img_size // grid_size
        img_size = _pair(img_size)
        patch_size = (img_size[0] // n_grid // grid_size, img_size[1] // n_grid // grid_size)  # (1,1)
        patch_size_real = (patch_size[0] * n_grid, patch_size[1] * n_grid)  # (16,16)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  # 14*14

        self.hybrid_model = ResNetV2(n_in=n_in, block_units=[3, 4, 9], width_factor=width_factor)
        in_channels = self.hybrid_model.width * n_grid
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, hidden_size))  # (1,n_patches, hidden): (1,196,768)

        self.dropout = Dropout(0.1)

    def forward(self, x):
        x, features = self.hybrid_model(x)  # (B,1,H,W) -> (B,1024,H/16,W/16)
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2)) = (B,768,H/16,W/16)
        x = x.flatten(2)  # (B, hidden, n_patches)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden) = (B, 1024, 768)
        embeddings = x + self.position_embeddings  # (B, 1024, 768)
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, hidden_size=768):
        super(Block, self).__init__()
        self.norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention()

    def forward(self, x):
        h = x
        x = self.norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.norm(x)
        x = self.ffn(x)
        x = x + h
        return x  # (B, 1024, 768)


class Encoder(nn.Module):
    def __init__(self, hidden_size=768, num_layers=12):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded  # (B, 1024, 768)


class Transformer(nn.Module):
    def __init__(self, img_size, n_in, hidden_size=768, num_layers=12):
        super(Transformer, self).__init__()
        self.embedding = Embeddings(img_size, n_in)
        self.encoder = Encoder(hidden_size, num_layers)

    def forward(self, input_ids):
        # embedding_output: (B, 1024, 768)
        # features: (B, 512, H/8, W/8);(B, 256, H/4, W/4); (B, 64, H/2, W/2)
        embedding_output, features = self.embedding(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patches, hidden)
        return encoded, features


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_bn=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_bn),
            nn.ReLU(inplace=True)
        ]
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        super().__init__(*layers)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_bn=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_bn=use_bn)
        self.conv2 = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_bn=use_bn)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip=None):
        x = self.up(x)  # (B, 512, 64, 64)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # (B, 1024, 64, 64)
        x = self.conv1(x)  # (B, 256, 64, 64)
        x = self.conv2(x)  # (B, 256, 64, 64)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsample=1):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)]
        if upsample > 1:
            layers.append(nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True))
        super().__init__(*layers)


class DecoderCup(nn.Module):
    def __init__(self, n_skip, hidden_size=768, head_channels=512):
        super().__init__()
        self.conv_more = Conv2dReLU(hidden_size, head_channels, kernel_size=3, padding=1, use_bn=True)
        decoder_channels = [256, 128, 64, 16]
        in_channels = [head_channels] + decoder_channels[:-1]
        skip_channels = [512, 256, 64, 0] if n_skip != 0 else [0, 0, 0, 0]
        for i in range(4 - n_skip):
            skip_channels[3 - i] = 0

        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
            zip(in_channels, decoder_channels, skip_channels)
        ])

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))  # 32, 32
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)  # (B, 768, 32, 32)
        x = self.conv_more(x)  # (B,512,32,32)
        for i, decoder_block in enumerate(self.blocks):  # (512, 256, 512); (256, 128, 256); (128, 64, 64); (64, 16, 0)
            skip = features[i] if (features is not None and i < len(features)) else None
            x = decoder_block(x, skip=skip)
        return x  # (B, 16, 512, 512)


class TransUnet(nn.Module):
    def __init__(self, num_classes, n_skip, img_size=224, n_in=1, hidden_size=768, head_channels=512, num_layers=12):
        super(TransUnet, self).__init__()
        self.transformer = Transformer(img_size, n_in, hidden_size, num_layers)
        self.decoder = DecoderCup(n_skip, hidden_size, head_channels)
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=num_classes,
            kernel_size=3,
        )

    def forward(self, x):
        # x: (B, n_patch, hidden) = (B, 196, 768)
        # features: (B, 512, H/8, W/8); (B, 256, H/4, W/4); (B, 64, H/2, W/2)
        x, features = self.transformer(x)
        x = self.decoder(x, features)
        output = self.segmentation_head(x)
        return output


if __name__ == "__main__":
    tensor = torch.randn(1, 1, 224, 224)
    model = TransUnet(2, 3)
    output = model(tensor)
    print(output.shape)
    print("finish")
