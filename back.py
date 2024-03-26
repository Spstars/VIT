import numpy as np
import torch
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from scipy.interpolate import interpn


def make_custom_vit(img_size=(800, 1280)):
    # base model: Vit_B_16
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    # extract positional embedding of ViT_B_16 canonical model
    pos_embedding = model.encoder.pos_embedding.detach().numpy()

    # reconstruct the 2D structure of positional embedding
    embed = pos_embedding[:, 1:, :]  # exclude class token embedding
    # reshape to (H_patch, W_patch, D)
    embed = np.reshape(embed, (14, 14, pos_embedding.shape[-1]))

    # compute new (H_patch, W_patch) to match resolution
    H_SRC, W_SRC = 14, 14
    H_DST, W_DST = img_size[0] // 16, img_size[1] // 16

    # 2D grid of reference points for interpolation
    src_grid = np.linspace(0, 1, num=H_SRC), np.linspace(0, 1, num=W_SRC)
    dst_grid = np.stack(np.meshgrid(np.linspace(0, 1, num=H_DST),
                        np.linspace(0, 1, num=W_DST), indexing='ij'), axis=-1)

    # linear interpolation of pos_embedding: (14, 14) -> (H_DST, W_DST)
    new_embed = interpn(points=src_grid, values=embed,
                        xi=dst_grid).astype('float32')

    # flatten and merge with class token
    new_embed = np.reshape(new_embed, (1, -1, pos_embedding.shape[-1]))
    new_pos_embedding = np.concatenate(
        [pos_embedding[:, 0:1, :], new_embed], axis=1)

    # inject new positional embedding to the model
    model.encoder.pos_embedding.data = torch.tensor(new_pos_embedding)

    # modify input preprocessor to ignore input size check
    model._process_input = _modified_process_input.__get__(model)

    return model


def _modified_process_input(self, x: torch.Tensor) -> torch.Tensor:
    '''modified version of VisionTransformer._process_input()'''
    n, c, h, w = x.shape
    p = self.patch_size
    
    '''
    NOTE commented out block
    torch._assert(
        h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
    torch._assert(
        w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
    '''
    
    n_h = h // p
    n_w = w // p

    # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
    x = self.conv_proj(x)
    # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
    x = x.reshape(n, self.hidden_dim, n_h * n_w)

    # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
    # The self attention layer expects inputs in the format (N, S, E)
    # where S is the source sequence length, N is the batch size, E is the
    # embedding dimension
    x = x.permute(0, 2, 1)

    return x
