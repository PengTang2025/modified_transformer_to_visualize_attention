from typing import Optional
import torch
import torch.nn as nn

# 派生 nn.TransformerEncoderLayer 以便添加注意力权重存储
# redefine a new TransformerEncoderLayer that stores attention weights
class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # store attention weights
        self.attn_weights = None  

    def _sa_block(self, x, attn_mask, key_padding_mask):
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # muti-head attention
        )
        self.attn_weights = attn_weights  # store attention weights
        x = x + self.dropout1(attn_output)
        return self.norm1(x)
    # 重写 _sa_block 方法，添加 need_weights=True 和 average_attn_weights=False 参数
    # 这样可以在 forward 时获取注意力权重
    # rewrite the _sa_block method to add need_weights=True and average_attn_weights=False parameters
    # so that attention weights can be obtained during forward pass
    def _sa_block(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            key_padding_mask: Optional[torch.Tensor],
            is_causal: bool = False,
        ) -> torch.Tensor:
            attn_output, attn_weights = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,               # change here
                average_attn_weights=False,      # optional, to keep multi-head attention weights separate
                is_causal=is_causal,
            )
            self.attn_weights = attn_weights    # add a line here to store attention weights
            return self.dropout1(attn_output)