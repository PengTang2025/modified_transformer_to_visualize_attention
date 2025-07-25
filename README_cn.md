# Transformer Attention Visualization

本项目旨在**便捷提取 Transformer 模型中的注意力权重（Attention Weights）用于可视化与分析**。为此，我自定义了两个基于 PyTorch 的派生类，分别用于 Encoder 和 Decoder 层，扩展其注意力提取能力。

## ✨ 功能特点

- ✅ 支持提取 TransformerEncoderLayer 每层的 self-attention 权重
- ✅ 支持提取 TransformerDecoderLayer 中的：
  - masked self-attention（自回归）
  - cross-attention（跨序列对齐）
- ✅ 不破坏 PyTorch 原有结构与 forward 逻辑，兼容性好
- ✅ 支持多头注意力（Multi-Head Attention）可视化

## 🧩 实现方式

由于 PyTorch 官方的 `nn.TransformerEncoderLayer` 和 `nn.TransformerDecoderLayer` 默认 **不返回注意力权重**，本项目通过**继承方式**重写了 `_sa_block()`（self-attention）和 `_mha_block()`（multi-head/cross attention）等内部方法，从而：

- 保留原始 forward 接口；
- 在每次前向传播时自动保存 `attn_weights` 到类属性中。

这样可直接通过 `.attn_weights` 访问每层注意力矩阵。


## 🔍 示例

完整代码与训练逻辑可查transformer_copy project

URL:https://github.com/PengTang2025/transformer_copy

```python

from coderlayer_with_attn import TransformerEncoderLayerWithAttn, TransformerDecoderLayerWithAttn

class TransformerXXXModel(nn.Module):
  def __init__(...):
    ...
    self.last_attn = None
    ...
  ...
  encoder_layer = TransformerEncoderLayerWithAttn(...)
  self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
  ...

  def forward(...):
    self.last_attn = self.encoder.layers[-1].attn_weights
    ...

#visualize
def plot_attention_weights(model, sample_src, sample_input):
    _ = model(sample_src, sample_input)  
    attn_weights = model.last_attn  
    ...

# train
model = TransformerXXXModel(...)
...
plot_attention_weights(model, ...)

## 📜 License
MIT License. © 2025 PengTang
