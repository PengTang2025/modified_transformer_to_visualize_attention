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

完整代码与训练逻辑可查[transformer_copy project](https://github.com/PengTang2025/transformer_copy)

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
model.eval() # 此处不可同时使用model.eval()和with torch.no_grad()，详见下文注意事项
plot_attention_weights(model, ...)
```
## ⚠️ 注意事项：针对 Encoder 注意力提取
（截至 PyTorch 2.7）关于 eval() 与 no_grad() 的使用，请特别注意：  
❗ 若同时启用 with torch.no_grad() 和 model.eval()，Encoder 的注意力提取将失效！  
这是由于 PyTorch 在 TransformerEncoderLayer 中启用了稀疏计算路径，当符合条件时会直接return torch._transformer_encoder_layer_fwd() 结束 forward(), 跳过后续的 _sa_block() 调用，导致 self.attn_weights = None。（详见 why_not_sparsity_fast_path 变量的一系列逻辑，它是 PyTorch Transformer 模块内部用于控制是否使用稀疏计算路径的标志，在 forward() 函数的最前端）  
TransformerDecoderLayer 无稀疏计算路径，不受此限制。
当with torch.no_grad() 和 model.eval()只能选其一时，鉴于我们更希望在测试与绘图中不使用dropout/batchnorm，建议在使用时先完成最终模型的保存，确保可视化使用的数据不再更新模型，再在 eval() 模式下进行注意力可视化绘图。


## 📜 License
MIT License. © 2025 PengTang
