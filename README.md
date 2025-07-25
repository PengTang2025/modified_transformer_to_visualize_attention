# Transformer Attention Visualization
中文版：[README_cn.md](https://github.com/PengTang2025/transformer_customized/blob/main/README_cn.md)

This project aims to **facilitate the extraction and visualization of attention weights from Transformer models**. To achieve this, two PyTorch-based subclasses were implemented—one for the Encoder layer and one for the Decoder layer—extending their ability to expose internal attention matrices.

## ✨ Features

- ✅ Extract self-attention weights from each `TransformerEncoderLayer`
- ✅ Extract both types of attention from `TransformerDecoderLayer`:
  - Masked self-attention (for autoregression)
  - Cross-attention (for sequence alignment)
- ✅ Preserves PyTorch's native structure and `forward()` logic
- ✅ Supports multi-head attention visualization

## 🧩 Implementation Details

By default, PyTorch's `nn.TransformerEncoderLayer` and `nn.TransformerDecoderLayer` **do not return attention weights**.  
This project overcomes that by **inheriting and overriding internal methods** such as `_sa_block()` (for self-attention) and `_mha_block()` (for cross/multi-head attention):

- Keeps the original `forward()` interface unchanged  
- Automatically stores `attn_weights` as a class attribute after each forward pass

This allows direct access to per-layer attention matrices via `.attn_weights`.

## 🔍 Example

Full code and training logic can be found in the [transformer_copy project](https://github.com/PengTang2025/transformer_copy)

```python
from coderlayer_with_attn import TransformerEncoderLayerWithAttn, TransformerDecoderLayerWithAttn

class TransformerXXXModel(nn.Module):
    def __init__(...):
        ...
        self.last_attn = None
        ...
        encoder_layer = TransformerEncoderLayerWithAttn(...)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        ...

    def forward(...):
        self.last_attn = self.encoder.layers[-1].attn_weights
        ...

# Visualize attention
def plot_attention_weights(model, sample_src, sample_input):
    _ = model(sample_src, sample_input)
    attn_weights = model.last_attn
    ...

# Training
model = TransformerXXXModel(...)
...
plot_attention_weights(model, ...)
```

## 📜 License
MIT License. © 2025 PengTang
