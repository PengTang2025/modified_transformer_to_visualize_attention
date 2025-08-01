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

Full code and training logic can be found in the [transformer_copy project](https://github.com/PengTang2025/transformer_copy) and [transformer seq2seq with piglatin](https://github.com/PengTang2025/TransformerSeq2Seq-CopyTask-with-AttentionVis-CustomPigLatin)

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
model.eval() # do not use model.eval() and with torch.no_grad() at the same time, the reason is explained in Important Note
plot_attention_weights(model, ...)
```
## ⚠️ Important Note: Attention Extraction from the Encoder (PyTorch 2.7)
  
❗ If you enable both `model.eval()` and `with torch.no_grad()` **at the same time**, attention extraction from the Encoder will **fail**!  
  
This happens because PyTorch introduces **a sparse computation** path in `TransformerEncoderLayer`. When certain conditions are met (e.g., inference mode, specific shapes, no gradient tracking), the forward method will **directly return** via `torch._transformer_encoder_layer_fwd()`, skipping the`_sa_block()` logic where attention weights (self.attn_weights) would have been saved.  
  
This behavior is controlled by the internal flag `why_not_sparsity_fast_path`, which determines whether the sparse path is allowed. This check happens **at the very beginning** of the `forward()` function, before any of your custom logic runs.  
  
In contrast, `TransformerDecoderLayer` does not use this sparse computation path and is not affected by this issue.    
  
✅ Recommendation:  
Since we usually want dropout and batchnorm to be disabled during testing and visualization, it is recommended to:  
- First, run your model training and save the final model. This ensures that the data used for visualization will not cause further updates to the model.  
- Then, visualize attention using **only** `model.eval()`, without wrapping the forward pass in `torch.no_grad()`.  
  
This ensures that attention weights are properly recorded for visualization purposes.  
  
💡 Memory impact is negligible as attention visualization uses small inputs and no backpropagation is involved.  

## 📜 License
MIT License. © 2025 PengTang
