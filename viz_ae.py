# viz_ae.py
import torch
from torchviz import make_dot
from model import ConvAutoencoder_v2

model = ConvAutoencoder_v2()
dummy_input = torch.randn(1, 3, 256, 256)
output = model(dummy_input)

dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = "png"
dot.render("conv_autoencoder_architecture")
print("✅ Đã lưu sơ đồ kiến trúc: conv_autoencoder_architecture.png")

