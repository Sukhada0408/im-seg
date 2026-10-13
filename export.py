import torch
from fast_scnn import FastSCNN

NUM_CLASSES = 19  # Cityscapes ('citys') — from data_loader.datasets['citys'].NUM_CLASS

# 1. Build the architecture (aux=False — we don't need the auxiliary head for inference)
model = FastSCNN(num_classes=NUM_CLASSES, aux=False)

# 2. Load the trained weights
state_dict = torch.load('fast_scnn_citys.pth', map_location='cpu')
model.load_state_dict(state_dict)

# 3. CRITICAL: eval mode
# BatchNorm and Dropout behave differently in train vs eval.
# In train mode, BatchNorm uses batch statistics; in eval mode it uses the
# running mean/var learned during training. If you forget this, your ONNX
# model's outputs will be wrong/noisy even though the export "succeeds" silently.
model.eval()

# 4. Wrap the model so forward() returns a single Tensor, not a tuple
class FastSCNNWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)[0]  # take only the main output, drop the tuple wrapper

wrapped = FastSCNNWrapper(model)
wrapped.eval()

# 5. Dummy input matching real usage: batch=1, 3 channels, HxW
# Use a size close to what you'll feed at inference time (e.g. 512x1024 is
# Cityscapes' native aspect ratio downscaled; 256x512 matches their test script)
dummy_input = torch.randn(1, 3, 384, 768)

# 6. Export
torch.onnx.export(
    wrapped,
    dummy_input,
    "fast_scnn_citys.onnx",
    export_params=True,
    opset_version=17,              # 17 is well-supported by recent ONNX Runtime; use 11 if you hit ops issues
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size",},
        "output": {0: "batch_size",},
    },
    dynamo=False,
)
print("Exported to fast_scnn_citys.onnx")