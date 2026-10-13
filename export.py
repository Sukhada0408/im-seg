import torch
from fast_scnn import FastSCNN

NUM_CLASSES = 19

# Build the architecture
model = FastSCNN(num_classes=NUM_CLASSES, aux=False)

# Load the trained weights
state_dict = torch.load('fast_scnn_citys.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

class FastSCNNWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)[0]  # take only the main output, drop the tuple wrapper

wrapped = FastSCNNWrapper(model)
wrapped.eval()

# Dummy input matching real usage: batch=1, 3 channels, HxW
dummy_input = torch.randn(1, 3, 384, 768)

# Export model
torch.onnx.export(
    wrapped,
    dummy_input,
    "fast_scnn_citys.onnx",
    export_params=True,
    opset_version=17,
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
