import torch
import onnxruntime
import numpy as np
from torch.onnx.symbolic_registry import register_op


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)


def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)


if __name__ == '__main__':
    register_op('asinh', asinh_symbolic, '', 9)
    
    model = Model()
    input = torch.rand(1, 3, 10, 10)
    torch.onnx.export(model, input, 'asinh.onnx')

    torch_output = model(input).detach().numpy()
    
    sess = onnxruntime.InferenceSession('asinh.onnx')
    ort_output = sess.run(None, {'0': input.numpy()})[0]
    
    assert np.allclose(torch_output, ort_output)