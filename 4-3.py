import torch
import onnxruntime
import numpy as np
import my_lib


class MyAddFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        return my_lib.my_add(a, b)

    @staticmethod
    def symbolic(g, a, b):
        a = g.op('Mul', a, g.op("Constant", value_t=torch.tensor([2])))
        return g.op('Add', a, b)


my_add = MyAddFunction.apply


class MyAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return my_add(a, b)


if __name__ == '__main__':
    model = MyAdd()
    input = torch.rand(1, 3, 10, 10)
    torch.onnx.export(model, (input, input), 'my_add.onnx')
    torch_output = model(input, input).detach().numpy()

    sess = onnxruntime.InferenceSession('my_add.onnx')
    ort_output = sess.run(None, {'a': input.numpy(), 'b': input.numpy()})[0]
    
    assert np.allclose(torch_output, ort_output)