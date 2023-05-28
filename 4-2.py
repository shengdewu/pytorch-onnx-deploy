import torch
import torchvision
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, 3)
        self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3)
        return

    def forward(self, x):
        return self.conv2(x, self.conv1(x))


@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "none")
def symbolic(g,
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask):
    return g.op("custom::deform_conv2d", input, offset)


if __name__ == '__main__':
    import torch.utils
    v = torch.utils.cmake_prefix_path

    register_custom_op_symbolic("torchvision::deform_conv2d", symbolic, 9)

    model = Model()
    input = torch.rand(1, 3, 10, 10)
    torch.onnx.export(model, input, 'dcn.onnx')
