# onnx自定义算子（pytorch算子转onnx算子）

### 支持Aten算子
*算子在Aten中实现了，onnx中也有相关算子的定义，单是相关算子映射成onnx的规则没写 ，只需要为Aten算子补充描述映射规则的符号函数*  
+ 获取Aten中算子接口定义
> 为了编写符号函数，我们需要获得 asinh 推理接口的输入参数定义。
> 这时，我们要去 torch/_C/_VariableFunctions.pyi 和 torch/nn/functional.pyi 这两个文件中搜索我们刚刚得到的这个算子名。
> 这两个文件是编译 PyTorch 时本地自动生成的文件，里面包含了 ATen 算子的 PyTorch 调用接口。
> 通过搜索，我们可以知道 asinh 在文件 torch/_C/_VariableFunctions.pyi 中，其接口定义为

```python
def asinh(input: Tensor, *, out: Optional[Tensor]=None) -> Tensor: ...
```  
> 经过这些步骤，我们确认了缺失的算子名为 asinh，它是一个有实现的 ATen 算子。我们还记下了 asinh 的调用接口。接下来，我们要为它补充符号函数，
> 使它在转换成 ONNX 模型时不再报错  
+ 添加符号函数   
*符号函数，可以看成是 PyTorch 算子类的一个静态方法。在把 PyTorch 模型转换成 ONNX 模型时，各个 PyTorch 算子的符号函数会被依次调用，以完成 PyTorch 算子到 ONNX 算子的转换* 
> 符号函数的定义一般如下：
```python
def symbolic(g: torch._C.Graph, input_0: torch._C.Value, input_1: torch._C.Value, ...):
```
> 其中，torch._C.Graph 和 torch._C.Value 都对应 PyTorch 的 C++ 实现里的一些类。
> 我们在这篇文章不深究它们的细节（感兴趣的话可以参考我们的 TorchScript 系列文章中对 trace 机制的解读），
> 只需要知道第一个参数就固定叫 g，它表示和计算图相关的内容；后面的每个参数都表示算子的输入，需要和算子的前向推理接口的输入相同。
> 对于 ATen 算子来说，它们的前向推理接口就是上述两个 .pyi 文件里的函数接口  
> g 有一个方法 op。在把 PyTorch 算子转换成 ONNX 算子时，需要在符号函数中调用此方法来为最终的计算图添加一个 ONNX 算子。其定义如下
```python
def op(name: str, input_0: torch._C.Value, input_1: torch._C.Value, ...)
    """
    name: 算子名称。如果该算子是普通的 ONNX 算子，只需要把它在 ONNX 官方文档里的名称填进去即可（我们稍后再讲其他情况）
    """  
```  
> 在最简单的情况下，我们只要把 PyTorch 算子的输入用g.op()一一对应到 ONNX 算子上即可，并把g.op()的返回值作为符号函数的返回值。
> 在情况更复杂时，我们转换一个 PyTorch 算子可能要新建若干个 ONNX 算子。  

+ 查看 Asinh 在onnx文档（https://github.com/onnx/onnx/blob/main/docs/Operators.md#asinh）的说明
+ 根据 Asinh在pytoch的接口定义补充映射关系
```python
from torch.onnx.symbolic_registry import register_op

def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)

register_op('asinh', asinh_symbolic, '', 9)
```  
>这里的asinh_symbolic就是asinh的符号函数。从除g以外的第二个输入参数开始，其输入参数应该严格对应它在 ATen 中的定义：
```python
def asinh(input: Tensor, *, out: Optional[Tensor]=None) -> Tensor: ...
```
> 在符号函数的函数体中，
> g.op("Asinh", input)则完成了 ONNX 算子的定义。
> 第一个参数"Asinh"是算子在 ONNX 中的名称。
> 至于第二个参数 input，如我们刚刚在文档里所见，这个算子只有一个输入，因此我们只要把符号函数的输入参数 input 对应过去就行。
> ONNX 的 Asinh 的输出和 ATen 的 asinh 的输出是一致的，因此我们直接把 g.op() 的结果返回即可  

+ 绑定
> 我们要把这个符号函数和原来的 ATen 算子“绑定”起来。这里，我们要用到 register_op 这个 PyTorch API 来完成绑定。
> 如示例所示，只需要一行简单的代码即可把符号函数 asinh_symbolic 绑定到算子 asinh 上:
```python
register_op('asinh', asinh_symbolic, '', 9)
"""
第一个参数是目标 ATen 算子名
第二个是要注册的符号函数
第三个参数是算子的“域” 对于普通 ONNX 算子，直接填空字符串即可
第四个参数表示向哪个算子集版本注册 值得注意的是，这里向第 9 号算子集注册，不代表较新的算子集（第 10 号、第 11 号……）都得到了注册
"""
```  
+ 测试
> 成功导出的话，asinh.onnx
```python
import onnxruntime
import torch
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x)

model = Model()
input = torch.rand(1, 3, 10, 10)
torch_output = model(input).detach().numpy()

sess = onnxruntime.InferenceSession('asinh.onnx')
ort_output = sess.run(None, {'0': input.numpy()})[0]

assert np.allclose(torch_output, ort_output)
```  
### 支持TorchScript算子  
*注意：可以使用自定义算法，本例采用pytorch自带的 torchvision.ops.DeformConv2d*  
*注意：自定义算法参见 https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html*
  
##### 为算子添加符号函数
1. 获取原算子的前向推理接口  
2. 获取目标 ONNX 算子的定义  
3. 编写符号函数并绑定  

+ 定义包含算子的模型  
```python
import torch
import torchvision

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, 3)
        self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3)

    def forward(self, x):
        return self.conv2(x, self.conv1(x))
```

+ 确定算子调用接口  
```python
#torchvision/csrc/ops/deform_conv2d.cpp
 m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::deform_conv2d(Tensor input, 
                                  Tensor weight, 
                                  Tensor offset, 
                                  Tensor mask,
                                  Tensor bias, 
                                  int stride_h, 
                                  int stride_w, 
                                  int pad_h, 
                                  int pad_w, 
                                  int dilation_h, 
                                  int dilation_w, 
                                  int groups, 
                                  int offset_groups,
                                  bool use_mask) -> Tensor"));
```
+ 自定义onnx算子  
> 如果我们去 ONNX 的官方算子页面搜索 "deform"，将搜不出任何内容。
> 目前，ONNX 还没有提供可变形卷积的算子，我们要自己定义一个 ONNX 算子了  
> 1. g.op() 是用来定义 ONNX 算子的函数  
> 2. 对于 ONNX 官方定义的算子，g.op() 的第一个参数就是该算子的名称  
> 3. 而对于一个自定义算子，g.op() 的第一个参数是一个带命名空间的算子名
```python
g.op("custom::deform_conv2d, ...)
"""
"::"前面的内容就是我们的命名空间,是为了防止命名冲突而设定的, 不加前面的命名空间，则算子会被默认成 ONNX 的官方算子
"""
```
*注意：ONNX 是一套标准，本身不包括实现*  
*所以我们就简略地定义一个 ONNX 可变形卷积算子，而不去写它在某个推理引擎上的实现* 
```python
@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "none")
"""
"v"表示 Torch 库里的 value 类型，一般用于标注张量，
"i"表示 int 类型，
"f"表示 float 类型，
"none"表示该参数为空
具体含义参见 https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_helper.py
"""
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
```  
+ 注册
```python
register_custom_op_symbolic("torchvision::deform_conv2d", symbolic, 9)
# 于前面的 register_op 类似，不同的是 这里的算子集版本是最早生效版本，在这里设定版本 9，意味着之后的第 10 号、第 11 号……版本集都能使用这个新算子
```

### 使用 torch.autograd.Function  
*注意相对比较简单，推荐*  
+ 为pytorch 添加 c++扩展  
 *实例 def(a, b) return 2a + b*  
 ```python
// my_add.cpp
#include <torch/torch.h>
torch::Tensor my_add(torch::Tensor a, torch::Tensor b)
{
    return 2 * a + b;
}

"""
PYBIND11_MODULE 来为 C++ 函数提供 Python 调用接口
"""
PYBIND11_MODULE(my_lib, m)
{
    m.def("my_add", my_add);
}
```  
+ 编译源文件 
> 通过 setup.py 编译
```python
from setuptools import setup
from torch.utils import cpp_extension

setup(name='my_add',
      ext_modules=[cpp_extension.CppExtension('my_lib', ['my_add.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
```  
> 执行命令 python setup.py develop  
+ 用torch.autograd.Function 封装  
```python
import torch
import my_lib
class MyAddFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        return my_lib.my_add(a, b)

    @staticmethod
    def symbolic(g, a, b):
        two = g.op("Constant", value_t=torch.tensor([2]))
        a = g.op('Mul', a, two)
        return g.op('Add', a, b)
```  
> Function  类有一个很好的性质：如果它定义了 symbolic 静态方法，
> 该 Function 在执行 torch.onnx.export() 时就可以根据 symbolic 中定义的规则转换成 ONNX 算子  
+ 调用算子 
```python
my_add = MyAddFunction.apply
# my_add = MyAddFunction.apply 获取了一个奇怪的变量。
# 这个变量是用来做什么的呢？ 
# 其实，apply是torch.autograd.Function 的一个方法，
# 这个方法完成了 Function 在前向推理或者反向传播时的调度。
# 我们在使用 Function 的派生类做推理时，不应该显式地调用 forward()，
# 而应该调用其 apply 方法

class MyAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return my_add(a, b)
```  
+ 测试算子  
```python
model = MyAdd()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, (input, input), 'my_add.onnx')
torch_output = model(input, input).detach().numpy()

import onnxruntime
import numpy as np
sess = onnxruntime.InferenceSession('my_add.onnx')
ort_output = sess.run(None, {'a': input.numpy(), 'b': input.numpy()})[0]

assert np.allclose(torch_output, ort_output)
```


