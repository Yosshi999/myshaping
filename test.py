from typing import Literal, Any
from jaxtyping import Float, Float32
from torch import Tensor
import torch

from myshaping import reveal_jaxtype

def f(x: Float32[Tensor, "3 224 224"]): ...
def g(x: Float[Tensor, "1 224 224"]): ...

x = torch.randn(1, 224, 224, dtype=torch.float32)
reveal_jaxtype(x)  # Float32[Tensor, "1 224 224"]

y = torch.ones(1, 224, 224, dtype=torch.int)
reveal_jaxtype(y)  # Int32[Tensor, "1 224 224"]

x = x + x  # safe
x = -x  # safe (TODO: unsafe for UINT)

x2 = torch.randn(3, 224, 224, dtype=torch.float32)
x3 = torch.randn(1, 10, 10, dtype=torch.float32)
z = x + x2  # safe: broadcastable
reveal_jaxtype(z)  # Float32[Tensor, "3 224 224"]
w = x + x3  # fail: shape mismatch

x32 = torch.zeros(1, dtype=torch.float32)
x64 = torch.zeros(1, dtype=torch.float64)

reveal_jaxtype(x32)  # Float32[Tensor, "1"]
reveal_jaxtype(x64)  # Float64[Tensor, "1"]
reveal_jaxtype(x64.float())  # Float32[Tensor, "1"]

x64 += x32  # safe type promotion
x32 += x64  # unsafe: implicit type conversion
x32 += x64.float()  # safe

f(torch.randn(3, 224, 224))  # Correct usage
f(torch.randn(1, 224, 224))  # Incorrect usage, should be (3, 224, 224)

g(torch.randn(3, 224, 224))  # Correct usage