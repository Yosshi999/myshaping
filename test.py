from typing import Literal, Any
from jaxtyping import Float, Float32
from torch import Tensor
import torch

def f(x: Float32[Tensor, "3 224 224"]): ...
def g(x: Float[Tensor, "1 224 224"]): ...

x = torch.randn(1, 224, 224, dtype=torch.float32)
reveal_type(x)  # Should show Float32[Tensor, "1 224 224"]

y = torch.ones(1, 224, 224, dtype=torch.int)
reveal_type(y)  # Should show Int32[Tensor, "1 224 224"]

f(torch.randn(3, 224, 224))  # Correct usage
f(torch.randn(1, 224, 224))  # Incorrect usage, should be (3, 224, 224)

g(torch.randn(3, 224, 224))  # Correct usage