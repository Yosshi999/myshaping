from typing import Literal
from jaxtyping import Float, Float64
from torch import Tensor
import torch

def f(x: Float64[Tensor, "3 224 224"]): ...
def g(x: Float[Tensor, "1 224 224"]): ...

x = torch.randn(1, 224, 224)
reveal_type(x)  # Should show Float[Tensor, "1 224 224"]
f(torch.randn(3, 224, 224))  # Correct usage
# g(torch.randn((3, 224, 224)))  # Incorrect usage, should be (1, 224, 224)
