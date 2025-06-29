from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker
from torch import Tensor
import torch

@jaxtyped(typechecker=typechecker)
def f(x: Float[Tensor, "1"]) -> Float[Tensor, "1"]:
    return x

print(f(torch.zeros(1)))

T = Float[Tensor, "1"]
print(T)