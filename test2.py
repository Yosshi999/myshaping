from jaxtyping import Float
from torch import Tensor

x: Float[Tensor, "3 224 224"]
reveal_type(x)