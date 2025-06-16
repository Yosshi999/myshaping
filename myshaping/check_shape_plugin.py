from typing import Any, Optional, List, Tuple
import re
from mypy.plugin import Plugin, FunctionContext, AnalyzeTypeContext
from mypy.types import Instance, TupleType, Type, UnboundType, LiteralType


# class Dim:
#     def is_static(self) -> bool:
#         return True
#     @property
#     def value(self) -> Any:
#         return None

# class IntDim(Dim):
#     def __init__(self, value: int):
#         self._value = value
#     def is_static(self) -> bool:
#         return True
#     @property
#     def value(self) -> int:
#         return self._value

# class SymbolicDim(Dim):
#     def __init__(self, name: str):
#         self._name = name
#     def is_static(self) -> bool:
#         return False
#     @property
#     def value(self) -> str:
#         return self._name

# class UnknownDim(Dim):
#     def is_static(self) -> bool:
#         return False
#     @property
#     def value(self):
#         return None

# Shape = List[Dim]
# TensorInfo = Tuple[str, Shape]  # (dtype, shape)

# def parse_shape(shape_str: str) -> Shape:
#     symbols = shape_str.split()
#     parsed_symbols: List[Dim] = []
#     for s in symbols:
#         s = re.sub(r'.*=', '', s)  # Remove any assignments like "x=3"
#         if s.isdigit():
#             parsed_symbols.append(IntDim(int(s)))
#         else:
#             if re.match(r'^_\w*$', s):
#                 parsed_symbols.append(UnknownDim())
#             elif re.match(r'^[a-zA-Z_]\w*$', s):
#                 parsed_symbols.append(SymbolicDim(s))
#             elif re.match(r'^\*[a-zA-Z_]\w*$', s):
#                 # multiple axis not supported
#                 raise NotImplementedError
#             elif re.match(r'^\#[a-zA-Z_]\w*$', s):
#                 # broadcastable axis not supported
#                 parsed_symbols.append(SymbolicDim(s[1:]))  # Remove leading '#'
#             else:
#                 # Handle unexpected formats
#                 raise ValueError(f"Unexpected shape format: {s}")
#     return parsed_symbols


class ShapePlugin(Plugin):
    jaxtyping_names = set([
        "jaxtyping._array_types.UInt2",
        "jaxtyping._array_types.UInt4",
        "jaxtyping._array_types.UInt8",
        "jaxtyping._array_types.UInt16",
        "jaxtyping._array_types.UInt32",
        "jaxtyping._array_types.UInt64",
        "jaxtyping._array_types.Int2",
        "jaxtyping._array_types.Int4",
        "jaxtyping._array_types.Int8",
        "jaxtyping._array_types.Int16",
        "jaxtyping._array_types.Int32",
        "jaxtyping._array_types.Int64",
        "jaxtyping._array_types.Float8e4m3b11fnuz",
        "jaxtyping._array_types.Float8e4m3fn",
        "jaxtyping._array_types.Float8e4m3fnuz",
        "jaxtyping._array_types.Float8e5m2",
        "jaxtyping._array_types.Float8e5m2fnuz",
        "jaxtyping._array_types.BFloat16",
        "jaxtyping._array_types.Float16",
        "jaxtyping._array_types.Float32",
        "jaxtyping._array_types.Float64",
        "jaxtyping._array_types.Complex64",
        "jaxtyping._array_types.Complex128",
        "jaxtyping._array_types.Bool",
        "jaxtyping._array_types.UInt",
        "jaxtyping._array_types.Int",
        "jaxtyping._array_types.Integer",
        "jaxtyping._array_types.Float",
        "jaxtyping._array_types.Complex",
        "jaxtyping._array_types.Inexact",
        "jaxtyping._array_types.Real",
        "jaxtyping._array_types.Num",
        "jaxtyping._array_types.Shaped",
        "jaxtyping._array_types.Key",
    ])
    def get_type_analyze_hook(self, fullname: str):
        if fullname in self.jaxtyping_names:
            print("Detected jaxtyping type annotation:", fullname)
            return analyze_jaxtyping
        return None

    def get_function_hook(self, fullname: str):
        if fullname.startswith("torch.randn"):
            return torch_function_hook
        if fullname.endswith(".f"):
            return hook
        return None

def hook(ctx: FunctionContext):
    breakpoint()
    # (Pdb) p ctx.context.callee.node.type.arg_types[0]
    # jaxtyping._array_types.Float64[Tensor?, Literal['3 224 224']]
    return ctx.default_return_type

def torch_function_hook(ctx: FunctionContext):
    args = ctx.arg_types[0]
    dimensions: List[Type] = []
    if len(args) == 1 and isinstance(args[0], TupleType):
        dimensions.extend(args[0].items)
    elif len(args) > 1:
        dimensions.extend(args)
    if all((
        isinstance(dim, Instance) and
        dim.last_known_value is not None and
        type(dim.last_known_value.value) is int
    ) for dim in dimensions):
        # All dimensions are static integers
        shape_str = " ".join(str(dim.last_known_value.value) for dim in dimensions)
        # return UnboundType(
        #     "Float",
        #     [ctx.default_return_type, LiteralType(value=shape_str, fallback=ctx.api.named_type("str"))]
        # )
        return Instance(
            ctx.api.named_type("jaxtyping.Float").type,
            [ctx.default_return_type, LiteralType(value=shape_str, fallback=ctx.api.named_type("builtins.str"))]
        )
    
    return ctx.default_return_type

def analyze_jaxtyping(ctx: AnalyzeTypeContext):
    """Rename Dtype[Array, "shape"] to Dtype[Array, Literal["shape"]], because raw string in the Generic is not allowed."""
    try:
        dtype = ctx.type.name
        backend, shape = ctx.type.args
        return Instance(
            ctx.api.named_type("jaxtyping." + dtype).type,
            [backend, LiteralType(value=shape.literal_value, fallback=ctx.api.named_type("builtins.str"))]
        )
    except Exception as e:
        print(e)
        return ctx.type

def plugin(version: str):
    return ShapePlugin