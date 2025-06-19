from typing import Any, Optional, List, Tuple
import re
from mypy.plugin import Plugin, FunctionContext, AnalyzeTypeContext
from mypy.types import Instance, TupleType, Type, UnboundType, LiteralType, EllipsisType, RawExpressionType
from mypy.checker import TypeChecker

from myshaping.type_translator import construct_instance
from myshaping.registry import register_type_analyze_hook, get_function_hook, get_type_analyze_hook, get_method_hook
import myshaping.torch_function_hooks
import myshaping.tensor_method_hooks


@register_type_analyze_hook(
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
)
def analyze_jaxtyping(ctx: AnalyzeTypeContext):
    """Parse Dtype[Array, "shape"] to the mypy-friendly type, because raw string in the Generic is not allowed."""
    dtype = ctx.type.name
    if len(ctx.type.args) != 2:
        return ctx.type
    backend, shape = ctx.type.args
    if not isinstance(shape, RawExpressionType) or shape.literal_value is None:
        return ctx.type
    backend = ctx.api.analyze_type(backend)
    dim_str = shape.literal_value
    print(dtype, backend, dim_str)
    return construct_instance(ctx.api, dtype, backend, dim_str)


def hook(ctx: FunctionContext):
    # breakpoint()
    # (Pdb) p ctx.context.callee.node.type.arg_types[0]
    # jaxtyping._array_types.Float64[Tensor?, Literal['3 224 224']]
    return ctx.default_return_type

class ShapePlugin(Plugin):
    def get_type_analyze_hook(self, fullname: str):
        return get_type_analyze_hook(fullname)

    def get_function_hook(self, fullname: str):
        if fullname.endswith(".f"):
            return hook
        return get_function_hook(fullname)
    
    def get_method_hook(self, fullname: str):
        return get_method_hook(fullname)

def plugin(version: str):
    return ShapePlugin