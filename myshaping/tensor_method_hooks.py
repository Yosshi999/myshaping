from typing import Any, Optional, List, Tuple
from mypy.plugin import MethodContext
from mypy.checker import TypeChecker
from mypy.types import Instance, TupleType, Type, UnboundType, LiteralType, EllipsisType, RawExpressionType

from myshaping.type_translator import construct_instance
from myshaping.function_helper import transpose_funcargs
from myshaping.registry import register_method_hook

base_array_types = [
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
]

# Possibly implicit type promotions
binary_operators = [
    "__add__", "__radd__", "__iadd__",
    "__sub__", "__rsub__", "__isub__",
    "__mul__", "__rmul__", "__imul__",
    "__matmul__",  # TODO: how?
    "__pow__", "__ipow__",
    "__div__", "__truediv__", "__rdiv__", "__rtruediv__", "__idiv__",
    "__mod__",
    "__eq__", "__ne__",
    "__lt__", "__le__", "__gt__", "__ge__",
]

@register_method_hook(
    *[f"{arr}.{binop}" for arr in base_array_types for binop in binary_operators]
)
def handle_binary_operations(ctx: MethodContext) -> Type:
    ctxdict = transpose_funcargs(ctx)
    return ctx.default_return_type
