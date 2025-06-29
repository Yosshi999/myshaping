from typing import Any, Optional, List, Tuple
from mypy.plugin import MethodContext
from mypy.checker import TypeChecker
from mypy.types import Instance, TupleType, Type, UnboundType, LiteralType, EllipsisType, RawExpressionType

from myshaping.type_translator import check_shape_compatibility, decompose_instance, parse_dimstr, repr_instance, construct_instance, compare_dtype, dump_dims
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

cast_mapper = {
    "half": "Float16",
    "bfloat16": "BFloat16",
    "float": "Float32",
    "double": "Float64",
    "short": "Int16",
    "int": "Int32",
    "long": "Int64",
}

for op, dtype in cast_mapper.items():
    def handle_cast(ctx: MethodContext) -> Type:
        xtype = ctx.type  # Self
        _, x_backend, x_dimstr = decompose_instance(xtype)
        return construct_instance(ctx.api, dtype, x_backend, x_dimstr)
    register_method_hook(*[f"{arr}.{op}" for arr in base_array_types])(handle_cast)

# Possibly implicit type promotions
binary_promotable = set([
    "__add__", "__radd__",
    "__sub__", "__rsub__",
    "__mul__", "__rmul__",
    "__pow__",
    "__div__", "__rdiv__",
])

binary_comparison = set([
    "__eq__", "__ne__",
    "__lt__", "__le__", "__gt__", "__ge__",
])

# NOTE: "__matmul__"
# NOTE: "__truediv__", "__rtruediv__", "__mod__"

inplace_operators = set([
    "__iadd__", "__isub__", "__imul__", "__ipow__", "__idiv__",
])

@register_method_hook(
    *[f"{arr}.{binop}" for arr in base_array_types for binop in binary_promotable]
)
def handle_binary_promotable(ctx: MethodContext) -> Type:
    xtype = ctx.type  # Self
    ytype = ctx.arg_types[0][0]  # Other
    x_dtype, x_backend, x_dimstr = decompose_instance(xtype)
    y_dtype, y_backend, y_dimstr = decompose_instance(ytype)
    x_shape = parse_dimstr(ctx.api, x_dimstr)
    y_shape = parse_dimstr(ctx.api, y_dimstr)

    # backend check
    if x_backend.type.fullname != y_backend.type.fullname:
        ctx.api.fail(f"Backend mismatch. self: {repr_instance(xtype, ctx.api.msg.options)} vs other: {repr_instance(ytype, ctx.api.msg.options)}", ctx.context)
        # TODO: torch vs numpy is sometimes allowed
        return ctx.default_return_type
    z_backend = x_backend
    
    # type check
    promotion = compare_dtype(x_dtype, y_dtype)
    if promotion is None:
        ctx.api.fail(f"Type mismatch. self: {repr_instance(xtype, ctx.api.msg.options)} vs other: {repr_instance(ytype, ctx.api.msg.options)}", ctx.context)
        return ctx.default_return_type
    z_dtype = y_dtype if promotion == 1 else x_dtype
    if x_dtype != z_dtype:
        ctx.api.msg.note(f"Implicit dtype conversion of self: {x_dtype} -> {z_dtype}", ctx.context)
    if y_dtype != z_dtype:
        ctx.api.msg.note(f"Implicit dtype conversion of other: {y_dtype} -> {z_dtype}", ctx.context)

    # shape check
    z_shape = check_shape_compatibility(x_shape, y_shape, allow_broadcast=True)
    if z_shape is None:
        ctx.api.fail(f"Shape mismatch. self: {repr_instance(xtype, ctx.api.msg.options)} vs other: {repr_instance(ytype, ctx.api.msg.options)}", ctx.context)
        return ctx.default_return_type
    
    return construct_instance(ctx.api, z_dtype, z_backend, dump_dims(z_shape))


@register_method_hook(
    *[f"{arr}.{binop}" for arr in base_array_types for binop in binary_comparison]
)
def handle_comparison(ctx: MethodContext) -> Type:
    xtype = ctx.type  # Self
    ytype = ctx.arg_types[0][0]  # Other
    x_dtype, x_backend, x_dimstr = decompose_instance(xtype)
    y_dtype, y_backend, y_dimstr = decompose_instance(ytype)
    x_shape = parse_dimstr(ctx.api, x_dimstr)
    y_shape = parse_dimstr(ctx.api, y_dimstr)

    # backend check
    if x_backend.type.fullname != y_backend.type.fullname:
        ctx.api.fail(f"Backend mismatch. self: {repr_instance(xtype, ctx.api.msg.options)} vs other: {repr_instance(ytype, ctx.api.msg.options)}", ctx.context)
        # TODO: torch vs numpy is sometimes allowed
        return ctx.default_return_type
    
    # type check
    promotion = compare_dtype(x_dtype, y_dtype)
    if promotion is None:
        ctx.api.fail(f"Type mismatch. self: {repr_instance(xtype, ctx.api.msg.options)} vs other: {repr_instance(ytype, ctx.api.msg.options)}", ctx.context)
        return ctx.default_return_type

    # shape check
    z_shape = check_shape_compatibility(x_shape, y_shape, allow_broadcast=True)
    if z_shape is None:
        ctx.api.fail(f"Shape mismatch. self: {repr_instance(xtype, ctx.api.msg.options)} vs other: {repr_instance(ytype, ctx.api.msg.options)}", ctx.context)
        return ctx.default_return_type
    
    return construct_instance(ctx.api, "Bool", x_backend, dump_dims(z_shape))


@register_method_hook(
    *[f"{arr}.{binop}" for arr in base_array_types for binop in inplace_operators]
)
def handle_inplace(ctx: MethodContext) -> Type:
    xtype = ctx.type  # Self
    ytype = ctx.arg_types[0][0]  # Other
    x_dtype, x_backend, x_dimstr = decompose_instance(xtype)
    y_dtype, y_backend, y_dimstr = decompose_instance(ytype)
    x_shape = parse_dimstr(ctx.api, x_dimstr)
    y_shape = parse_dimstr(ctx.api, y_dimstr)

    # backend check
    if x_backend.type.fullname != y_backend.type.fullname:
        ctx.api.fail(f"Backend mismatch. self: {repr_instance(xtype, ctx.api.msg.options)} vs other: {repr_instance(ytype, ctx.api.msg.options)}", ctx.context)
        # TODO: torch vs numpy is sometimes allowed
        return ctx.default_return_type
    
    # type check
    promotion = compare_dtype(x_dtype, y_dtype)
    if promotion is None:
        ctx.api.fail(f"Type mismatch. self: {repr_instance(xtype, ctx.api.msg.options)} vs other: {repr_instance(ytype, ctx.api.msg.options)}", ctx.context)
        return ctx.default_return_type
    if promotion == 1:
        ctx.api.msg.note(f"Implicit dtype conversion in update: other: {y_dtype} -> self: {x_dtype}", ctx.context)

    # shape check
    z_shape = check_shape_compatibility(x_shape, y_shape, allow_broadcast=True)
    if z_shape is None or dump_dims(x_shape) != dump_dims(z_shape):
        ctx.api.fail(f"Shape mismatch. self: {repr_instance(xtype, ctx.api.msg.options)} vs other: {repr_instance(ytype, ctx.api.msg.options)}", ctx.context)
        return ctx.default_return_type
    
    return construct_instance(ctx.api, x_dtype, x_backend, x_dimstr)
