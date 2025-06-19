from typing import Any, Optional, List, Tuple
from mypy.plugin import FunctionContext
from mypy.checker import TypeChecker
from mypy.types import Instance, TupleType, Type, UnboundType, LiteralType, EllipsisType, RawExpressionType

from myshaping.type_translator import construct_instance
from myshaping.function_helper import transpose_funcargs
from myshaping.registry import register_function_hook


dtype_mapper = {
    "float32": "Float32",
    "float": "Float32",
    "float64": "Float64",
    "double": "Float64",
    "complex64": "Complex64",
    "cfloat": "Complex64",
    "complex128": "Complex128",
    "cdouble": "Complex128",
    "float16": "Float16",
    "half": "Float16",
    "bfloat16": "BFloat16",
    "uint8": "UInt8",
    "int8": "Int8",
    "int16": "Int16",
    "short": "Int16",
    "int32": "Int32",
    "int": "Int32",
    "int64": "Int64",
    "long": "Int64",
    "bool": "Bool",
}

@register_function_hook(
    "torch.randn",
    "torch.rand",
    "torch.randint",
    "torch.zeros",
    "torch.ones",
    "torch.empty",
    "torch.full",
)
def construct_from_shape(ctx: FunctionContext):
    if not isinstance(ctx.api, TypeChecker):
        return ctx.default_return_type
    ctxdict = transpose_funcargs(ctx)
    if "size" not in ctxdict:
        return ctx.default_return_type

    args = ctxdict["size"].arg_type
    dimensions: List[Type] = []
    if len(args) == 1 and isinstance(args[0], TupleType):
        dimensions.extend(args[0].items)
    else:
        dimensions.extend(args)
    if all((
        isinstance(dim, Instance) and
        dim.last_known_value is not None and
        type(dim.last_known_value.value) is int
    ) for dim in dimensions):
        # All dimensions are static integers
        shape_str = " ".join(str(dim.last_known_value.value) for dim in dimensions)
        if "dtype" in ctxdict:
            dtype = ctxdict["dtype"]
            dtype_argtype = dtype.arg_type[0]
            if isinstance(dtype_argtype, Instance) and dtype_argtype.type.fullname in ["torch.dtype"]:
                jaxtype = dtype_mapper.get(dtype.arg[0].name, None)
                if jaxtype is None:
                    ctx.api.fail(
                        f"Unsupported dtype {ctxdict['args'][0].name} for torch function.",
                        ctx.context
                    )
                    return ctx.default_return_type
                return construct_instance(
                    ctx.api,
                    jaxtype,
                    ctx.api.named_type("torch.Tensor"),
                    shape_str
                )
            else:
                ctx.api.fail(
                    f"Unsupported dtype {dtype_argtype} for torch function.",
                    ctx.context
                )
                return ctx.default_return_type
        return construct_instance(
            ctx.api,
            "Float32",
            ctx.api.named_type("torch.Tensor"),
            shape_str
        )
    
    return ctx.default_return_type
