from typing import Any, Optional, List, Tuple
import re
from mypy.plugin import Plugin, FunctionContext, AnalyzeTypeContext
from mypy.types import Instance, TupleType, Type, UnboundType, LiteralType, EllipsisType, RawExpressionType
from mypy.checker import TypeChecker

from myshaping.type_translator import construct_instance
from myshaping.function_helper import transpose_funcargs


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
    # breakpoint()
    # (Pdb) p ctx.context.callee.node.type.arg_types[0]
    # jaxtyping._array_types.Float64[Tensor?, Literal['3 224 224']]
    return ctx.default_return_type

def torch_function_hook(ctx: FunctionContext):
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

def plugin(version: str):
    return ShapePlugin