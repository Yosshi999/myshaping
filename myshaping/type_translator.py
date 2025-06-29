"""Translate between jaxtyping annotations and mypy types."""

from dataclasses import dataclass
import enum
from typing import List, Any, Union, Optional
from mypy.types import Instance, TupleType, Type, UnboundType, LiteralType, EllipsisType, RawExpressionType, UnionType, TypeStrVisitor
from mypy.plugin import TypeAnalyzerPluginInterface
from jaxtyping._array_types import _DimType

union_mapper = {
    "UInt": ["UInt2", "UInt4", "UInt8", "UInt16", "UInt32", "UInt64"],
    "Int": ["Int2", "Int4", "Int8", "Int16", "Int32", "Int64"],
    "Integer": ["Int", "UInt"],
    "Float": ["Float8e4m3b11fnuz", "Float8e4m3fn", "Float8e4m3fnuz", "Float8e5m2", "Float8e5m2fnuz", "BFloat16", "Float16", "Float32", "Float64"],
    "Complex": ["Complex64", "Complex128"],
    "Inexact": ["Float", "Complex"],
    "Real": ["Float", "UInt", "Int"],
    "Num": ["Float", "Complex", "UInt", "Int"],
}

"""Order of dtypes.
1. Kind: {bool} < {integral dypes} < {floating point dtypes}
2. Width
3. Prefer Float16: BFloat16 < Float16
4. Prefer Unsigned
"""
dtype_orders = [
    "Bool",
    "Int2", "UInt2",
    "Int4", "UInt4",
    "Int8", "UInt8",
    "Int16", "UInt16",
    "Int32", "UInt32",
    "Int64", "UInt64",
    "BFloat16", "Float16", "Float32", "Float64"
]

class _DimType(enum.Enum):
    named = enum.auto()
    fixed = enum.auto()
    symbolic = enum.auto()

@dataclass(frozen=True)
class AnonymousDim:
    def __repr__(self):
        return "_"
@dataclass(frozen=True)
class AnonymousVariadicDim:
    def __repr__(self):
        return "..."
@dataclass(frozen=True)
class NamedDim:
    name: str
    broadcastable: bool
    def __repr__(self):
        return ("#" if self.broadcastable else "") + self.name
@dataclass(frozen=True)
class NamedVariadicDim:
    name: str
    broadcastable: bool
    def __repr__(self):
        return f"*{self.name}"
@dataclass(frozen=True)
class FixedDim:
    size: int
    def __repr__(self):
        return str(self.size)
@dataclass(frozen=True)
class SymbolicDim:
    elem: Any
    broadcastable: bool
    def __repr__(self):
        return str(self.elem)

AbstractDimOrVariadicDim = Union[
    AnonymousDim,
    AnonymousVariadicDim,
    NamedDim,
    NamedVariadicDim,
    FixedDim,
    SymbolicDim,
]

AbstractDim = Union[
    AnonymousDim,
    NamedDim,
    FixedDim,
    SymbolicDim,
]

VariadicDim = Union[
    AnonymousVariadicDim,
    NamedVariadicDim,
]

def construct_instance(api: TypeAnalyzerPluginInterface, dtype: str, backend: Type, dim_str: str) -> Type:
    """Construct an Instance of a jaxtyping type with the given dtype, backend, and shape."""
    if dtype in union_mapper:
        items = [
            Instance(
                api.named_type(f"jaxtyping.{dtype}").type,
                [backend, LiteralType(value=dim_str, fallback=api.named_type("builtins.str"))]
            )
            for dtype in union_mapper[dtype]
        ]
        return UnionType(items)
    return Instance(
        api.named_type(f"jaxtyping.{dtype}").type,
        [backend, LiteralType(value=dim_str, fallback=api.named_type("builtins.str"))]
    )

def decompose_instance(typ: Instance):
    assert typ.type.fullname.startswith("jaxtyping._array_types")
    dtype = typ.type.fullname.split(".")[-1]
    backend: Instance = typ.args[0]
    shape: LiteralType = typ.args[1]
    return dtype, backend, shape.value

def parse_dimstr(api: TypeAnalyzerPluginInterface, dim_str: str) -> List[AbstractDimOrVariadicDim]:
    # Copied from jaxtyping/_array_types.py and modified to mypy languages.
    dims: List[AbstractDimOrVariadicDim] = []
    index_variadic = None
    for index, elem in enumerate(dim_str.split()):
        if "," in elem and "(" not in elem:
            # Common mistake.
            # Disable in the case that there's brackets to allow for function calls,
            # e.g. `min(foo,bar)`, in symbolic axes.
            raise ValueError("Axes should be separated with spaces, not commas")
        if elem.endswith("#"):
            raise ValueError(
                "As of jaxtyping v0.1.0, broadcastable axes are now denoted "
                "with a # at the start, rather than at the end"
            )

        if "..." in elem:
            if elem != "...":
                raise ValueError(
                    "Anonymous multiple axes '...' must be used on its own; "
                    f"got {elem}"
                )
            broadcastable = False
            variadic = True
            anonymous = True
            treepath = False
            dim_type = _DimType.named
        else:
            broadcastable = False
            variadic = False
            anonymous = False
            treepath = False
            while True:
                if len(elem) == 0:
                    # This branch needed as just `_` is valid
                    break
                first_char = elem[0]
                if first_char == "#":
                    if broadcastable:
                        raise ValueError(
                            "Do not use # twice to denote broadcastability, e.g. "
                            "`##foo` is not allowed"
                        )
                    broadcastable = True
                    elem = elem[1:]
                elif first_char == "*":
                    if variadic:
                        raise ValueError(
                            "Do not use * twice to denote accepting multiple "
                            "axes, e.g. `**foo` is not allowed"
                        )
                    variadic = True
                    elem = elem[1:]
                elif first_char == "_":
                    if anonymous:
                        raise ValueError(
                            "Do not use _ twice to denote anonymity, e.g. `__foo` "
                            "is not allowed"
                        )
                    anonymous = True
                    elem = elem[1:]
                elif first_char == "?":
                    raise NotImplementedError("PyTree is not supported")
                    # if treepath:
                    #     raise ValueError(
                    #         "Do not use ? twice to denote dependence on location "
                    #         "within a PyTree, e.g. `??foo` is not allowed"
                    #     )
                    # treepath = True
                    # elem = elem[1:]
                # Allow e.g. `foo=4` as an alternate syntax for just `4`, so that one
                # can write e.g. `Float[Array, "rows=3 cols=4"]`
                elif elem.count("=") == 1:
                    _, elem = elem.split("=")
                else:
                    break
            if len(elem) == 0 or elem.isidentifier():
                dim_type = _DimType.named
            else:
                try:
                    elem = int(elem)
                except ValueError:
                    dim_type = _DimType.symbolic
                else:
                    dim_type = _DimType.fixed

        if variadic:
            if index_variadic is not None:
                raise ValueError(
                    "Cannot use variadic specifiers (`*name` or `...`) "
                    "more than once."
                )
            index_variadic = index

        parsed: AbstractDimOrVariadicDim
        if dim_type is _DimType.fixed:
            if variadic:
                raise ValueError(
                    "Cannot have a fixed axis bind to multiple axes, e.g. "
                    "`*4` is not allowed."
                )
            if anonymous:
                raise ValueError(
                    "Cannot have a fixed axis be anonymous, e.g. `_4` is not allowed."
                )
            if treepath:
                raise ValueError(
                    "Cannot have a fixed axis have tree-path dependence, e.g. `?4` is "
                    "not allowed."
                )
            parsed = FixedDim(elem)
        elif dim_type is _DimType.named:
            if anonymous:
                if broadcastable:
                    raise ValueError(
                        "Cannot have an axis be both anonymous and "
                        "broadcastable, e.g. `#_` is not allowed."
                    )
                if variadic:
                    parsed = AnonymousVariadicDim()
                else:
                    parsed = AnonymousDim()
            else:
                if variadic:
                    parsed = NamedVariadicDim(elem, broadcastable=broadcastable)
                else:
                    parsed = NamedDim(elem, broadcastable=broadcastable)
        else:
            assert dim_type is _DimType.symbolic
            if anonymous:
                raise ValueError(
                    "Cannot have a symbolic axis be anonymous, e.g. "
                    "`_foo+bar` is not allowed"
                )
            if variadic:
                raise ValueError(
                    "Cannot have symbolic multiple-axes, e.g. "
                    "`*foo+bar` is not allowed"
                )
            if treepath:
                raise ValueError(
                    "Cannot have a symbolic axis with tree-path dependence, e.g. "
                    "`?foo+bar` is not allowed"
                )
            raise NotImplementedError("Symbolic axes are not supported yet")
            # elem = SymbolicDim(elem, broadcastable)
        dims.append(parsed)
    return dims

def dump_dims(dims: List[AbstractDimOrVariadicDim]) -> str:
    result = [str(d) for d in dims]
    return " ".join(result)

def repr_instance(typ: Instance, options):
    assert typ.type.fullname.startswith("jaxtyping._array_types")
    visitor = TypeStrVisitor(options=options)
    dtype = typ.type.fullname.split(".")[-1]
    backend: Instance = typ.args[0]
    shape: LiteralType = typ.args[1]
    result = f"{dtype}[{backend}, '{shape.value}']"
    return result


def check_shape_compatibility(
    xs: List[AbstractDimOrVariadicDim],
    ys: List[AbstractDimOrVariadicDim],
    allow_broadcast: bool,
) -> Optional[List[AbstractDimOrVariadicDim]]:
    """Check compatibility between xs and ys.
    If not compatible, return None.
    """
    if len(ys) > len(xs):
        return check_shape_compatibility(ys, xs, allow_broadcast=allow_broadcast)
    # Require: len(xs) >= len(ys)
    if len(xs) != len(ys):
        if not allow_broadcast:
            return None
        # dim expansion
        assert len(xs) > len(ys)
        ys = [FixedDim(1)] * (max(len(xs), len(ys)) - len(ys)) + ys
    
    assert len(xs) == len(ys)
    zs: List[AbstractDimOrVariadicDim] = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        match x:
            case AnonymousDim():
                if isinstance(y, AbstractDim):
                    zs.append(AnonymousDim())  # propagate unknown
                else:
                    return None
            case AnonymousVariadicDim():
                if isinstance(y, VariadicDim):
                    zs.append(AnonymousVariadicDim())
                else:
                    return None
            case NamedDim(name=name):
                if isinstance(y, AnonymousDim):
                    zs.append(AnonymousDim())
                elif allow_broadcast and isinstance(y, FixedDim) and y.size == 1:
                    zs.append(x)
                elif isinstance(y, NamedDim) and y.name == name:
                    zs.append(x)
                else:
                    return None
            case NamedVariadicDim(name=name):
                if isinstance(y, AnonymousVariadicDim):
                    zs.append(AnonymousVariadicDim())
                elif allow_broadcast and i == 0 and isinstance(y, FixedDim) and y.size == 1:
                    zs.append(x)
                elif isinstance(y, NamedVariadicDim) and y.name == name:
                    zs.append(x)
                else:
                    return None
            case FixedDim(size=size):
                if isinstance(y, AnonymousDim):
                    zs.append(AnonymousDim())
                elif allow_broadcast and isinstance(y, FixedDim) and y.size == 1:
                    zs.append(x)
                elif allow_broadcast and size == 1 and isinstance(y, AbstractDim):
                    zs.append(y)
                elif allow_broadcast and size == 1 and i == 0 and isinstance(y, VariadicDim):
                    zs.append(y)
                elif isinstance(y, FixedDim) and y.size == size:
                    zs.append(x)
                else:
                    return None
            case SymbolicDim(elem=elem):
                if isinstance(y, AnonymousDim):
                    zs.append(AnonymousDim())
                elif allow_broadcast and isinstance(y, FixedDim) and y.size == 1:
                    zs.append(x)
                elif isinstance(y, SymbolicDim) and y.elem == elem:
                    zs.append(y)
                else:
                    return None
    return zs

def compare_dtype(x: str, y: str) -> Optional[int]:
    """Compare dtypes.
    if x < y (x will be promoted to y), return 1
    if x > y, return -1
    if x == y, return 0
    if x and y is incompatible, return None
    """
    if x in dtype_orders and y in dtype_orders:
        # promotable
        x_rank = dtype_orders.index(x)
        y_rank = dtype_orders.index(y)
        if x_rank < y_rank:
            return 1
        elif x_rank > y_rank:
            return -1
        else:
            return 0
    else:
        if x == y:
            return 0
        else:
            return None
