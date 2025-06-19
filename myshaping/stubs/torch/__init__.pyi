from torch._tensor import Tensor
from typing import TypeAlias, Any

class dtype: ...
_Float32: TypeAlias = dtype
_Float64: TypeAlias = dtype
_Complex64: TypeAlias = dtype
_Complex128: TypeAlias = dtype
_Float16: TypeAlias = dtype
_BFloat16: TypeAlias = dtype
_UInt8: TypeAlias = dtype
_Int8: TypeAlias = dtype
_UInt16: TypeAlias = dtype
_Int16: TypeAlias = dtype
_Int32: TypeAlias = dtype
_Int64: TypeAlias = dtype
_Bool: TypeAlias = dtype

float32: _Float32
float: _Float32
float64: _Float64
double: _Float64
complex64: _Complex64
cfloat: _Complex64
complex128: _Complex128
cdouble: _Complex128
float16: _Float16
half: _Float16
bfloat16: _BFloat16
uint8: _UInt8
int8: _Int8
int16: _Int16
short: _Int16
int32: _Int32
int: _Int32
int64: _Int64
long: _Int64
bool: _Bool

def randn(*size: int, out=None, dtype=None, **kwargs) -> Tensor: ...
def rand(*size: int, out=None, dtype=None, **kwargs) -> Tensor: ...
def zeros(*size: int, out=None, dtype=None, **kwargs) -> Tensor: ...
def ones(*size: int, out=None, dtype=None, **kwargs) -> Tensor: ...
def empty(*size: int, out=None, dtype=None, **kwargs) -> Tensor: ...
def full(*size: int, fill_value: Any, out=None, dtype=None, **kwargs) -> Tensor: ...
def randint(low: int, high: int, *size: int, out=None, dtype=None, **kwargs) -> Tensor: ...