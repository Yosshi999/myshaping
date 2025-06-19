from mypy.plugin import Plugin, FunctionContext, MethodContext
from collections import namedtuple

Argument = namedtuple('Argument', ['arg_type', 'arg_kind', 'arg_name', 'arg'])

def transpose_funcargs(ctx: FunctionContext | MethodContext) -> dict[str, Argument]:
    ctxdict = {}
    for i, name in enumerate(ctx.callee_arg_names):
        if len(ctx.arg_kinds[i]) == 0:
            continue
        ctxdict[name] = Argument(
            arg_type=ctx.arg_types[i],
            arg_kind=ctx.arg_kinds[i],
            arg_name=ctx.arg_names[i],
            arg=ctx.args[i]
        )
    return ctxdict
