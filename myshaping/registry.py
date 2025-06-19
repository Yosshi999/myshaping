from typing import List, Callable, Optional
FUNCTION_HOOKS = {}
TYPE_ANALYZE_HOOKS = {}
METHOD_HOOKS = {}

def construct_registry(hooks: dict):
    def register(*names: str):
        def decorator(func: Callable):
            for name in names:
                hooks[name] = func
            return func
        return decorator
    def get(name: str) -> Optional[Callable]:
        return hooks.get(name, None)
    return register, get

register_function_hook, get_function_hook = construct_registry(FUNCTION_HOOKS)
register_type_analyze_hook, get_type_analyze_hook = construct_registry(TYPE_ANALYZE_HOOKS)
register_method_hook, get_method_hook = construct_registry(METHOD_HOOKS)