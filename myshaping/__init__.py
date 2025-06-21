def reveal_jaxtype(x):
    from wadler_lindig import pformat
    print("Runtime type is", pformat(x))
    return x