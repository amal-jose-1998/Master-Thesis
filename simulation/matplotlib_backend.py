
def is_non_interactive_backend(backend_name):
    backend_key = str(backend_name).lower().strip()
    return backend_key in {
        "agg",
        "module://matplotlib.backends.backend_agg",
        "module://matplotlib_inline.backend_inline",
    }


def ensure_interactive_backend(get_backend, set_backend, *, log_prefix, candidates=("TkAgg", "QtAgg")):
    backend_name = str(get_backend()).lower().strip()
    if not is_non_interactive_backend(backend_name):
        return backend_name

    for candidate in candidates:
        try:
            set_backend(candidate)
            backend_name = str(get_backend()).lower().strip()
            print(f"{log_prefix} Matplotlib backend switched to {backend_name}")
            break
        except Exception:
            pass

    return backend_name
