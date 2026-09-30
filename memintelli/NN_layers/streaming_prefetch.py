def configure_from_execution_trace(model, linear_type, trace):
    empty = {
        "enabled": False,
        "trace_length": len(trace or []),
        "streaming_layers": 0,
        "links": 0,
        "max_prefetch_window": 0,
    }
    if linear_type is None or not trace:
        return empty

    mem_layers = {
        id(module): module
        for module in model.modules()
        if isinstance(module, linear_type)
    }
    streaming_layers = []
    seen = set()
    for module in trace:
        module_id = id(module)
        if module_id not in mem_layers or module_id in seen:
            continue
        seen.add(module_id)
        if bool(getattr(module, "_streaming", False)):
            streaming_layers.append(module)

    for module in mem_layers.values():
        object.__setattr__(module, "_next_streaming_layer", None)
    for current, target in zip(streaming_layers, streaming_layers[1:]):
        object.__setattr__(current, "_next_streaming_layer", target)

    links = max(0, len(streaming_layers) - 1)
    return {
        "enabled": True,
        "trace_length": len(trace),
        "streaming_layers": len(streaming_layers),
        "links": links,
        "max_prefetch_window": 1 if links else 0,
    }
