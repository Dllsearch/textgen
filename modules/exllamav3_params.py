"""
Resolves the ExLlamaV3 launch parameters in one place.

Both the loaders (modules/exllamav3.py, modules/exllamav3_hf.py) and the
"Launch parameters" preview in the Model tab build their view of the launch
from build_plan(), so what the preview shows is what actually gets passed to
ExLlamaV3.

This module must not import exllamav3, since the UI imports it at startup.
"""

from pathlib import Path

PAGE_SIZE = 256

TP_PARALLELISM_KEYS = ('attn', 'mlp', 'moe', 'linear', 'linear_attn')


def _get(source, key, default=None):
    if isinstance(source, dict):
        value = source.get(key, default)
    else:
        value = getattr(source, key, default)

    return default if value is None else value


def parse_cache_type(cache_type):
    """
    Returns (layer_type, cache_kwargs, notes), where layer_type is 'fp16' or 'quant'.
    """
    notes = []
    cache_type = (cache_type or 'fp16').lower()

    if cache_type == 'fp16':
        return 'fp16', {}, notes

    if not cache_type.startswith('q'):
        notes.append(f"Unrecognized cache type: {cache_type}. Falling back to fp16.")
        return 'fp16', {}, notes

    try:
        if '_' in cache_type:
            # Different bits for k and v (e.g., q4_q8)
            k_part, v_part = cache_type.split('_')
            k_bits = int(k_part[1:])
            v_bits = int(v_part[1:])
        else:
            # Same bits for k and v (e.g., q4)
            k_bits = v_bits = int(cache_type[1:])
    except ValueError:
        notes.append(f"Unrecognized cache type: {cache_type}. Falling back to fp16.")
        return 'fp16', {}, notes

    if not (2 <= k_bits <= 8 and 2 <= v_bits <= 8):
        notes.append(f"Invalid quantization bits: k_bits={k_bits}, v_bits={v_bits}. Must be between 2 and 8. Falling back to fp16.")
        return 'fp16', {}, notes

    return 'quant', {'k_bits': k_bits, 'v_bits': v_bits}, notes


def parse_tp_parallelism(value):
    """
    Parses "attn=4, mlp=2" into {'attn': 4, 'mlp': 2}. Returns (limits, notes).
    """
    limits = {}
    notes = []
    if not value:
        return limits, notes

    for part in str(value).replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue

        if '=' not in part and ':' not in part:
            notes.append(f"Ignoring tp-parallelism entry without a value: {part}")
            continue

        key, _, num = part.replace(':', '=').partition('=')
        key = key.strip().lower()
        if key not in TP_PARALLELISM_KEYS:
            notes.append(f"Ignoring unknown tp-parallelism key: {key} (valid: {', '.join(TP_PARALLELISM_KEYS)})")
            continue

        try:
            limits[key] = int(num.strip())
        except ValueError:
            notes.append(f"Ignoring non-numeric tp-parallelism value: {part}")

    return limits, notes


def parse_float_list(value):
    """
    Parses "20, 7, 7" into [20.0, 7.0, 7.0]. Returns (values, ok).
    """
    value = (value or '').strip()
    if not value:
        return None, True

    try:
        return [float(v) for v in value.split(',')], True
    except ValueError:
        return None, False


def normalize_device(value):
    """
    Accepts "1", "cuda:1" or "cuda 1" and returns "cuda:1". None if empty.
    """
    value = str(value or '').strip().lower()
    if not value:
        return None

    if value.isdigit():
        return f"cuda:{value}"

    return value.replace(' ', ':')


def detect_components(model_dir):
    """
    Best-effort component detection from a model's config.json, for the UI
    preview. The loaders use config.model_classes instead, which is what
    ExLlamaV3 actually goes by.
    """
    import json

    path = Path(model_dir) / 'config.json'
    if not path.is_file():
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    except Exception:
        return None

    components = ['text']
    for key, name in (('vision_config', 'vision'), ('audio_config', 'audio')):
        if key in config_dict or key in config_dict.get('text_config', {}):
            components.append(name)

    return components


def round_cache_tokens(ctx_size):
    """
    ExLlamaV3 allocates the cache in pages of 256 tokens.
    """
    ctx_size = int(ctx_size or 0) or 8192
    if ctx_size % PAGE_SIZE != 0:
        ctx_size = ((ctx_size // PAGE_SIZE) + 1) * PAGE_SIZE

    return ctx_size


def build_plan(source, hf=False):
    """
    Resolves every ExLlamaV3 launch parameter from an args namespace or from a
    UI state dict. Returns a dict of the kwargs each ExLlamaV3 object receives,
    plus a list of notes about values that were adjusted or ignored.

    :param source:
        shared.args, or an interface state dict with the same keys

    :param hf:
        Build the plan for the ExLlamav3_HF loader, which has no generator
    """
    notes = []

    requested_ctx = int(_get(source, 'ctx_size', 0) or 0)
    cache_tokens = round_cache_tokens(requested_ctx)
    if requested_ctx and cache_tokens != requested_ctx:
        notes.append(f"max_num_tokens must be a multiple of {PAGE_SIZE}. Adjusting from {requested_ctx} to {cache_tokens}.")
    elif not requested_ctx:
        notes.append(f"ctx-size is 0, defaulting to {cache_tokens} tokens.")

    layer_type, cache_kwargs, cache_notes = parse_cache_type(_get(source, 'cache_type', 'fp16'))
    notes += cache_notes

    # Config-level options
    config = {}
    layer_map = (_get(source, 'exl3_layer_map', '') or '').strip()
    if layer_map:
        config['layer_map'] = layer_map

    override = (_get(source, 'exl3_override', '') or '').strip()
    if override:
        if not Path(override).is_file():
            notes.append(f"Tensor override file not found, ignoring: {override}")
        else:
            config['override'] = override

    enable_tp = bool(_get(source, 'enable_tp', False))

    moe_cpu_layers = int(_get(source, 'exl3_moe_cpu_layers', 0) or 0)
    moe_cpu_split = int(_get(source, 'exl3_moe_cpu_split', 0) or 0)
    if moe_cpu_layers and moe_cpu_split:
        notes.append("moe-cpu-layers and moe-cpu-split are mutually exclusive. Ignoring moe-cpu-split.")
        moe_cpu_split = 0

    if (moe_cpu_layers or moe_cpu_split) and enable_tp:
        notes.append("MoE CPU offloading requires layer-split mode. Ignoring it because enable_tp is set.")
        moe_cpu_layers = moe_cpu_split = 0

    if moe_cpu_layers:
        config['moe_cpu_offload'] = moe_cpu_layers
    if moe_cpu_split:
        config['moe_cpu_split'] = moe_cpu_split

    moe_cpu_threads = int(_get(source, 'exl3_moe_cpu_threads', 0) or 0)
    if moe_cpu_threads and (moe_cpu_layers or moe_cpu_split):
        config['moe_cpu_threads'] = moe_cpu_threads
    elif moe_cpu_threads:
        notes.append("moe-cpu-threads only applies with moe-cpu-layers or moe-cpu-split. Ignoring it.")

    if _get(source, 'exl3_ngram_ram', False):
        config['ngram_stream_from_disk'] = False

    # Model.load() options
    model_load = {'progressbar': True}

    gpu_split = (_get(source, 'gpu_split', '') or '').strip()
    split, ok = parse_float_list(gpu_split)
    if not ok:
        notes.append(f"Could not parse gpu-split, ignoring it: {gpu_split}")
    elif split:
        model_load['use_per_device'] = split

    reserve, ok = parse_float_list(_get(source, 'exl3_reserve_per_device', ''))
    if not ok:
        notes.append("Could not parse reserve-per-device, ignoring it.")
        reserve = None
    elif reserve:
        model_load['reserve_per_device'] = reserve

    if enable_tp:
        model_load['tensor_p'] = True
        model_load['tp_backend'] = _get(source, 'tp_backend', 'native')

        tp_limits, tp_notes = parse_tp_parallelism(_get(source, 'exl3_tp_parallelism', ''))
        notes += tp_notes
        if tp_limits:
            model_load['tp_dev_limits'] = tp_limits

        if _get(source, 'exl3_moe_tensor_split', False):
            model_load['tp_options'] = {'moe_tensor_split': True}
    elif (_get(source, 'exl3_tp_parallelism', '') or _get(source, 'exl3_moe_tensor_split', False)):
        notes.append("tp-parallelism and moe-tensor-split only apply in tensor-parallel mode. Ignoring them.")

    autosplit_batch_size = int(_get(source, 'exl3_autosplit_batch_size', 1) or 1)
    model_load['max_batch_size'] = autosplit_batch_size

    if _get(source, 'exl3_load_verbose', False):
        model_load['verbose'] = True

    swa_full = bool(_get(source, 'exl3_swa_full', False))

    # Cache
    cache = {'max_num_tokens': cache_tokens, 'layer_type': layer_type}
    cache.update(cache_kwargs)

    # Recurrent models allocate one set of states per slot, so this is a real
    # VRAM knob on those and a no-op on everything else.
    cache['max_batch_size'] = max(1, int(_get(source, 'exl3_cache_slots', 16) or 16))

    # Component sub-models (vision, MTP). ExLlamaV3 has no audio component.
    load_vision = not bool(_get(source, 'exl3_no_vision', False))
    vision_load = {'progressbar': True}
    vision_device = normalize_device(_get(source, 'exl3_vision_device', ''))
    if vision_device:
        vision_load['device'] = vision_device
        if enable_tp:
            notes.append("vision-device pins the vision component to one device; the text model still loads in TP mode.")
    else:
        if split:
            vision_load['use_per_device'] = split
        if reserve:
            vision_load['reserve_per_device'] = reserve

    if _get(source, 'exl3_load_verbose', False):
        vision_load['verbose'] = True

    # Draft model / speculative decoding
    draft = {
        'model_draft': (_get(source, 'model_draft', '') or '').strip(),
        'mtp': bool(_get(source, 'exl3_mtp', False)),
        'draft_max': int(_get(source, 'draft_max', 0) or 0),
    }

    if draft['mtp'] and draft['model_draft'] and draft['model_draft'].lower() != 'none':
        notes.append("exl3-mtp uses the main model as its own draft. Ignoring model-draft.")
        draft['model_draft'] = ''

    if draft['model_draft'].lower() == 'none':
        draft['model_draft'] = ''

    draft_load = {'progressbar': True}
    if split:
        draft_load['use_per_device'] = split
    if reserve:
        draft_load['reserve_per_device'] = reserve
    draft_load['max_batch_size'] = autosplit_batch_size
    if _get(source, 'exl3_load_verbose', False):
        draft_load['verbose'] = True

    draft['load'] = draft_load

    # Generator
    generator = {}
    if not hf:
        generator['max_batch_size'] = int(_get(source, 'exl3_max_batch_size', 256) or 256)
        generator['max_chunk_size'] = int(_get(source, 'exl3_max_chunk_size', 2048) or 2048)

        if draft['model_draft'] or draft['mtp']:
            generator['num_draft_tokens'] = draft['draft_max']

        ngram_match_min = int(_get(source, 'exl3_ngram_match_min', 0) or 0)
        if ngram_match_min:
            if draft['model_draft'] or draft['mtp']:
                notes.append("ngram-match-min is ignored while a draft model or MTP head is used.")
            else:
                generator['ngram_match_min'] = ngram_match_min
                if draft['draft_max']:
                    generator['num_draft_tokens'] = draft['draft_max']

        if _get(source, 'exl3_dynamic_draft', False):
            generator['dynamic_draft_tokens'] = True
            generator['draft_confidence'] = float(_get(source, 'exl3_draft_confidence', 0.4) or 0.4)

        cpu_cache_gb = float(_get(source, 'exl3_cpu_cache', 0) or 0)
        if cpu_cache_gb:
            if enable_tp:
                notes.append("cpu-cache is not supported in tensor-parallel mode. Ignoring it.")
            else:
                generator['cpu_cache_size'] = int(cpu_cache_gb * 1024 ** 3)

        recurrent_cache_gb = float(_get(source, 'exl3_recurrent_cache', 4.0) or 4.0)
        generator['recurrent_cache_size'] = int(recurrent_cache_gb * 1024 ** 3)

        if _get(source, 'exl3_no_defrag', False):
            generator['enable_defrag'] = False

    # Recurrent/SWA models reserve one past state per draft token. Without this,
    # drafting fails with "recurrent_state must be [num_slots, max_history + 1, ...]".
    # The loader raises it further if the draft model asks for a longer default.
    if not hf:
        if draft['mtp'] or draft['model_draft']:
            cache['max_history'] = max(draft['draft_max'], 4)
        elif generator.get('ngram_match_min'):
            cache['max_history'] = max(draft['draft_max'], 4)

    return {
        'hf': hf,
        'ctx_size': requested_ctx,
        'cache_tokens': cache_tokens,
        'swa_full': swa_full,
        'load_metrics': bool(_get(source, 'exl3_load_metrics', False)),
        'config': config,
        'cache': cache,
        'model_load': model_load,
        'load_vision': load_vision,
        'vision_load': vision_load,
        'generator': generator,
        'draft': draft,
        'notes': notes,
    }


def apply_config_options(config, plan):
    """
    Applies the config-level options of a plan to a loaded Config object.
    """
    options = plan['config']

    for key in ('moe_cpu_offload', 'moe_cpu_split', 'moe_cpu_threads', 'ngram_stream_from_disk'):
        if key in options:
            setattr(config.infer_params, key, options[key])

    override = options.get('override')
    if override:
        import yaml
        from exllamav3.loader import (
            SafetensorsCollection,
            VariantSafetensorsCollection
        )

        with open(override, 'r') as f:
            comp = yaml.safe_load(f)

        sources = {s['id']: s['model_dir'] for s in comp['sources']}
        overrides = {o['key']: sources[o['source']] for o in comp['overrides']}

        collections = {}
        for o_key, o_dir in overrides.items():
            collections.setdefault(o_dir, []).append(o_key)

        if collections:
            vstc = VariantSafetensorsCollection(config.stc)
            for o_dir, o_keys in collections.items():
                vstc.add_stc(o_keys, SafetensorsCollection(o_dir))

            config.stc = vstc

    return config


def _format_kwargs(kwargs):
    parts = []
    for key, value in kwargs.items():
        parts.append(f"{key}={value!r}")

    return ', '.join(parts)


def format_plan(plan, model_name=None, components=None):
    """
    Renders the plan as pseudo-code mirroring the actual ExLlamaV3 calls.

    :param components:
        (optional) Components detected in the model, from detect_components()
    """
    lines = []

    if model_name and model_name not in ('None', None):
        lines.append(f"# {model_name}")

    config = dict(plan['config'])
    override = config.pop('override', None)
    layer_map = config.pop('layer_map', None)

    config_args = []
    if layer_map:
        config_args.append(f"layer_map={layer_map!r}")

    lines.append(f"Config.from_directory(model_dir{', ' + ', '.join(config_args) if config_args else ''})")
    for key, value in config.items():
        lines.append(f"    config.infer_params.{key} = {value!r}")
    if override:
        lines.append(f"    config.stc = VariantSafetensorsCollection(...)  # {override}")

    lines.append(f"Model.from_config(config, swa_full={plan['swa_full']!r})")

    cache = dict(plan['cache'])
    layer_type = cache.pop('layer_type')
    layer_type = 'CacheLayer_fp16' if layer_type == 'fp16' else 'CacheLayer_quant'
    cache_str = _format_kwargs(cache)
    lines.append(f"Cache(model, layer_type={layer_type}{', ' + cache_str if cache_str else ''})")

    lines.append(f"model.load({_format_kwargs(plan['model_load'])})")

    if plan['hf']:
        # The HF wrapper loads the text component only
        if components and 'vision' in components:
            lines.append("# Vision component: not supported by ExLlamav3_HF, use the ExLlamav3 loader")

        if plan['load_metrics']:
            lines.append("config.stc.metrics.print()")

        text = '\n'.join(lines)
        if plan['notes']:
            text += '\n\n# Notes:\n' + '\n'.join(f"#  - {note}" for note in plan['notes'])

        return text

    has_vision = components is None or 'vision' in components
    if not plan['load_vision']:
        lines.append("# Vision component: disabled by no-vision")
    elif has_vision:
        suffix = '' if components else '  # only if the model has one'
        lines.append(f"vision_model = Model.from_config(config, component='vision'){suffix}")
        lines.append(f"vision_model.load({_format_kwargs(plan['vision_load'])})")

    if components and 'audio' in components:
        lines.append("# Audio component: present in the model config, but ExLlamaV3 does not implement one")

    draft = plan['draft']
    if draft['mtp']:
        lines.append("draft_model = Model.from_config(config, component='mtp')")
        lines.append(f"draft_model.load({_format_kwargs(draft['load'])})")
        lines.append(f"# num_draft_tokens={draft['draft_max']}")
    elif draft['model_draft']:
        lines.append(f"draft_model = Model.from_config(Config.from_directory({draft['model_draft']!r}))")
        lines.append(f"draft_model.load({_format_kwargs(draft['load'])})")
        lines.append(f"# num_draft_tokens={draft['draft_max']}")

    if not plan['hf']:
        lines.append(f"Generator({_format_kwargs(plan['generator'])})")

    if plan['load_metrics']:
        lines.append("config.stc.metrics.print()")

    text = '\n'.join(lines)
    if plan['notes']:
        text += '\n\n# Notes:\n' + '\n'.join(f"#  - {note}" for note in plan['notes'])

    return text
