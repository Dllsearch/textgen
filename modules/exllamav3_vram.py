"""
Estimates the VRAM an ExLlamaV3 launch will need, from the plan built by
modules/exllamav3_params.py.

Weights and cache are measured, not fitted: tensor sizes come from the
safetensors headers and cache sizes from the cache layers themselves, both
of which ExLlamaV3 can report without allocating anything on the GPU. Only
the runtime overhead is an approximation.

exllamav3 is imported lazily so the UI can import this module at startup.
"""

from pathlib import Path

from modules.logging_colors import logger

MIB = 1024 ** 2
GIB = 1024 ** 3

# Rough runtime overhead that is not weights or cache: CUDA context, cuBLAS
# workspaces and the per-forward activation buffers.
BASE_OVERHEAD = 512 * MIB
ACTIVATION_BUFFERS = 6
MAX_OUTPUT_ROWS = 32

_config_cache = {}


def _get_config(model_dir):
    """
    Config.from_directory() reads the safetensors headers, which takes about a
    second, so keep it around per directory.
    """
    from exllamav3 import Config

    path = Path(model_dir)
    key = str(path.resolve())
    mtime = path.stat().st_mtime

    cached = _config_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    config = Config.from_directory(str(path))
    _config_cache[key] = (mtime, config)
    return config


def _component_keys(config, component):
    """
    Tensor keys belonging to one component of a model, as a set so components
    that share tensors (vision towers reusing the token embeddings, for
    instance) are not counted twice.
    """
    from exllamav3 import Model

    model = Model.from_config(config, component=component)
    stc = config.stc
    keys = set()

    try:
        trie = stc.get_tensor_file_map_trie()
    except Exception:
        return model, None

    for module in model.modules:
        prefix = getattr(module, 'key', None)
        if not prefix:
            continue

        if prefix in stc.tensor_file_map:
            keys.add(prefix)

        keys.update(trie.keys(prefix + "."))

    return model, keys


def _keys_size(config, keys):
    return sum(config.stc.get_tensor_size(k, optional=True) for k in keys)


def _files_size(model_dir):
    return sum(f.stat().st_size for f in Path(model_dir).glob('*.safetensors'))


def _cache_size(model, plan, max_history=None):
    """
    Returns (kv_bytes, recurrent_state_bytes) for a model, without allocating.
    """
    from exllamav3 import Cache
    from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant

    spec = plan['cache']
    kwargs = {k: v for k, v in spec.items() if k not in ('layer_type', 'max_num_tokens', 'max_history')}
    layer_type = CacheLayer_fp16 if spec['layer_type'] == 'fp16' else CacheLayer_quant

    if max_history is None:
        max_history = spec.get('max_history', 0)

    cache = Cache(
        model,
        max_num_tokens=spec['max_num_tokens'],
        layer_type=layer_type,
        max_history=max_history,
        **kwargs
    )

    kv = sum(layer.storage_size() + layer.overhead_size() for layer in cache.layers.values())

    # Recurrent/linear-attention layers keep their state on the GPU, sized by the
    # number of slots and by max_history, which is what speculative decoding grows.
    recurrent = 0
    for layer in cache.recurrent_layers.values():
        try:
            recurrent += layer.storage_size()
        except Exception:
            pass

    return kv, recurrent


def _overhead(config, plan):
    hidden = getattr(config, 'hidden_size', 0) or 0
    vocab = getattr(config, 'vocab_size', 0) or 0
    chunk = plan['generator'].get('max_chunk_size', 2048) if plan['generator'] else 2048

    activations = chunk * hidden * 2 * ACTIVATION_BUFFERS
    logits = MAX_OUTPUT_ROWS * vocab * 4

    return BASE_OVERHEAD + activations + logits


def estimate(model_dir, plan, model_dir_draft=None):
    """
    Returns a dict of byte counts per part plus a 'total', or a dict with an
    'error' key if the model cannot be inspected.
    """
    try:
        config = _get_config(model_dir)
    except Exception as e:
        return {'error': str(e)}

    parts = {}
    notes = []

    text_model, text_keys = _component_keys(config, 'text')
    counted = set(text_keys) if text_keys else set()

    if text_keys:
        parts['Weights'] = _keys_size(config, text_keys)
    else:
        parts['Weights'] = _files_size(model_dir)
        notes.append("Could not break the weights down per component; using the total file size.")

    if plan['load_vision'] and 'vision' in config.model_classes and text_keys:
        _, vision_keys = _component_keys(config, 'vision')
        if vision_keys:
            extra = vision_keys - counted
            parts['Vision'] = _keys_size(config, extra)
            counted |= extra

    kv, recurrent = _cache_size(text_model, plan)
    parts['KV cache'] = kv
    if recurrent:
        parts['Recurrent states'] = recurrent

    draft = plan['draft']
    if draft['mtp'] and 'mtp' in config.model_classes:
        mtp_model, mtp_keys = _component_keys(config, 'mtp')
        if mtp_keys and text_keys:
            parts['MTP head'] = _keys_size(config, mtp_keys - counted)

        mtp_kv, mtp_recurrent = _cache_size(mtp_model, plan, max_history=0)
        parts['MTP cache'] = mtp_kv + mtp_recurrent
    elif draft['model_draft'] and model_dir_draft:
        try:
            draft_config = _get_config(model_dir_draft)
            draft_model, draft_keys = _component_keys(draft_config, 'text')
            parts['Draft weights'] = _keys_size(draft_config, draft_keys) if draft_keys else _files_size(model_dir_draft)
            draft_kv, draft_recurrent = _cache_size(draft_model, plan, max_history=0)
            parts['Draft cache'] = draft_kv + draft_recurrent
        except Exception as e:
            notes.append(f"Could not inspect the draft model: {e}")

    parts['Overhead'] = _overhead(config, plan)

    return {'parts': parts, 'total': sum(parts.values()), 'notes': notes}


def _fmt(value):
    return f"{value / GIB:.2f} GiB" if value >= GIB else f"{value / MIB:.0f} MiB"


def format_html(estimate_result, plan=None):
    """
    Renders the estimate as the HTML shown in the Model tab.
    """
    if not estimate_result:
        return "<div id=\"vram-info\">Estimated VRAM: <span class=\"value\">select a model</span></div>"

    if 'error' in estimate_result:
        return f"<div id=\"vram-info\">Estimated VRAM: <span class=\"value\">unavailable</span><br><small>{estimate_result['error']}</small></div>"

    parts = estimate_result['parts']
    breakdown = ' + '.join(f"{name} {_fmt(value)}" for name, value in parts.items() if value)

    notes = list(estimate_result['notes'])
    if plan:
        if plan['model_load'].get('tensor_p') or plan['model_load'].get('use_per_device'):
            notes.append("This is the total across all GPUs; it is split between them at load time.")
        if plan['config'].get('moe_cpu_offload') or plan['config'].get('moe_cpu_split'):
            notes.append("Experts offloaded to the CPU are still counted here.")

    html = (
        "<div id=\"vram-info\">Estimated VRAM: "
        f"<span class=\"value\">{_fmt(estimate_result['total'])}</span>"
        f"<br><small>{breakdown}</small>"
    )

    for note in notes:
        html += f"<br><small>{note}</small>"

    return html + "</div>"


def estimate_html(model_name, plan, model_dir=None, model_dir_draft=None):
    """
    Convenience wrapper used by the UI.
    """
    if not model_name or model_name in ('None', ''):
        return format_html(None)

    from modules import shared

    path = Path(model_dir) if model_dir else Path(shared.args.model_dir) / model_name
    if not path.is_dir():
        return format_html(None)

    try:
        result = estimate(path, plan, model_dir_draft=model_dir_draft)
    except Exception as e:
        logger.warning(f"Failed to estimate ExLlamaV3 VRAM usage: {e}")
        result = {'error': str(e)}

    return format_html(result, plan)
