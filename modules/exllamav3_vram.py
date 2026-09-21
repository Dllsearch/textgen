"""
Estimates where an ExLlamaV3 launch puts its memory: VRAM, system RAM and
disk, from the plan built by modules/exllamav3_params.py.

Weights and cache are measured, not fitted: tensor sizes come from the
safetensors headers and cache sizes from the cache layers themselves, both
of which ExLlamaV3 can report without allocating anything on the GPU. The
placement follows the loader's rules: modules marked prefer_cpu (token
embeddings) load to system RAM, n-gram embedding tables (PLE models such as
Qwen3.8-Flash-Next) stream from disk unless ngram-ram is set, and MoE CPU
offloading moves routed experts to RAM, per layer or per expert slice. Only
the runtime overheads are approximations.

exllamav3 is imported lazily so the UI can import this module at startup.
"""

import json
import threading
from pathlib import Path

from modules.logging_colors import logger

MIB = 1024 ** 2
GIB = 1024 ** 3

# Rough runtime overhead that is not weights or cache: CUDA context, cuBLAS
# workspaces and the per-forward activation buffers.
BASE_OVERHEAD = 768 * MIB
ACTIVATION_BUFFERS = 8
MAX_OUTPUT_ROWS = 32

# Host memory of the process itself: Python, torch, the CUDA runtime and the
# web UI. Measured at about 2.5-3 GiB on Windows.
HOST_RUNTIME = 3 * GIB

# ExLlamaV3 keeps this much VRAM free per device when reserve-per-device is empty.
DEFAULT_RESERVE_GB = 0.5

# Headroom to leave under the VRAM budget. The autosplit also needs room for
# the largest transient any layer allocates during its measuring forward, and
# loads that estimated closer than this to the limit were seen to fail.
FIT_MARGIN = 768 * MIB

_config_cache = {}
_layout_cache = {}

# The UI can ask for several estimates at once; building models and touching
# CUDA from parallel threads is neither safe nor useful.
_lock = threading.Lock()


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
    _layout_cache.pop(key, None)
    return config


def _walk(module):
    yield module
    for child in module.modules:
        yield from _walk(child)


def _module_keys(stc, trie, prefix):
    keys = set()
    if not prefix:
        return keys

    if prefix in stc.tensor_file_map:
        keys.add(prefix)

    keys.update(trie.keys(prefix + "."))
    return keys


def _component_layout(config, component):
    """
    Splits the tensors of one model component by where the loader puts them:
    'gpu' (the default), 'cpu' (modules marked prefer_cpu), 'ngram' (n-gram
    embedding tables) and 'moe' (routed experts, one entry per MoE layer in
    load order, as (num_experts, keys, module)).
    """
    from exllamav3 import Model
    from exllamav3.modules import BlockSparseMLP, NGramEmbedding

    model = Model.from_config(config, component=component)
    stc = config.stc

    try:
        trie = stc.get_tensor_file_map_trie()
    except Exception:
        return {'model': model, 'mapped': False}

    # A key belongs to the module with the longest matching prefix: the PLE layer
    # of Qwen3.8-Flash-Next is keyed "layers.1.ple", inside block "layers.1".
    owner = {}
    for top in model.modules:
        prefix = getattr(top, 'key', None)
        for k in _module_keys(stc, trie, prefix):
            if k not in owner or len(prefix) > len(owner[k].key):
                owner[k] = top

    owned = {}
    for k, top in owner.items():
        owned.setdefault(id(top), set()).add(k)

    layout = {'model': model, 'mapped': True, 'gpu': set(), 'cpu': set(), 'ngram': set(), 'moe': []}
    for top in model.modules:
        keys = owned.get(id(top))
        if not keys:
            continue

        if top.caps.get('prefer_cpu'):
            layout['cpu'] |= keys
            continue

        for sub in _walk(top):
            if isinstance(sub, NGramEmbedding):
                ngram_keys = {k for k in keys if k == sub.key or k.startswith(sub.key + '.')}
                layout['ngram'] |= ngram_keys
                keys -= ngram_keys
            elif isinstance(sub, BlockSparseMLP):
                expert_keys = {k for k in keys if k.startswith(sub.key + '.experts.')}
                if expert_keys:
                    layout['moe'].append((sub.num_experts, expert_keys, sub))
                    keys -= expert_keys

        layout['gpu'] |= keys

    return layout


def _get_layout(model_dir, config, component):
    """
    Walking the module tree of a large MoE takes a moment, so cache the
    layouts per directory along with the config they came from.
    """
    key = str(Path(model_dir).resolve())
    per_dir = _layout_cache.setdefault(key, {})
    if component not in per_dir:
        per_dir[component] = _component_layout(config, component)

    return per_dir[component]


class _Counter:
    """
    Sums tensor sizes, counting each tensor once across components (the MTP
    head and the vision tower share the token embeddings and the head with
    the text model).
    """

    def __init__(self, config):
        self.config = config
        self.seen = set()

    def size(self, keys):
        new = set(keys) - self.seen
        self.seen |= new
        return sum(self.config.stc.get_tensor_size(k, optional=True) for k in new)


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

    kv = int(sum(layer.storage_size() + layer.overhead_size() for layer in cache.layers.values()))

    # Recurrent/linear-attention layers keep their state on the GPU, sized by the
    # number of slots and by max_history, which is what speculative decoding grows.
    recurrent = 0
    for layer in cache.recurrent_layers.values():
        try:
            recurrent += int(layer.storage_size())
        except Exception:
            pass

    # The model instance is cached between estimates, so the cache must not
    # stay attached to it.
    cache.detach_from_model()

    return kv, recurrent


def _overhead(config, plan):
    hidden = getattr(config, 'hidden_size', 0) or 0
    vocab = getattr(config, 'vocab_size', 0) or 0
    streams = getattr(config, 'hc_mult', 1) or 1
    chunk = plan['generator'].get('max_chunk_size', 2048) if plan['generator'] else 2048

    # Hyper-connection models (Qwen3.8-Flash-Next) carry several fp32 residual
    # streams through every layer instead of one fp16 hidden state.
    state_bytes = 2 if streams == 1 else 4 * streams
    activations = chunk * hidden * state_bytes * ACTIVATION_BUFFERS
    logits = MAX_OUTPUT_ROWS * vocab * 4

    return BASE_OVERHEAD + activations + logits


def _codebook(model_dir):
    try:
        with open(Path(model_dir) / 'config.json', 'r', encoding='utf-8') as f:
            quant = json.load(f).get('quantization_config', {})
    except Exception:
        return None

    return quant.get('codebook')


def _moe_sizes(counter, moe, chunk):
    """
    Per-layer (num_experts, expert_bytes, buffer_bytes, staging_bytes) of
    routed experts, in load order. Every MoE layer that keeps experts on the
    GPU holds an fp16 intermediate buffer for a full chunk of routed tokens; a
    layer offloaded whole to the CPU instead keeps fp16 input and output
    staging for a chunk, which the CPU worker reads and fills.
    """
    layers = []
    for n, keys, module in moe:
        interm = getattr(module, 'intermediate_size_padded', None) or getattr(module, 'intermediate_size', 0)
        buffer = chunk * (getattr(module, 'num_experts_per_tok', 0) or 0) * interm * 2
        staging = 2 * chunk * (getattr(module, 'hidden_size', 0) or 0) * 2
        layers.append((n, counter.size(keys), buffer, staging))

    return layers


def _place_experts(layers, offload_layers=0, split=0):
    """
    Returns (gpu_bytes, cpu_bytes, gpu_buffer_bytes) for one component's
    expert layers.
    """
    gpu = cpu = buffers = 0
    for i, (n, size, buffer, staging) in enumerate(layers):
        if i < offload_layers:
            cpu += size
            buffers += staging
            continue

        buffers += buffer
        if 0 < split < n:
            on_cpu = size * split // n
            cpu += on_cpu
            gpu += size - on_cpu
        else:
            gpu += size

    return gpu, cpu, buffers


def _ngram_row_bytes(config, keys):
    """
    Bytes per n-gram table row and rows read per token, for the disk note.
    """
    rows = 0
    total = 0
    for k in keys:
        if not (k.endswith('.trellis') or k.endswith('.weight')):
            continue

        try:
            meta = config.stc.get_tensor_meta(k)
            shape = meta[k]['shape'] if k in meta else next(iter(meta.values()))['shape']
        except Exception:
            continue

        rows += shape[0]
        total += config.stc.get_tensor_size(k, optional=True)

    if not rows:
        return None, None

    heads = (getattr(config, 'ngram_size', 0) - 1) * getattr(config, 'heads_per_ngram', 0)
    return total / rows, heads or None


def _gpu_budget(plan):
    """
    VRAM this process could use for a load, the way ExLlamaV3 budgets it:
    what it already holds plus what is free, minus the per-device reserve.
    Returns (budget_bytes, total_bytes) or (None, None) without CUDA.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None

        count = torch.cuda.device_count()
    except Exception:
        return None, None

    split = plan['model_load'].get('use_per_device')
    reserve = plan['model_load'].get('reserve_per_device') or []

    budget = 0
    capacity = 0
    for i in range(count):
        try:
            free, total = torch.cuda.mem_get_info(i)
            held = torch.cuda.memory_reserved(i)
        except Exception:
            continue

        if split is not None:
            if i < len(split):
                budget += int(split[i] * GIB)
                capacity += total
            continue

        r = reserve[i] if i < len(reserve) else DEFAULT_RESERVE_GB
        if r < 0:
            continue

        budget += max(0, held + free - int(r * GIB))
        capacity += total

    return budget, capacity


def _ram_status():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.available, vm.total
    except Exception:
        return None, None


def _fit_suggestion(gpu_fixed, text_moe, mtp_moe, budget, plan):
    """
    Smallest moe-cpu-split and moe-cpu-layers values that bring the GPU total
    under the budget, as (split, split_ram, layers, layers_ram). Either value
    is None when no setting of that option fits.
    """
    if not text_moe or budget is None:
        return None

    budget -= FIT_MARGIN
    num_experts = text_moe[0][0]

    split_value = split_ram = None
    for k in range(0, num_experts):
        gpu_t, cpu_t, buf_t = _place_experts(text_moe, split=k)
        gpu_m, cpu_m, buf_m = _place_experts(mtp_moe, split=k)
        if gpu_fixed + gpu_t + buf_t + gpu_m + buf_m <= budget:
            split_value, split_ram = k, cpu_t + cpu_m
            break

    draft_layers = plan['draft'].get('moe_cpu_layers', 0)
    gpu_m, cpu_m, buf_m = _place_experts(mtp_moe, offload_layers=draft_layers)
    layers_value = layers_ram = None
    for n in range(0, len(text_moe) + 1):
        gpu_t, cpu_t, buf_t = _place_experts(text_moe, offload_layers=n)
        if gpu_fixed + gpu_t + buf_t + gpu_m + buf_m <= budget:
            layers_value, layers_ram = n, cpu_t + cpu_m
            break

    return split_value, split_ram, layers_value, layers_ram


def estimate(model_dir, plan, model_dir_draft=None):
    """
    Returns {'gpu': {...}, 'ram': {...}, 'disk': {...}, 'notes': [...], ...}
    with byte counts per part, or a dict with an 'error' key if the model
    cannot be inspected.
    """
    try:
        config = _get_config(model_dir)
    except Exception as e:
        return {'error': str(e)}

    gpu, ram, disk = {}, {}, {}
    notes = []
    info = {}

    options = plan['config']
    offload_layers = options.get('moe_cpu_offload', 0)
    split = options.get('moe_cpu_split', 0)
    offload_requested = bool(offload_layers or split or plan['draft'].get('moe_cpu_layers'))

    mul1 = _codebook(model_dir) == 'mul1'
    if offload_requested and not mul1:
        notes.append("This model's experts are not mul1-quantized, so ExLlamaV3 will keep them on the GPU despite the MoE CPU options.")
        offload_layers = split = 0

    chunk = plan['generator'].get('max_chunk_size', 2048) if plan['generator'] else 2048
    counter = _Counter(config)
    text = _get_layout(model_dir, config, 'text')

    if not text['mapped']:
        gpu['Weights'] = _files_size(model_dir)
        notes.append("Could not break the weights down per component; counting every file as VRAM.")
        text_moe = []
    else:
        gpu['Weights'] = counter.size(text['gpu'])
        ram['Embeddings'] = counter.size(text['cpu'])

        ngram_size = counter.size(text['ngram'])
        if ngram_size:
            if options.get('ngram_stream_from_disk') is False:
                ram['N-gram table'] = ngram_size
            else:
                disk['N-gram table'] = ngram_size
                row_bytes, rows_per_token = _ngram_row_bytes(config, text['ngram'])
                if row_bytes and rows_per_token:
                    info['ngram_per_token'] = row_bytes * rows_per_token

        text_moe = _moe_sizes(counter, text['moe'], chunk)
        experts_gpu, experts_cpu, moe_buffers = _place_experts(text_moe, offload_layers, split)
        gpu['Experts'] = experts_gpu
        gpu['MoE buffers'] = moe_buffers
        ram['Experts'] = experts_cpu

    if plan['load_vision'] and not plan['hf'] and 'vision' in config.model_classes and text['mapped']:
        vision = _get_layout(model_dir, config, 'vision')
        if vision['mapped']:
            size = counter.size(vision['gpu'] | vision['cpu'])
            if options.get('vision_pinned'):
                ram['Vision (pinned)'] = size
            else:
                gpu['Vision'] = size

    kv, recurrent = _cache_size(text['model'], plan)
    gpu['KV cache'] = kv
    if recurrent:
        gpu['Recurrent states'] = recurrent

    draft = plan['draft']
    mtp_moe = []
    if not plan['hf'] and draft['mtp'] and 'mtp' in config.model_classes and text['mapped']:
        mtp = _get_layout(model_dir, config, 'mtp')
        if mtp['mapped']:
            gpu['MTP head'] = counter.size(mtp['gpu'])
            ram['Embeddings'] += counter.size(mtp['cpu'])
            mtp_moe = _moe_sizes(counter, mtp['moe'], chunk)
            m_gpu, m_cpu, m_buf = _place_experts(mtp_moe, 0 if split else draft.get('moe_cpu_layers', 0), split)
            gpu['MTP head'] += m_gpu + m_buf
            ram['MTP experts'] = m_cpu

        mtp_kv, mtp_recurrent = _cache_size(mtp['model'], plan, max_history=0)
        gpu['MTP cache'] = mtp_kv + mtp_recurrent
    elif not plan['hf'] and draft['model_draft'] and model_dir_draft:
        try:
            draft_config = _get_config(model_dir_draft)
            draft_layout = _get_layout(model_dir_draft, draft_config, 'text')
            draft_counter = _Counter(draft_config)
            if draft_layout['mapped']:
                d_moe = _moe_sizes(draft_counter, draft_layout['moe'], chunk)
                d_gpu, d_cpu, d_buf = _place_experts(d_moe, draft.get('moe_cpu_layers', 0) if mul1 else 0)
                gpu['Draft weights'] = draft_counter.size(draft_layout['gpu']) + d_gpu + d_buf
                ram['Draft weights'] = draft_counter.size(draft_layout['cpu']) + d_cpu
            else:
                gpu['Draft weights'] = _files_size(model_dir_draft)

            draft_kv, draft_recurrent = _cache_size(draft_layout['model'], plan, max_history=0)
            gpu['Draft cache'] = draft_kv + draft_recurrent
        except Exception as e:
            notes.append(f"Could not inspect the draft model: {e}")

    gpu['Overhead'] = _overhead(config, plan)

    generator = plan['generator']
    if generator.get('cpu_cache_size'):
        ram['CPU page cache'] = generator['cpu_cache_size']
    if generator.get('recurrent_cache_size') and text['model'].caps.get('recurrent_states'):
        ram['Recurrent checkpoints (max)'] = generator['recurrent_cache_size']

    ram['Runtime'] = HOST_RUNTIME

    gpu = {k: v for k, v in gpu.items() if v}
    ram = {k: v for k, v in ram.items() if v}
    disk = {k: v for k, v in disk.items() if v}

    result = {
        'gpu': gpu,
        'ram': ram,
        'disk': disk,
        'gpu_total': sum(gpu.values()),
        'ram_total': sum(ram.values()),
        'disk_total': sum(disk.values()),
        'notes': notes,
        'info': info,
    }

    budget, capacity = _gpu_budget(plan)
    result['gpu_budget'] = budget
    result['gpu_capacity'] = capacity
    result['ram_available'], result['ram_capacity'] = _ram_status()

    enable_tp = plan['model_load'].get('tensor_p')
    if text_moe and mul1 and not enable_tp and budget is not None:
        experts_on_gpu = gpu.get('Experts', 0) + gpu.get('MoE buffers', 0)
        mtp_on_gpu = 0
        if mtp_moe:
            m_gpu, _, m_buf = _place_experts(mtp_moe, 0 if split else draft.get('moe_cpu_layers', 0), split)
            mtp_on_gpu = m_gpu + m_buf

        gpu_fixed = result['gpu_total'] - experts_on_gpu - mtp_on_gpu
        result['fit'] = _fit_suggestion(gpu_fixed, text_moe, mtp_moe, budget, plan)

    return result


def _fmt(value):
    return f"{value / GIB:.2f} GiB" if value >= GIB else f"{value / MIB:.0f} MiB"


def _row(label, total, parts, limit=None, limit_label=None):
    over = limit is not None and total > limit
    cls = 'value over' if over else 'value'
    html = f"{label}: <span class=\"{cls}\">{_fmt(total)}</span>"
    if limit is not None:
        html += f" <small>of {_fmt(limit)} {limit_label}</small>"

    breakdown = ' + '.join(f"{name} {_fmt(value)}" for name, value in parts.items())
    if breakdown:
        html += f"<br><small>{breakdown}</small>"

    return html


def format_html(result, plan=None):
    """
    Renders the estimate as the HTML shown in the Model tab.
    """
    if not result:
        return "<div id=\"vram-info\">Estimated memory: <span class=\"value\">select a model</span></div>"

    if 'error' in result:
        return f"<div id=\"vram-info\">Estimated memory: <span class=\"value\">unavailable</span><br><small>{result['error']}</small></div>"

    rows = [_row("VRAM", result['gpu_total'], result['gpu'], result.get('gpu_budget'), "usable")]

    ram_available = result.get('ram_available')
    ram_label = None
    if ram_available is not None:
        ram_label = f"free now, {_fmt(result['ram_capacity'])} installed"

    rows.append(_row("RAM", result['ram_total'], result['ram'], ram_available, ram_label))

    if result['disk']:
        disk_row = _row("Disk (streamed)", result['disk_total'], result['disk'])
        per_token = result['info'].get('ngram_per_token')
        if per_token:
            disk_row += f"<br><small>Decoding reads about {per_token / 1024:.1f} KiB of it per token; the OS keeps hot rows in its file cache, so a fast NVMe drive is enough. ngram-ram moves it to RAM.</small>"
        rows.append(disk_row)

    notes = list(result['notes'])
    if plan:
        if plan['model_load'].get('tensor_p') or (plan['model_load'].get('use_per_device') and len(plan['model_load']['use_per_device']) > 1):
            notes.append("VRAM is the total across all GPUs; it is split between them at load time.")

    fit = result.get('fit')
    if fit is not None:
        split_value, split_ram, layers_value, layers_ram = fit
        current_split = plan['config'].get('moe_cpu_split', 0) if plan else 0
        current_layers = plan['config'].get('moe_cpu_offload', 0) if plan else 0
        fits_now = result['gpu_budget'] is not None and result['gpu_total'] <= result['gpu_budget'] - FIT_MARGIN

        options = []
        if split_value is not None:
            options.append(f"moe-cpu-split {split_value} ({_fmt(split_ram)} of experts in RAM)")
        if layers_value is not None:
            options.append(f"moe-cpu-layers {layers_value} ({_fmt(layers_ram)} of experts in RAM)")

        if not options:
            notes.append("Does not fit in VRAM even with every expert on the CPU; lower ctx-size or cache-slots, or disable vision/MTP.")
        elif not fits_now:
            if result['gpu_total'] <= result['gpu_budget']:
                notes.append(f"Less than {_fmt(FIT_MARGIN)} of headroom left, which the loader needs for transient buffers, so this may still fail to load.")

            notes.append("To fit in VRAM, the smallest offload is " + " or ".join(options) + ".")
        elif (current_split and split_value is not None and split_value <= current_split - 8) or (current_layers and layers_value is not None and layers_value < current_layers):
            notes.append("More experts could stay on the GPU: " + " or ".join(options) + " would still fit.")

    html = "<div id=\"vram-info\">" + "<br>".join(rows)
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
        with _lock:
            result = estimate(path, plan, model_dir_draft=model_dir_draft)
    except Exception as e:
        logger.warning(f"Failed to estimate ExLlamaV3 memory usage: {e}")
        result = {'error': str(e)}

    return format_html(result, plan)
