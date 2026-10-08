"""Native OLMoE development entry point; smoke results are engineering-only."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from lm_data import BatchStream, evaluate_lm, prepare_documents, split_manifest, tokenize_documents
from model_runtime import ExpertAdamW, LearnerCheckpoint, configure_trainable_experts

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.glob('*.py'))}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hardware():
    versions = {}
    for name in ['torch', 'numpy', 'transformers', 'safetensors', 'huggingface-hub']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(dict(index=index, name=props.name, memory_bytes=props.total_memory))
    return dict(python=sys.version, platform=platform.platform(), versions=versions,
                cuda_available=torch.cuda.is_available(), cuda_devices=devices,
                mps_available=torch.backends.mps.is_available(), threads=torch.get_num_threads())


def pure_lm_loss(logits, labels):
    """Auxiliary routing loss is deliberately excluded from the objective."""
    targets = labels[:, 1:].contiguous()
    count = int((targets != -100).sum())
    if not count:
        raise ValueError('batch has no predicted tokens')
    loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           targets.reshape(-1), ignore_index=-100, reduction='sum')
    return loss, count


class RoutingCounter:
    """Observe actual token routing without modifying the forward computation."""
    def __init__(self, model, selected_layers, selected_experts):
        self.selected_layers = selected_layers
        self.selected_experts = selected_experts
        self.mask = None
        self.counts = {}
        self.handles = []
        self.top_k = model.config.num_experts_per_tok
        for index, layer in enumerate(model.model.layers):
            self.handles.append(layer.mlp.gate.register_forward_hook(self._hook(index)))

    def _hook(self, index):
        def record(_module, _inputs, output):
            values = output.detach().reshape(-1, output.shape[-1])
            if self.mask is not None:
                mask = self.mask.reshape(-1).bool()
                if len(mask) != len(values):
                    raise RuntimeError('routing mask does not match token logits')
                values = values[mask]
            weights = F.softmax(values, dim=-1, dtype=torch.float32)
            ids = torch.topk(weights, self.top_k, dim=-1).indices.reshape(-1)
            counts = torch.bincount(ids, minlength=output.shape[-1]).cpu()
            self.counts[index] = self.counts.get(index, torch.zeros_like(counts)) + counts
        return record

    def begin(self):
        self.counts = {}

    def summary(self):
        all_counts = {str(layer): values.tolist() for layer, values in self.counts.items()}
        exposure = sum(int(self.counts[layer][self.selected_experts].sum())
                       for layer in self.selected_layers if layer in self.counts)
        total = sum(int(values.sum()) for values in self.counts.values())
        return dict(selected_expert_assignments=exposure, all_expert_assignments=total,
                    layer_expert_assignments=all_counts)

    def close(self):
        for handle in self.handles:
            handle.remove()


def evaluate_roles(model, blocks, config, device, cost):
    result = {}
    for domain in ['A', 'B']:
        for role, limit in [('monitor', config['monitor_blocks_per_domain']),
                            ('selection', config['selection_blocks_per_domain'])]:
            chosen = blocks[domain][role][:limit]
            if not chosen:
                raise ValueError(f'empty {domain}/{role} split')
            # Fixed, whole-block probe pool; no replacement or adaptive selection.
            batches = [dict(input_ids=x['input_ids'].to(device).unsqueeze(0),
                            attention_mask=torch.ones_like(x['input_ids'], device=device).unsqueeze(0),
                            labels=x['input_ids'].to(device).unsqueeze(0)) for x in chosen]
            stats = evaluate_lm(model, batches)
            result[f'{domain}_{role}'] = stats
            cost['evaluation_tokens'] += stats['tokens']
            cost['evaluation_calls'] += 1
            cost['evaluation_seconds'] += stats['seconds']
    return result


def update(model, optimizer, stream, multiplier, config, device, routing, cost):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    batches = [stream.next_batch(config['micro_batch'], device)
               for _ in range(config['gradient_accumulation'])]
    total_tokens = sum(int((b['labels'][:, 1:] != -100).sum()) for b in batches)
    if total_tokens == 0:
        raise ValueError('empty gradient accumulation')
    model.train()
    optimizer.zero_grad()
    routing.begin()
    train_loss = 0.
    for batch in batches:
        routing.mask = batch['attention_mask']
        output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                       use_cache=False, output_router_logits=False)
        loss_sum, count = pure_lm_loss(output.logits, batch['labels'])
        if loss_sum.requires_grad:
            (loss_sum / total_tokens).backward()
        train_loss += float(loss_sum.detach())
    routing.mask = None
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], config['gradient_clip_norm'])
    if not torch.isfinite(grad_norm):
        raise RuntimeError('non-finite gradient')
    displacement = optimizer.step(multiplier)
    batch_digest = hashlib.sha256()
    for batch in batches:
        for key in ['input_ids', 'attention_mask', 'labels']:
            value = batch[key].detach().cpu().contiguous()
            batch_digest.update(str((key, list(value.shape), str(value.dtype))).encode())
            batch_digest.update(value.numpy().tobytes())
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    cost['learner_updates'] += 1
    cost['train_tokens'] += total_tokens
    cost['training_seconds'] += elapsed
    return dict(train_CE=train_loss / total_tokens, tokens=total_tokens,
                gradient_norm_before_clip=float(grad_norm), displacement=displacement, routing=routing.summary(),
                batch_digest=batch_digest.hexdigest(), seconds=elapsed)


def fake_blocks(seed, length):
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for domain in ['A', 'B']:
        result[domain] = {}
        for role in ['train', 'monitor', 'selection']:
            result[domain][role] = [dict(input_ids=torch.randint(3, 97, (length,), generator=generator),
                document_id=f'{domain}-{role}-{i}', source=f'{domain}-{role}-{i}', domain=domain)
                for i in range(12)]
    return result


def fixed_probe_pools(blocks, config):
    """Select fixed probes across source families, avoiding one long article."""
    result = copy.copy(blocks)
    manifest = {}
    for domain_index, domain in enumerate(['A', 'B']):
        result[domain] = dict(blocks[domain])
        manifest[domain] = {}
        for role_index, (role, count) in enumerate([
                ('monitor', config['monitor_blocks_per_domain']),
                ('selection', config['selection_blocks_per_domain'])]):
            rng = random.Random(1701 + domain_index * 10 + role_index)
            groups = {}
            for block in blocks[domain][role]:
                groups.setdefault(block['source'], []).append(block)
            sources = sorted(groups)
            rng.shuffle(sources)
            for group in groups.values():
                rng.shuffle(group)
            chosen = []
            cursor = 0
            while sources and len(chosen) < count:
                remaining = []
                for source in sources:
                    if cursor < len(groups[source]):
                        chosen.append(groups[source][cursor])
                        remaining.append(source)
                        if len(chosen) == count:
                            break
                sources = remaining
                cursor += 1
            if len(chosen) < count:
                raise ValueError(f'{domain}/{role} has fewer than {count} probe blocks')
            result[domain][role] = chosen
            manifest[domain][role] = [dict(document_id=b['document_id'], source=b['source'],
                block_index=b.get('block_index', 0), predicted_tokens=len(b['input_ids']) - 1)
                for b in chosen]
    return result, manifest


def validate_protocol(protocol):
    cfg = protocol['development']
    for key in ['max_steps', 'evaluation_interval', 'sequence_length', 'micro_batch',
                'gradient_accumulation', 'branch_steps', 'monitor_blocks_per_domain',
                'selection_blocks_per_domain']:
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['sequence_length'] < 2:
        raise ValueError('sequence length must allow next-token prediction')
    if not np.isfinite(cfg['learning_rate']) or cfg['learning_rate'] <= 0:
        raise ValueError('learning rate must be positive and finite')
    if not np.isfinite(cfg['weight_decay']) or cfg['weight_decay'] < 0:
        raise ValueError('weight decay must be non-negative and finite')
    if not np.isfinite(cfg['gradient_clip_norm']) or cfg['gradient_clip_norm'] <= 0:
        raise ValueError('gradient clipping norm must be positive and finite')
    if cfg.get('replay_blocks') != 0:
        raise ValueError('this development implementation has no replay; cannot silently ignore it')
    if cfg.get('behavior_multiplier') != .5:
        raise ValueError('this protocol version fixes branching behavior at multiplier .5')
    for key in ['roots', 'branch_futures', 'checkpoint_steps']:
        values = cfg[key]
        if not values or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in values) or len(set(values)) != len(values):
            raise ValueError(f'{key} must contain distinct non-negative integers')
    actions = cfg['actions']
    if actions != [0., .25, .5, .75, 1.]:
        raise ValueError('v10 preflight action grid must be [0,.25,.5,.75,1]')
    if any(step > cfg['max_steps'] or step % cfg['evaluation_interval'] for step in cfg['checkpoint_steps']):
        raise ValueError('branch checkpoints must be in budget and have fresh monitor evaluations')


def load_model(protocol, smoke, device, allow_download):
    from transformers import AutoTokenizer, OlmoeConfig, OlmoeForCausalLM
    if smoke:
        config = OlmoeConfig(vocab_size=97, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            num_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
            attention_dropout=0.0, use_cache=False, pad_token_id=0)
        return OlmoeForCausalLM(config).to(device), None
    spec = protocol['model']
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'],
                                            local_files_only=not allow_download)
    model = OlmoeForCausalLM.from_pretrained(spec['id'], revision=spec['revision'],
        torch_dtype=torch.bfloat16, attn_implementation=spec['attention'],
        local_files_only=not allow_download).to(device)
    model.config.use_cache = False
    return model, tokenizer


def run_root(model, blocks, protocol, seed, mode, out):
    cfg = protocol['development']
    device = next(model.parameters()).device
    optimizer = ExpertAdamW(model.named_parameters(), lr=cfg['learning_rate'],
                           weight_decay=cfg['weight_decay'])
    stream = BatchStream(blocks['B']['train'], seed=seed)
    cost = dict(learner_updates=0, train_tokens=0, evaluation_tokens=0,
                evaluation_calls=0, restore_seconds=0., training_seconds=0.,
                evaluation_seconds=0., checkpoint_seconds=0.)
    capture_start = time.perf_counter()
    initial = LearnerCheckpoint.capture(model, optimizer, stream.state_dict())
    cost['checkpoint_seconds'] += time.perf_counter() - capture_start
    initial_eval = evaluate_roles(model, blocks, cfg, device, cost)
    results, branches = [], []
    # Preflight is a bounded baseline pass. No confirmatory policy is fitted here.
    methods = [('global_0', 0.), ('global_0.5', .5), ('global_1', 1.)] if mode == 'preflight' else [
        (f'global_{m:g}', m) for m in cfg['actions']] + [('time_decay', None)]
    checkpoints = cfg['checkpoint_steps'] if mode in ['preflight', 'branches', 'smoke'] else []
    for method, multiplier in methods:
        restore_start = time.perf_counter()
        stream.load_state_dict(initial.restore(model, optimizer))
        cost['restore_seconds'] += time.perf_counter() - restore_start
        evaluations = initial_eval
        previous_B = evaluations['B_monitor']['loss']
        last_multiplier = 0.
        trajectory_tokens = 0
        last_routing = 0
        trajectory = []
        for step in range(cfg['max_steps'] + 1):
            if method == 'global_0.5' and step in checkpoints:
                capture_start = time.perf_counter()
                checkpoint = LearnerCheckpoint.capture(model, optimizer, stream.state_dict())
                cost['checkpoint_seconds'] += time.perf_counter() - capture_start
                state_id = f'{seed}:{step}'
                features = dict(B_monitor_CE=evaluations['B_monitor']['loss'],
                    B_monitor_CE_change=evaluations['B_monitor']['loss'] - previous_B,
                    optimizer_steps=step, training_tokens=trajectory_tokens,
                    remaining_steps=cfg['max_steps'] - step, last_multiplier=last_multiplier,
                    selected_expert_token_exposure=last_routing,
                    A_monitor_change=evaluations['A_monitor']['loss'] - initial_eval['A_monitor']['loss'])
                for future in cfg['branch_futures']:
                    future_stream = BatchStream(blocks['B']['train'], seed=seed * 1000 + step * 10 + future)
                    future_state = future_stream.state_dict()
                    for action in cfg['actions']:
                        counter_before = dict(cost)
                        restore_start = time.perf_counter()
                        checkpoint.restore(model, optimizer)
                        future_stream.load_state_dict(copy.deepcopy(future_state))
                        cost['restore_seconds'] += time.perf_counter() - restore_start
                        updates = []
                        routing = RoutingCounter(model, protocol['model']['layers'], protocol['model']['experts'])
                        try:
                            for _ in range(cfg['branch_steps']):
                                updates.append(update(model, optimizer, future_stream, action, cfg, device, routing, cost))
                        finally:
                            routing.close()
                        endpoint = evaluate_roles(model, blocks, cfg, device, cost)
                        row = dict(root=seed, state_id=state_id, checkpoint_step=step,
                            future=future, action=action, features=features,
                            baseline=evaluations, endpoint=endpoint, updates=updates,
                            B_selection_improvement=evaluations['B_selection']['loss'] - endpoint['B_selection']['loss'],
                            A_selection_deterioration=endpoint['A_selection']['loss'] - evaluations['A_selection']['loss'],
                            cost={k:cost[k] - counter_before[k] for k in cost})
                        branches.append(row)
                        save(out / f'branches-{seed}.json', branches)
                restore_start = time.perf_counter()
                stream.load_state_dict(checkpoint.restore(model, optimizer))
                cost['restore_seconds'] += time.perf_counter() - restore_start
            if step == cfg['max_steps']:
                break
            action = multiplier if multiplier is not None else 1. - .75 * step / max(1, cfg['max_steps'] - 1)
            routing = RoutingCounter(model, protocol['model']['layers'], protocol['model']['experts'])
            try:
                stats = update(model, optimizer, stream, action, cfg, device, routing, cost)
            finally:
                routing.close()
            trajectory_tokens += stats['tokens']
            last_routing = stats['routing']['selected_expert_assignments']
            last_multiplier = action
            if (step + 1) % cfg['evaluation_interval'] == 0 or step + 1 == cfg['max_steps']:
                previous_B = evaluations['B_monitor']['loss']
                evaluations = evaluate_roles(model, blocks, cfg, device, cost)
            trajectory.append(dict(step=step + 1, action=action, **stats,
                evaluation=evaluations if (step + 1) % cfg['evaluation_interval'] == 0 or step + 1 == cfg['max_steps'] else None))
        results.append(dict(root=seed, method=method, initial=initial_eval, endpoint=evaluations, trajectory=trajectory))
        save(out / f'baselines-{seed}.json', results)
        print(f'root={seed} method={method} B-selection={evaluations["B_selection"]["loss"]:.5f} '
              f'A-selection={evaluations["A_selection"]["loss"]:.5f}', flush=True)
    restore_start = time.perf_counter()
    initial.restore(model, optimizer)
    cost['restore_seconds'] += time.perf_counter() - restore_start
    return dict(root=seed, baseline_trajectories=len(results), branches=len(branches), cost=cost)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['inspect', 'smoke', 'preflight', 'baselines', 'branches'], default='inspect')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=HERE / 'protocol.json')
    parser.add_argument('--data', type=Path, help='Prepared source-grouped JSONL documents.')
    parser.add_argument('--download', action='store_true', help='Allow pinned model/tokenizer download on the GPU machine.')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists() or (ROOT / 'experiments') in args.out.resolve().parents:
        parser.error('choose a fresh output directory outside experiments/')
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    args.out.mkdir(parents=True)
    env = hardware()
    save(args.out / 'environment.json', env)
    if args.mode == 'inspect':
        save(args.out / 'status.json', dict(status='ready_for_gpu_preflight' if env['cuda_available'] else 'gpu_connection_required',
            scientific_result=False, model_revision=protocol['model']['revision'], environment=env))
        print(json.dumps(dict(cuda_available=env['cuda_available'], devices=env['cuda_devices']), ensure_ascii=False))
        return
    smoke = args.mode == 'smoke'
    if not smoke and (not torch.cuda.is_available() or not args.device.startswith('cuda')):
        save(args.out / 'status.json', dict(status='blocked_no_cuda', scientific_result=False))
        parser.error('native pretrained runs require CUDA; use --mode smoke for CPU mechanism checks')
    if not smoke and args.data is None:
        parser.error('--data is required for native pretrained runs')
    if not smoke:
        with torch.cuda.device(args.device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                parser.error('native preflight requires hardware BF16 support')
    if smoke:
        protocol = copy.deepcopy(protocol)
        protocol['model'].update(layers=[0, 1], experts=[0, 1], dtype='float32',
                                 id='random-tiny-HuggingFace-OLMoE', revision=None)
        protocol['development'].update(roots=[100201], max_steps=4, evaluation_interval=2,
            sequence_length=16, micro_batch=1, gradient_accumulation=1,
            checkpoint_steps=[0, 2], branch_steps=2, monitor_blocks_per_domain=2,
            selection_blocks_per_domain=2, learning_rate=.001)
        torch.set_num_threads(1)
    elif args.mode == 'preflight':
        protocol['development'].update(roots=protocol['development']['roots'][:1],
            max_steps=2, evaluation_interval=1, checkpoint_steps=[1],
            branch_steps=1, branch_futures=[0])
    cfg = protocol['development']
    validate_protocol(protocol)
    protocol['execution'] = dict(mode=args.mode, pretrained_model_run=not smoke,
        confirmatory=False, source_hashes=source_hashes(), official_test_evaluated=False)
    if args.data:
        protocol['execution']['data_sha256'] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    save(args.out / 'locked_protocol.json', protocol)
    start = time.perf_counter()
    device = 'cpu' if smoke else args.device
    seed_all(cfg['roots'][0])
    model, tokenizer = load_model(protocol, smoke, device, args.download)
    report = configure_trainable_experts(model, protocol['model']['layers'], protocol['model']['experts'])
    save(args.out / 'trainable_parameters.json', report)
    if smoke:
        blocks = fake_blocks(4701, cfg['sequence_length'])
        save(args.out / 'data_manifest.json', dict(synthetic_tokens=True, final_test_loaded=False))
    else:
        prepared = prepare_documents(args.data, seed=1701)
        save(args.out / 'data_manifest.json', split_manifest(prepared, seed=1701))
        blocks = {domain:{role:tokenize_documents(prepared[domain][role], tokenizer,
            cfg['sequence_length']) for role in ['train', 'monitor', 'selection']} for domain in ['A', 'B']}
    blocks, probes = fixed_probe_pools(blocks, cfg)
    save(args.out / 'probe_blocks.json', probes)
    if not smoke:
        torch.cuda.reset_peak_memory_stats(device)
    runs = []
    for root in cfg['roots']:
        seed_all(root)
        runs.append(run_root(model, blocks, protocol, root, args.mode, args.out))
    if source_hashes() != protocol['execution']['source_hashes']:
        raise RuntimeError('source changed during locked run')
    elapsed = time.perf_counter() - start
    totals = {key:sum(run['cost'][key] for run in runs) for key in runs[0]['cost']}
    save(args.out / 'summary.json', dict(status='engineering_only' if smoke else 'development_only',
        model=protocol['model'], runs=runs, physical_cost=totals, wall_seconds=elapsed,
        overall_train_tokens_per_second=totals['train_tokens'] / elapsed,
        training_tokens_per_second=totals['train_tokens'] / totals['training_seconds'],
        peak_cuda_allocated_bytes=0 if smoke else torch.cuda.max_memory_allocated(device),
        peak_cuda_reserved_bytes=0 if smoke else torch.cuda.max_memory_reserved(device),
        feedback_benefit_tested=False, final_test_evaluated=False,
        caveat='Random tiny architecture smoke is not a pretrained language result.' if smoke else
        'Fixed expert-group development experiment; no policy learning or confirmatory claim.'))
    print(f'Completed {args.mode}: {args.out}', flush=True)


if __name__ == '__main__':
    main()
