#!/usr/bin/env python3
"""Export both historical HotpotQA runs, including unabridged raw records.

Python 3.8+ standard library only. Reads one JSONL record at a time. It never
contacts model/retriever services and never changes source experiment files.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

BASE = Path('/home/researcher/smartsearch_rebuild')
OLD = BASE / 'outputs/hotpot_concat100_random500_v4/random500_top5_ctx40960_v43_g2g3_seed2026'
NEW = BASE / 'outputs/hotpot_budget_full7405_v433/full7405_fresh_v433_20260924'
RAW_FILES = ['roots/roots.merged.jsonl.gz', 'trajectories/trajectories.merged.jsonl.gz']
OPTIONAL_FILES = [
    'roots/roots.merged.jsonl.gz.manifest.json',
    'trajectories/trajectories.merged.jsonl.gz.manifest.json',
    'analysis/scored_records.jsonl.gz', 'analysis/summary.json',
    'FULL7405_CONFIG.json', 'last_service_check.json', 'PIPELINE_STATUS.json',
]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def canonical(value):
    return digest_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  separators=(',', ':')))


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_signature(path):
    s = path.stat()
    return (s.st_size, s.st_mtime_ns, s.st_ino)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                    require(isinstance(value, dict), 'record is not an object')
                    yield value
                except (ValueError, TypeError) as exc:
                    raise ValueError('{} line {}: {}'.format(path, number, exc)) from exc


def check_record_hash(row, field, location, counters):
    if field not in row:
        counters['missing_' + field] += 1
        return
    payload = {k: v for k, v in row.items() if k != field}
    require(canonical(payload) == row[field], location + ': stored record hash mismatch')


def check_attempts(attempts, location, counters):
    require(isinstance(attempts, list) and len(attempts) > 0,
            location + ': missing raw generation attempts')
    fingerprints = []
    for i, a in enumerate(attempts):
        loc = '{} attempt {}'.format(location, i)
        for key in ('prompt', 'raw_output', 'output'):
            require(isinstance(a.get(key), str), loc + ': missing raw field ' + key)
        require('finish_reason' in a, loc + ': missing finish_reason')
        for text_field, hash_field in [('prompt', 'prompt_sha256'), ('output', 'output_sha256')]:
            if hash_field in a:
                require(digest_text(a[text_field]) == a[hash_field], loc + ': ' + hash_field + ' mismatch')
        reason = a['finish_reason']
        finish = reason.get('type', 'unknown') if isinstance(reason, dict) else str(reason or 'unknown')
        counters['finish_' + finish] += 1
        counters['model_attempts'] += 1
        counters['empty_raw_output'] += int(not a['raw_output'].strip())
        counters['reported_input_tokens'] += int(a.get('input_tokens', 0))
        counters['reported_output_tokens'] += int(a.get('output_tokens', 0))
        fingerprints.append((digest_text(a['prompt']), digest_text(a['raw_output'])))
    return fingerprints


def check_documents(documents, location):
    require(isinstance(documents, list) and len(documents) > 0,
            location + ': missing complete retrieved documents')
    values = []
    for d in documents:
        require(isinstance(d.get('contents'), str) and bool(d['contents']),
                location + ': retrieved document has no contents')
        h = digest_text(d['contents'])
        if 'contents_sha256' in d:
            require(d['contents_sha256'] == h, location + ': document hash mismatch')
        values.append(h)
    # Content and list order are compared independently of IDs and timings.
    return canonical(values)


def inspect_run(run, expected_roots, wanted_ids=None):
    root_path, trajectory_path = [run / p for p in RAW_FILES]
    for p in (root_path, trajectory_path):
        require(p.is_file(), 'Missing RAW file: {}\nA .manifest.json file is not its replacement.'.format(p))
    root_stats, trajectory_stats = Counter(), Counter()
    roots, keys, pairs = {}, set(), {}
    for row in rows(root_path):
        sid = str(row['sample_id'])
        require(sid not in roots, 'Duplicate root ' + sid)
        require(isinstance(row.get('question'), str), sid + ': missing question')
        require(isinstance(row.get('query'), str), sid + ': missing query')
        check_record_hash(row, 'root_checkpoint_sha256', sid, root_stats)
        attempts = check_attempts(row.get('action_attempts'), sid, root_stats)
        docs_hash = check_documents(row.get('documents'), sid)
        roots[sid] = {'question_sha256': digest_text(row['question']),
                      'query': row['query'], 'documents_content_order_sha256': docs_hash,
                      'generation_hashes': attempts,
                      'root_checkpoint_sha256': row.get('root_checkpoint_sha256')}
        root_stats['roots'] += 1
        root_stats['query_fallback'] += int(row.get('query_parse_mode') == 'fallback_question_as_query')
    if expected_roots is not None:
        require(len(roots) == expected_roots,
                '{}: expected {} roots, found {}'.format(run, expected_roots, len(roots)))
    wanted = set(roots) if wanted_ids is None else set(wanted_ids)
    require(wanted <= set(roots), '{}: missing {} matched IDs'.format(run, len(wanted - set(roots))))
    print('[raw roots] {}: {}'.format(run.name, len(roots)), flush=True)
    last_progress = time.monotonic()
    for row in rows(trajectory_path):
        sid = str(row['sample_id'])
        require(sid in roots, 'Trajectory without root: ' + sid)
        key = (sid, str(row['condition_id']))
        require(key not in keys, 'Duplicate trajectory: ' + str(key))
        keys.add(key)
        require(isinstance(row.get('final_answer'), str), str(key) + ': missing final_answer')
        if roots[sid]['root_checkpoint_sha256'] is not None:
            require(row.get('root_checkpoint_sha256') == roots[sid]['root_checkpoint_sha256'],
                    str(key) + ': root checkpoint mismatch')
        check_record_hash(row, 'record_sha256', str(key), trajectory_stats)
        events, searches = row.get('decision_events'), row.get('search_rounds')
        require(isinstance(events, list) and len(events) > 0, str(key) + ': missing decision_events')
        require(isinstance(searches, list) and len(searches) > 0, str(key) + ': missing search_rounds')
        generations = []
        for event in events:
            generations.extend(check_attempts(event.get('attempts'), str(key), trajectory_stats))
        for search in searches:
            evidence = search.get('evidence') or {}
            require(isinstance(evidence.get('evidence_text'), str), str(key) + ': missing visible evidence text')
            check_documents(evidence.get('retrieved_docs'), str(key))
            trajectory_stats['search_rounds'] += 1
        trajectory_stats['records'] += 1
        trajectory_stats['fallback_records'] += int(row.get('answer_status') == 'local_fallback')
        trajectory_stats['context_compacted_records'] += int(int(row.get('context_compactions', 0)) > 0)
        if sid in wanted:
            budget = 'FULL' if row.get('exposure_mode') == 'full_per_search' else str(int(row['root_exposure_tokens']))
            pair_key = (sid, budget)
            require(pair_key not in pairs, 'Duplicate matched budget: ' + str(pair_key))
            pairs[pair_key] = {'generation_hashes': generations,
                              'answer_sha256': digest_text(row['final_answer']),
                              'answer_status': row.get('answer_status'),
                              'search_calls': row.get('search_calls')}
        if time.monotonic() - last_progress >= 10:
            print('[scan] {}: {} complete raw trajectories'.format(run.name, len(keys)), flush=True)
            last_progress = time.monotonic()
    require({sid for sid, _ in keys} == set(roots), str(run) + ': roots with no trajectories')
    for relative, actual, field in [(RAW_FILES[0], len(roots), 'roots'), (RAW_FILES[1], len(keys), 'records')]:
        manifest = run / (relative + '.manifest.json')
        if manifest.exists():
            with manifest.open(encoding='utf-8') as f:
                recorded = json.load(f)
            require(recorded.get(field) == actual, '{}: count differs from manifest'.format(relative))
    return {'source_run': str(run), 'roots': dict(root_stats),
            'trajectories': dict(trajectory_stats)}, roots, pairs


def compare_runs(old_roots, new_roots, old_pairs, new_pairs):
    root_counts, pair_counts = Counter(), Counter()
    for sid, a in old_roots.items():
        b = new_roots[sid]
        root_counts['matched_questions'] += 1
        root_counts['same_question_text'] += int(a['question_sha256'] == b['question_sha256'])
        root_counts['same_first_prompt'] += int(a['generation_hashes'][0][0] == b['generation_hashes'][0][0])
        root_counts['same_all_root_generation_prompts'] += int([v[0] for v in a['generation_hashes']] == [v[0] for v in b['generation_hashes']])
        root_counts['same_all_root_raw_outputs'] += int([v[1] for v in a['generation_hashes']] == [v[1] for v in b['generation_hashes']])
        root_counts['same_query'] += int(a['query'] == b['query'])
        root_counts['same_retrieved_contents_and_order'] += int(a['documents_content_order_sha256'] == b['documents_content_order_sha256'])
    for key in sorted(set(old_pairs) & set(new_pairs)):
        a, b = old_pairs[key], new_pairs[key]
        ga, gb = a['generation_hashes'], b['generation_hashes']
        pair_counts['common_question_budget_pairs'] += 1
        pair_counts['same_first_continuation_prompt'] += int(ga[0][0] == gb[0][0])
        pair_counts['same_final_answer_exact'] += int(a['answer_sha256'] == b['answer_sha256'])
        pair_counts['same_search_count'] += int(a['search_calls'] == b['search_calls'])
        pair_counts['same_prompt_but_different_first_raw_output'] += int(ga[0][0] == gb[0][0] and ga[0][1] != gb[0][1])
    pair_counts['old_only_question_budget_pairs'] = len(set(old_pairs) - set(new_pairs))
    pair_counts['new_only_question_budget_pairs'] = len(set(new_pairs) - set(old_pairs))
    return {'root_comparison': dict(root_counts), 'trajectory_comparison': dict(pair_counts),
            'interpretation': 'Hashes compare stored text. Different outputs for equal prompt text do not by themselves identify a runtime cause. Root thinking is not passed into continuation prompts by this protocol. FULL denotes the full retrieved set, not equal tokens across runs.'}


def add_bytes(archive, name, content):
    data = content.encode('utf-8')
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(data), 0o644
    archive.addfile(info, io.BytesIO(data))


class HashingReader:
    def __init__(self, stream):
        self.stream, self.hash = stream, hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.hash.update(data)
        return data


def make_archive(destination, inventory, report):
    partial = destination.with_name(destination.name + '.partial')
    require(not destination.exists() and not partial.exists(), 'Output already exists: ' + str(destination))
    with partial.open('xb') as sink:
        with tarfile.open(fileobj=sink, mode='w:gz', compresslevel=1) as archive:
            for item in inventory:
                path = Path(item['source'])
                require(file_signature(path) == item['_signature'], 'Source changed during audit: ' + str(path))
                print('[pack] {} ({:.1f} MiB)'.format(item['archive_name'], item['bytes'] / 1024**2), flush=True)
                info = archive.gettarinfo(str(path), arcname=item['archive_name'])
                require(info.isfile(), 'Only regular data files are supported: ' + str(path))
                with path.open('rb') as f:
                    reader = HashingReader(f)
                    archive.addfile(info, reader)
                    require(reader.hash.hexdigest() == item['sha256'], 'Source checksum changed: ' + str(path))
                require(file_signature(path) == item['_signature'], 'Source changed during packaging: ' + str(path))
            public_inventory = [{k: v for k, v in item.items() if k != '_signature'} for item in inventory]
            add_bytes(archive, 'RAW_EXPORT_MANIFEST.json', json.dumps({
                'created_at': stamp(), 'unabridged_raw_records': True,
                'files': public_inventory, 'new_model_calls': 0,
                'scope': report.get('scope', 'old500_and_full7405'),
                'selected_sample_ids': report.get('selected_sample_ids'),
            }, ensure_ascii=False, indent=2))
            add_bytes(archive, 'RAW_PREFLIGHT_REPORT.json', json.dumps(report, ensure_ascii=False, indent=2))
            selection_note = (
                'old500/ and new500/ contain only the SAME selected questions, with ALL saved budget trajectories.\n'
                'Selection uses sample_id only, never answer quality or validity. Budget grids may differ.\n'
                'Gzip containers are newly compressed; every selected JSONL record line is preserved byte for byte.\n'
                'No unrelated question records or full-run quality summaries are included.\n'
                if report.get('scope') == 'matched500_only' else
                'old500/ and full7405/ contain the original unmodified gzip files.\n')
            add_bytes(archive, 'README.txt',
                      selection_note +
                      'Raw prompts and outputs are in root action_attempts and trajectory decision_events[].attempts.\n'
                      'Complete retrieved contents and visible evidence are in documents/search_rounds.\n'
                      'RAW_PREFLIGHT_REPORT.json is a structural/text comparison, not an online replication experiment.\n'
                      'Historical runtime versions or missing logs are not reconstructed or invented.\n')
    partial.rename(destination)
    checksum = digest_file(destination)
    with destination.with_name(destination.name + '.sha256').open('x', encoding='utf-8') as f:
        f.write(checksum + '  ' + destination.name + '\n')
    return checksum


def filter_jsonl(source, destination, wanted_ids):
    """Stream-select complete original JSONL lines, not reconstructed JSON objects."""
    require(source.is_file() and not source.is_symlink(), 'Missing/nonregular RAW file: ' + str(source))
    require(not destination.exists(), 'Output already exists: ' + str(destination))
    signature = file_signature(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scanned, selected, selected_bytes = 0, 0, 0
    selected_hash, seen = hashlib.sha256(), set()
    last_progress = time.monotonic()
    with gzip.open(source, 'rb') as src, destination.open('xb') as out:
        with gzip.GzipFile(fileobj=out, mode='wb', filename='', mtime=0, compresslevel=1) as dst:
            for number, line in enumerate(src, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    require(isinstance(row, dict) and 'sample_id' in row,
                            'record must be an object with sample_id')
                    sid = str(row['sample_id'])
                except (ValueError, TypeError) as exc:
                    raise ValueError('{} line {}: {}'.format(source, number, exc)) from exc
                scanned += 1
                if sid in wanted_ids:
                    dst.write(line)
                    selected_hash.update(line)
                    selected_bytes += len(line)
                    selected += 1
                    seen.add(sid)
                if time.monotonic() - last_progress >= 10:
                    print('[select] {}: scanned={}, kept={}'.format(source, scanned, selected), flush=True)
                    require(shutil.disk_usage(destination.parent).free >= 64 * 1024**2,
                            'Less than 64 MiB disk free; stopping export before filling disk')
                    last_progress = time.monotonic()
    require(file_signature(source) == signature, 'Source changed while filtering: ' + str(source))
    require(seen == set(wanted_ids), '{}: missing selected IDs: {}'.format(source, sorted(set(wanted_ids) - seen)))
    # Hash the actual exported uncompressed bytes independently of the selection loop.
    verified_hash = hashlib.sha256()
    with gzip.open(destination, 'rb') as check:
        for block in iter(lambda: check.read(1024 * 1024), b''):
            verified_hash.update(block)
    require(verified_hash.hexdigest() == selected_hash.hexdigest(), 'Selected raw bytes changed: ' + str(destination))
    print('[selected] {}: {}/{} records, {} questions'.format(
        source, selected, scanned, len(seen)), flush=True)
    return {'source': str(source), 'source_records_scanned': scanned,
            'selected_records': selected, 'selected_questions': len(seen),
            'selected_uncompressed_bytes': selected_bytes,
            'selected_jsonl_bytes_sha256': selected_hash.hexdigest(),
            'byte_preservation_verified': True}


def export_matched(old_run, new_run, output_dir, expected_questions=500):
    """Export all saved trajectories for the old run's questions in both runs."""
    source_items = []
    # Global scores/summaries are deliberately omitted. Only filtered scored
    # rows and explicitly named provenance/configuration files are eligible.
    filtered_files = RAW_FILES + ['analysis/scored_records.jsonl.gz']
    metadata_files = ['FULL7405_CONFIG.json', 'last_service_check.json', 'PIPELINE_STATUS.json']
    for label, run in [('old500', old_run), ('new500', new_run)]:
        for rel in RAW_FILES:
            require((run / rel).is_file(), 'Missing RAW file: ' + str(run / rel))
        for rel in filtered_files + [p + '.manifest.json' for p in RAW_FILES] + metadata_files:
            path = run / rel
            if path.is_file():
                require(not path.is_symlink(), 'Please supply a regular source file: ' + str(path))
                source_items.append({'label': label, 'relative_path': rel, 'source': str(path),
                                     'bytes': path.stat().st_size, '_signature': file_signature(path)})
    wanted = set()
    for row in rows(old_run / RAW_FILES[0]):
        sid = str(row['sample_id'])
        require(sid not in wanted, 'Duplicate old root: ' + sid)
        wanted.add(sid)
    require(len(wanted) == expected_questions, 'Expected {} old questions, found {}'.format(expected_questions, len(wanted)))
    output_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_dir).free
    require(free >= 128 * 1024**2, 'Less than 128 MiB disk free')
    tag = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid())
    stage = output_dir / ('hotpot_matched500_stage_' + tag)
    stage.mkdir(exist_ok=False)
    print('Export SAME {} questions only; all native budget conditions retained.\n'
          'Free disk {:.2f} GiB; streaming from complete source files.\n'
          'Staging: {}'.format(len(wanted), free / 1024**3, stage), flush=True)
    selection = {}
    for label, run in [('old500', old_run), ('new500', new_run)]:
        selection[label] = {}
        for rel in filtered_files:
            source = run / rel
            if not source.exists():
                continue
            result = filter_jsonl(source, stage / label / rel, wanted)
            selection[label][rel] = result
            if rel in RAW_FILES:
                field = 'roots' if rel == RAW_FILES[0] else 'records'
                original_manifest = run / (rel + '.manifest.json')
                if original_manifest.is_file():
                    with original_manifest.open(encoding='utf-8') as f:
                        declared = json.load(f)
                    require(declared.get(field) == result['source_records_scanned'],
                            'Source count differs from manifest: ' + str(original_manifest))
                subset_manifest = stage / label / (rel + '.manifest.json')
                with subset_manifest.open('x', encoding='utf-8') as f:
                    json.dump({'scope': 'matched500_only', field: result['selected_records'],
                               'source_records': result['source_records_scanned'],
                               'selection': 'sample_id only; all budgets/statuses',
                               'selected_jsonl_bytes_sha256': result['selected_jsonl_bytes_sha256']},
                              f, ensure_ascii=False, indent=2)
    old_report, old_roots, old_pairs = inspect_run(stage / 'old500', expected_questions)
    new_report, new_roots, new_pairs = inspect_run(stage / 'new500', expected_questions)
    require(set(old_roots) == set(new_roots) == wanted, 'Matched question ID sets differ')
    changed_questions = [sid for sid in wanted if old_roots[sid]['question_sha256'] != new_roots[sid]['question_sha256']]
    require(not changed_questions, 'Same ID but different question text: ' + str(changed_questions))
    old_report['source_run'], new_report['source_run'] = str(old_run), str(new_run)
    print('[source checksums] Verifying original input files; no model calls.', flush=True)
    for item in source_items:
        path = Path(item['source'])
        require(file_signature(path) == item['_signature'], 'Source changed during selection: ' + str(path))
        item['sha256'] = digest_file(path)
        require(file_signature(path) == item['_signature'], 'Source changed during checksum: ' + str(path))
    report = {'created_at': stamp(), 'scope': 'matched500_only',
              'raw_field_validation_passed': True, 'selected_sample_ids': sorted(wanted),
              'selection_rule': 'all old500 question IDs must exist with identical question text in new run; retain all saved budgets and answer statuses independently per run',
              'old500': old_report, 'new500': new_report, 'selection': selection,
              'matched500': compare_runs(old_roots, new_roots, old_pairs, new_pairs),
              'source_files': [{k: v for k, v in item.items() if k != '_signature'} for item in source_items],
              'new_model_calls': 0, 'quality_metrics_recomputed': False}
    inventory = []
    for label in ['old500', 'new500']:
        for rel in filtered_files + [p + '.manifest.json' for p in RAW_FILES]:
            path = stage / label / rel
            if path.is_file():
                inventory.append({'source': str(path), 'archive_name': label + '/' + rel,
                                  'bytes': path.stat().st_size, '_signature': file_signature(path),
                                  'sha256': digest_file(path)})
    for item in source_items:
        if item['relative_path'] in metadata_files:
            inventory.append(dict(item, archive_name='source_metadata/' + item['label'] + '/' + item['relative_path']))
    selected_size = sum(item['bytes'] for item in inventory)
    require(shutil.disk_usage(output_dir).free >= selected_size * 1.03 + 64 * 1024**2,
            'Not enough disk space for final archive; validated staging files retained at ' + str(stage))
    destination = output_dir / ('hotpot_matched500_RAW_' + tag + '.tar.gz')
    report_file = output_dir / ('hotpot_matched500_RAW_' + tag + '.precheck.json')
    with report_file.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    checksum = make_archive(destination, inventory, report)
    result = {'status': 'MATCHED500_RAW_EXPORT_COMPLETE', 'archive': str(destination),
              'questions_per_run': len(wanted), 'old_trajectories': len(old_pairs),
              'new_trajectories': len(new_pairs),
              'common_question_budget_pairs': len(set(old_pairs) & set(new_pairs)),
              'size_gib': round(destination.stat().st_size / 1024**3, 3),
              'sha256': checksum, 'precheck': str(report_file), 'staging_directory': str(stage),
              'upload': [str(destination), str(destination) + '.sha256'], 'new_model_calls': 0}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-run', type=Path, default=OLD)
    parser.add_argument('--new-run', type=Path, default=NEW)
    parser.add_argument('--output-dir', type=Path, default=BASE / 'outputs/raw_dialogue_exports')
    parser.add_argument('--matched-only', action='store_true',
                        help='Export only the same 500 questions in both runs, keeping all their budget trajectories')
    args = parser.parse_args()
    if args.matched_only:
        export_matched(args.old_run, args.new_run, args.output_dir)
        return
    inventory = []
    for label, run in [('old500', args.old_run), ('full7405', args.new_run)]:
        for rel in RAW_FILES:
            require((run / rel).is_file(), 'Missing RAW file: ' + str(run / rel))
        for rel in RAW_FILES + OPTIONAL_FILES:
            p = run / rel
            if p.is_file():
                require(not p.is_symlink(), 'Please supply a regular source file: ' + str(p))
                inventory.append({'source': str(p), 'archive_name': label + '/' + rel,
                                  'bytes': p.stat().st_size, '_signature': file_signature(p)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_size = sum(item['bytes'] for item in inventory)
    free = shutil.disk_usage(args.output_dir).free
    require(free >= source_size * 1.03 + 64 * 1024**2,
            'Insufficient free disk space: need about {:.2f} GiB, free {:.2f} GiB'.format(source_size * 1.03 / 1024**3 + .0625, free / 1024**3))
    print('Source size: {:.2f} GiB; free disk: {:.2f} GiB. Streaming validation starts.'.format(source_size / 1024**3, free / 1024**3), flush=True)
    old_report, old_roots, old_pairs = inspect_run(args.old_run, 500)
    new_report, new_roots, new_pairs = inspect_run(args.new_run, 7405, old_roots)
    report = {'created_at': stamp(), 'raw_field_validation_passed': True,
              'old500': old_report, 'full7405': new_report,
              'matched500': compare_runs(old_roots, new_roots, old_pairs, new_pairs),
              'new_model_calls': 0, 'quality_metrics_recomputed': False}
    for item in inventory:
        p = Path(item['source'])
        require(file_signature(p) == item['_signature'], 'Source changed during inspection: ' + str(p))
        item['sha256'] = digest_file(p)
        require(file_signature(p) == item['_signature'], 'Source changed during checksum: ' + str(p))
    tag = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid())
    destination = args.output_dir / ('hotpot_500_and7405_RAW_' + tag + '.tar.gz')
    report_file = args.output_dir / ('hotpot_500_and7405_RAW_' + tag + '.precheck.json')
    with report_file.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    checksum = make_archive(destination, inventory, report)
    print(json.dumps({'status': 'RAW_EXPORT_COMPLETE', 'archive': str(destination),
                      'size_gib': round(destination.stat().st_size / 1024**3, 3),
                      'sha256': checksum, 'precheck': str(report_file),
                      'upload': [str(destination), str(destination) + '.sha256'],
                      'new_model_calls': 0}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('RAW_EXPORT_FAILED: {}: {}'.format(type(exc).__name__, exc), file=sys.stderr)
        print('No source file was modified. A .partial output, if present, is incomplete and must not be uploaded.', file=sys.stderr)
        sys.exit(1)
