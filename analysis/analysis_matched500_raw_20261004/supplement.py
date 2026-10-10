"""Supplement raw audit with shared-query checks and failure categories."""
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def digest_docs(ds):
    return hashlib.sha256(json.dumps([d['contents'] for d in ds], ensure_ascii=False).encode('utf-8')).hexdigest()


def main():
    allqueries, report, roots_by_run = {}, {}, {}
    for label in ['old500', 'new500']:
        queries, counts, cases, root_inside = defaultdict(set), Counter(), [], Counter()
        roots_by_run[label] = {r['sample_id']: r for r in rows(BASE / 'input' / label / 'roots/roots.merged.jsonl.gz')}
        for root in roots_by_run[label].values():
            a = root['action_attempts'][-1]
            text = a['output']
            pos = text.rfind('<search>')
            if pos >= 0 and text.rfind('<think>', 0, pos) > text.rfind('</think>', 0, pos):
                root_inside['root_search_inside_unclosed_think'] += 1
        for row in rows(BASE / 'input' / label / 'trajectories/trajectories.merged.jsonl.gz'):
            for rnd in row['search_rounds']:
                queries[rnd['query']].add(digest_docs(rnd['evidence']['retrieved_docs']))
            if row['answer_status'] == 'local_fallback':
                a = row['decision_events'][-1]['attempts'][-1]
                counts['fallback_' + a['finish_reason']['type']] += 1
                if a['finish_reason']['type'] != 'length':
                    text = a['output']
                    candidates = re.findall(r'<(search|answer)>\s*(.*?)\s*</\1>', text, re.S | re.I)
                    if not candidates:
                        counts['fallback_stop_no_complete_tag'] += 1
                    elif candidates[-1][0].lower() == 'search' and len(' '.join(candidates[-1][1].split())) > 256:
                        counts['fallback_stop_query_exceeds_256_chars'] += 1
                    else:
                        counts['fallback_stop_other'] += 1
                    cases.append({'sample_id': row['sample_id'], 'budget': row['condition_label'],
                                  'forced_answer': row['decision_events'][-1]['forced_answer'],
                                  'finish_reason': a['finish_reason'], 'output_tail': a['output'][-1500:]})
        allqueries[label] = queries
        report[label] = {'fallback_finish_types': dict(counts), 'stop_fallback_examples': cases,
                         'root_tags': dict(root_inside), 'distinct_queries': len(queries),
                         'queries_with_multiple_doc_results_within_run': sum(len(ds) > 1 for ds in queries.values())}
    common = set(allqueries['old500']) & set(allqueries['new500'])
    changes = [q for q in common if allqueries['old500'][q] != allqueries['new500'][q]]
    report['retrieval_comparison'] = {'shared_exact_query_strings': len(common),
                                     'different_document_content_and_order_sets': len(changes),
                                     'changed_queries': sorted(changes)}
    endpoints = defaultdict(Counter)
    for sid, a in roots_by_run['old500'].items():
        b = roots_by_run['new500'][sid]
        group = 'same_port' if a['model_url'] == b['model_url'] else 'different_port'
        endpoints[group]['questions'] += 1
        endpoints[group]['raw_output_changed'] += a['action_attempts'][0]['raw_output'] != b['action_attempts'][0]['raw_output']
        endpoints[group]['query_changed'] += a['query'] != b['query']
    report['root_endpoint_groups_descriptive_only'] = {k: dict(v) for k, v in endpoints.items()}
    details = json.loads((BASE / 'results/per_question_answer_em.json').read_text(encoding='utf-8'))
    losses = [r for r in details if r['old_correct'] and not r['transfer_correct']]
    report['EM_losses'] = {'total': len(losses),
        'any_root_or_continuation_length_or_format': sum(r['either_root_length'] or r['old_length'] or r['new_length'] or r['old_fallback'] or r['new_fallback'] for r in losses),
        'first_input_ids_same_losses': [r['sample_id'] for r in losses if r['first_input_ids_same_reference_tokenizer']],
        'format_or_truncation_loss_ids': [r['sample_id'] for r in losses if r['new_length'] or r['new_fallback']]}
    (BASE / 'results/SUPPLEMENT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: {a: v for a, v in val.items() if a != 'stop_fallback_examples'} for k, val in report.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
