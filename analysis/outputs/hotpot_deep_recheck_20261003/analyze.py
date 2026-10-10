"""Explore free pre-exposure signals and observable search-persistence patterns.

No inference calls. All policy estimates select ONE already-observed full path.
Runtime-triggered policy outcomes are NOT estimated by splicing paths.
Retrospective hash split; no claim of an untouched prospective confirmation set.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
SOURCE = WORK / 'audit_hotpot_full7405_v433_20261001/input/analysis/scored_records.jsonl.gz'
GEO = WORK / 'outputs/hotpot_full7405_shape_mechanism_20261002/results/per_question_geometry.jsonl.gz'
LABELS = ['100', '200', '300', '400', '500', '600', 'FULL']
SEED = 20261003


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def norm(text):
    return ' '.join(text.casefold().split())


def words(text):
    return re.findall(r'[a-z0-9]+', text.casefold())


def mean(values, mask=None):
    values = np.asarray(values)
    if mask is not None:
        values = values[mask]
    return float(values.mean()) if values.size else None


def paired_ci(x, draws=2000):
    rng = np.random.default_rng(SEED)
    x = np.asarray(x)
    means = []
    for _ in range(draws // 100):
        ii = rng.integers(0, len(x), size=(100, len(x)))
        means.extend(x[ii].mean(axis=1))
    return np.quantile(means, [.025, .975]).tolist()


data = collections.defaultdict(dict)
for r in load(SOURCE):
    label = 'FULL' if r['exposure_mode'].startswith('full') else str(r['budget'])
    if label in LABELS:
        assert label not in data[r['sample_id']]
        data[r['sample_id']][label] = r
geo = {r['sample_id']: r for r in load(GEO)}
ids = sorted(data)
records = [[data[sid][b] for b in LABELS] for sid in ids]
assert len(ids) == 7405 and all(len(r) == 7 for r in records)
n = len(ids)


def matrix(key):
    return np.array([[r[key] for r in rr] for rr in records], float)


em, acc = matrix('answer_em'), matrix('answer_acc')
inp, out = matrix('standalone_llm_input_tokens'), matrix('standalone_llm_output_tokens')
cost = inp + out
thinking = matrix('standalone_thinking_tokens')
search = matrix('search_calls').astype(int)
ev = matrix('visible_evidence_tokens_total')
clean = ~matrix('local_fallback').astype(bool).any(axis=1)
rootout7 = out - matrix('continuation_llm_output_tokens')
rootin7 = inp - matrix('continuation_llm_input_tokens')
assert np.all(rootout7 == rootout7[:, :1]) and np.all(rootin7 == rootin7[:, :1])
rootout, rootin = rootout7[:, 0], rootin7[:, 0]
typ = np.array([rr[0]['type'] for rr in records])
qseq = [[tuple(norm(s['query']) for s in r['per_search_round_evidence']) for r in rr] for rr in records]
repeat = np.array([[len(q) != len(set(q)) for q in qq] for qq in qseq])
result = {'integrity': {'questions': n, 'records': n*7, 'root_usage_identical_all_conditions': True},
          'limits': ['Single-model retrospective analysis.', 'Root output signal is available only AFTER mandatory shared root generation.',
                     'No gold features or official type used by deployable policies.',
                     'Post-treatment search counts characterize candidates, not valid switched-policy outcomes.',
                     'F1 not used because original scorer needs official yes/no rules.'],
          'repeat_query': [], 'persistence': [], 'root_output_groups': [], 'same_plan_substitution': []}
for j, b in enumerate(LABELS):
    for name, mask in [('repeat_query', repeat[:, j]), ('no_repeat', ~repeat[:, j])]:
        result['repeat_query'].append({'budget': b, 'group': name, 'n': int(mask.sum()),
            'EM': mean(em[:, j], mask), 'ACC': mean(acc[:, j], mask), 'tokens': mean(cost[:, j], mask),
            'searches': mean(search[:, j], mask), 'clean_rate': mean(clean, mask)})

for small, large in [(0, 5), (1, 5), (4, 6)]:
    for subset, mask in [('all', np.ones(n, bool)), ('small_search_le2', search[:, small] <= 2),
                         ('small_search_ge3', search[:, small] >= 3), ('small_repeat', repeat[:, small]),
                         ('small_no_repeat', ~repeat[:, small]), ('both_same_plan',
                         np.array([q[small] == q[large] for q in qseq]))]:
        result['persistence'].append({'pair': f'{LABELS[small]}->{LABELS[large]}', 'group': subset,
            'n': int(mask.sum()), 'small_EM': mean(em[:, small], mask), 'large_EM': mean(em[:, large], mask),
            'small_ACC': mean(acc[:, small], mask), 'large_ACC': mean(acc[:, large], mask),
            'EM_delta_pp': 100*mean(em[:, large]-em[:, small], mask) if mask.any() else None,
            'ACC_delta_pp': 100*mean(acc[:, large]-acc[:, small], mask) if mask.any() else None,
            'token_delta': mean(cost[:, large]-cost[:, small], mask),
            'search_delta': mean(search[:, large]-search[:, small], mask),
            'large_quality_nondecrease_and_cost_lower_EM': mean((em[:, large] >= em[:, small]) & (cost[:, large] < cost[:, small]), mask)})

edges = np.quantile(rootout, [0, .2, .4, .6, .8, 1])
for k in range(5):
    mask = (rootout >= edges[k]) & (rootout < edges[k+1] if k < 4 else rootout <= edges[k+1])
    result['root_output_groups'].append({'low': float(edges[k]), 'high': float(edges[k+1]), 'n': int(mask.sum()),
        'budget_metrics': [{'budget': b, 'EM': mean(em[:, j], mask), 'ACC': mean(acc[:, j], mask),
                           'tokens': mean(cost[:, j], mask)} for j, b in enumerate(LABELS)]})

# Within observed same-plan comparisons, see whether input substitutes for output.
for a, b in [(0, 5), (4, 6)]:
    same = np.array([q[a] == q[b] for q in qseq])
    for name, base in [('all', np.ones(n, bool)), ('clean', clean)]:
        mask = same & base
        save = mask & (inp[:, b] > inp[:, a]) & (cost[:, b] < cost[:, a])
        result['same_plan_substitution'].append({'pair': f'{LABELS[a]}->{LABELS[b]}', 'cohort': name,
            'same_plan_n': int(mask.sum()), 'input_more_total_less_n': int(save.sum()),
            'fraction': mean(save, mask), 'mean_input_delta_in_saving_group': mean(inp[:, b]-inp[:, a], save),
            'mean_output_delta_in_saving_group': mean(out[:, b]-out[:, a], save),
            'mean_thinking_delta_in_saving_group': mean(thinking[:, b]-thinking[:, a], save),
            'mean_EM_delta_in_saving_group': mean(em[:, b]-em[:, a], save),
            'quality_nondecrease_in_saving_group': mean(em[:, b] >= em[:, a], save)})

# Features known before evidence-exposure choice; root is already paid in FULL.
feature_names = ['question_words', 'query_words', 'question_query_jaccard', 'query_compression',
    'question_numbers', 'query_numbers', 'question_capitalized_words', 'question_quotes',
    'comparison_cues', 'question_and', 'yes_no_form', 'temporal_cues',
    'root_full_tokens_log', 'root_output_tokens_log', 'root_input_tokens_log', 'root_output_per_query_word_log']
features, texts = [], []
for i, sid in enumerate(ids):
    question = geo[sid]['question']
    query = records[i][0]['per_search_round_evidence'][0]['query']
    qw, rw = words(question), words(query)
    qs, rs = set(qw), set(rw)
    comp = {'both', 'same', 'older', 'younger', 'earlier', 'later', 'longer', 'shorter', 'more', 'less', 'higher', 'lower'}
    features.append([len(qw), len(rw), len(qs & rs)/max(1, len(qs | rs)), len(rw)/max(1, len(qw)),
        len(re.findall(r'\d+', question)), len(re.findall(r'\d+', query)),
        len(re.findall(r'\b[A-Z][a-z]+\b', question)), question.count('"'),
        sum(w in comp for w in qw), qw.count('and'), int(bool(qw and qw[0] in {'are','is','was','were','do','does','did','have','has'})),
        sum(w in {'year', 'when', 'before', 'after', 'born', 'died', 'date', 'first'} for w in qw),
        np.log1p(records[i][0]['root_full_context_tokens']), np.log1p(rootout[i]), np.log1p(rootin[i]),
        np.log1p(rootout[i]/max(1,len(rw)))])
    texts.append(question+' QUERY '+query)
X = np.asarray(features, float)
split = np.array([int(hashlib.sha256(('deep-recheck-v1:'+sid).encode()).hexdigest()[:8],16) % 100 for sid in ids])
dev, val, test = split < 60, (split >= 60) & (split < 80), split >= 80
result['split'] = {'method': 'fixed hash 60/20/20, retrospective', 'dev': int(dev.sum()), 'validation': int(val.sum()), 'test': int(test.sum()),
                   'id_list_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest()}
print('Split:', result['split'], flush=True)


def policy_metrics(choice, mask):
    ii = np.where(mask)[0]
    jj = choice[mask]
    return {'n': len(ii), 'EM': mean(em[ii,jj]), 'ACC': mean(acc[ii,jj]), 'mean_total_tokens': mean(cost[ii,jj]),
            'mean_input_tokens': mean(inp[ii,jj]), 'mean_output_tokens': mean(out[ii,jj]),
            'EM_gain_pp': 100*mean(em[ii,jj]-em[ii,6]), 'ACC_gain_pp': 100*mean(acc[ii,jj]-acc[ii,6]),
            'mean_token_delta': mean(cost[ii,jj]-cost[ii,6]),
            'token_reduction_rate': 1-mean(cost[ii,jj])/mean(cost[ii,6]),
            'choices': dict(collections.Counter(LABELS[j] for j in jj))}


result['fixed_baselines'] = [{'budget': b, **policy_metrics(np.full(n,j,int),test)} for j,b in enumerate(LABELS)]
target = np.concatenate([em-em[:,6:7], acc-acc[:,6:7], (cost-cost[:,6:7])/10000], axis=1)
predictions = {}
root_columns = [12,13,14,15]
for name, x, model in [
    ('root_only_rf', X[:,root_columns], RandomForestRegressor(n_estimators=160,max_depth=6,min_samples_leaf=80,random_state=SEED,n_jobs=2)),
    ('all_numeric_rf', X, RandomForestRegressor(n_estimators=160,max_depth=7,min_samples_leaf=70,random_state=SEED,n_jobs=2)),
    ('all_numeric_extra', X, ExtraTreesRegressor(n_estimators=160,max_depth=8,min_samples_leaf=65,random_state=SEED,n_jobs=2)),
    ('interpretable_depth2', X, DecisionTreeRegressor(max_depth=2,min_samples_leaf=200,random_state=SEED)),
]:
    model.fit(x[dev],target[dev])
    predictions[name] = model.predict(x)
    if hasattr(model,'feature_importances_'):
        fn = [feature_names[c] for c in root_columns] if name == 'root_only_rf' else feature_names
        result.setdefault('feature_importance', {})[name] = dict(zip(fn, map(float,model.feature_importances_)))
    if name == 'interpretable_depth2':
        result['depth2_tree'] = export_text(model,feature_names=feature_names)
    print('Fitted',name,flush=True)
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=22000,sublinear_tf=True)
text_dev = vec.fit_transform(np.array(texts)[dev])
text_all = vec.transform(texts)
scaler = StandardScaler().fit(X[dev])
xxdev = sparse.hstack([text_dev,sparse.csr_matrix(scaler.transform(X[dev]))],format='csr')
xxall = sparse.hstack([text_all,sparse.csr_matrix(scaler.transform(X))],format='csr')
model = Ridge(alpha=100,solver='lsqr')
model.fit(xxdev,target[dev])
predictions['tfidf_numeric_ridge'] = model.predict(xxall)
print('Fitted tfidf_numeric_ridge',flush=True)

# This literal question rule is evaluated too; never feed Hotpot official labels.
literal_comparison = np.array([bool(re.search(r'\b(both|same|older|younger|earlier|later|longer|shorter)\b',t.split(' QUERY ')[0],re.I)) for t in texts])
literal_rule = np.where(literal_comparison,4,6)
choices_by_name = {'literal_comparison_B500_else_FULL': literal_rule}
result['literal_rule'] = {'validation':policy_metrics(literal_rule,val), 'test':policy_metrics(literal_rule,test)}
configs = []
for name, pred in predictions.items():
    for metric, offset in [('EM',0),('ACC',7)]:
        for penalty in [0,.002,.005,.01,.02,.05,.1]:
            utility = pred[:,offset:offset+7]-penalty*10*pred[:,14:21]
            chosen = utility.argmax(axis=1)
            key = f'{name}/{metric}/lambda={penalty}'
            choices_by_name[key] = chosen
            configs.append({'key':key,'model':name,'metric':metric,'penalty':penalty,'validation':policy_metrics(chosen,val)})
for j,b in enumerate(LABELS):
    key='fixed_'+b
    choices_by_name[key]=np.full(n,j,int)
    configs.append({'key':key,'model':'fixed','metric':'both','penalty':None,'validation':policy_metrics(choices_by_name[key],val)})
configs.append({'key':'literal_comparison_B500_else_FULL','model':'literal','metric':'both','penalty':None,'validation':policy_metrics(literal_rule,val)})
result['validation_grid'] = configs

result['champions'] = []
for metric in ['EM','ACC']:
    # Pick ONE configuration on validation; test is not used to choose it.
    feasible = [c for c in configs if c['validation']['mean_token_delta'] <= 0]
    winner = max(feasible,key=lambda c:(c['validation'][metric],-c['validation']['mean_total_tokens']))
    choice = choices_by_name[winner['key']]
    ii = np.where(test)[0]
    selected_em, selected_acc, selected_cost = em[ii,choice[test]],acc[ii,choice[test]],cost[ii,choice[test]]
    result['champions'].append({'objective':metric,'chosen_on_validation':winner,'test':policy_metrics(choice,test),
        'test_EM_delta_95CI_pp':[100*v for v in paired_ci(selected_em-em[ii,6])],
        'test_ACC_delta_95CI_pp':[100*v for v in paired_ci(selected_acc-acc[ii,6])],
        'test_token_delta_95CI':paired_ci(selected_cost-cost[ii,6]),
        'test_vs_fixed_B500_EM_delta_pp':100*mean(selected_em-em[ii,4]),
        'test_vs_fixed_B500_ACC_delta_pp':100*mean(selected_acc-acc[ii,4]),
        'test_vs_fixed_B500_token_delta':mean(selected_cost-cost[ii,4])})

# Post hoc attribution of gains: descriptive, never deployable type labels.
for row in result['champions']:
    choice=choices_by_name[row['chosen_on_validation']['key']]
    row['test_type_breakdown']=[{'type':t,**policy_metrics(choice,test & (typ==t))} for t in ['bridge','comparison']]

(HERE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
with gzip.open(HERE/'selected_predictions.jsonl.gz','wt',encoding='utf-8') as f:
    for i,sid in enumerate(ids):
        if test[i]:
            f.write(json.dumps({'sample_id':sid,'root_output_tokens':int(rootout[i]),
                'policies':{r['objective']:LABELS[int(choices_by_name[r['chosen_on_validation']['key']][i])] for r in result['champions']}},ensure_ascii=False)+'\n')
print(json.dumps({'champions':result['champions'],'literal_rule':result['literal_rule'],
                  'persistence':result['persistence'],'same_plan_substitution':result['same_plan_substitution']},ensure_ascii=False,indent=2),flush=True)
