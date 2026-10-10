"""Reproducible offline raw-feature and cross-run budget analysis."""
from __future__ import annotations
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata
import numpy as np

BASE = Path(__file__).resolve().parent
WORK = BASE.parent
RAW = WORK / 'analysis_matched500_raw_20261004'
OUT = BASE / 'results'
BUDGETS = ['100', '200', '300', '400', '500', '600', 'FULL']
SEEDS = [20261004, 20261005, 20261006]
BOOTSTRAPS = 5000
sys.path.insert(0, str(RAW / 'vendor'))
from tokenizers import Tokenizer
TOKENIZER = Tokenizer.from_file(str(RAW / 'support/tokenizer.json'))
STOP = set('a an the of in on at for to from and or is are was were be been being by with as it its that this these those who what which when where how does did do has have had named name other first'.split())
FEATURE_NAMES = [
    'question_tokens', 'question_words', 'question_comparison_cue',
    'root_output_tokens', 'root_thinking_tokens', 'query_tokens',
    'query_question_coverage', 'root_uncertainty_per100words',
    'root_revision_per100words', 'root_repeat_4gram_fraction',
    'root_length_or_fallback', 'unique_title_fraction',
    'doc_pair_jaccard', 'query_top1_coverage', 'query_top5_coverage',
    'question_top1_coverage', 'question_top5_coverage',
    'question_coverage_increment', 'query_title_coverage', 'root_full_tokens'
]
FAMILIES = {
    'global': [],
    'question_only': FEATURE_NAMES[:3],
    'root_length': ['root_output_tokens'],
    'root_generation': FEATURE_NAMES[3:11],
    'retrieved_lexical': FEATURE_NAMES[11:],
    'all_prior': FEATURE_NAMES,
}


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def tok(s):
    return len(TOKENIZER.encode(s, add_special_tokens=False).ids)


def words(s):
    return re.findall(r"[\w]+", unicodedata.normalize('NFC', s).lower())


def content(s):
    return set(words(s)) - STOP


def coverage(a, b):
    return len(a & b) / max(1, len(a))


def root_features(r):
    # Gold labels, dataset difficulty/type, support recalls and future events
    # are intentionally inaccessible to this feature function.
    q, query = r['question'], r['query']
    attempts = r['action_attempts']
    text = '\n'.join(a['raw_output'] for a in attempts)
    w = words(text)
    grams = [tuple(w[i:i+4]) for i in range(max(0, len(w)-3))]
    docs = r['documents']
    dw = [content(d['contents']) for d in docs]
    allw = set().union(*dw)
    qc, sc = content(q), content(query)
    qw1, qw5 = coverage(qc, dw[0]), coverage(qc, allw)
    pairs = [len(a & b) / max(1, len(a | b)) for i, a in enumerate(dw) for b in dw[i+1:]]
    titles = [unicodedata.normalize('NFC', d['title']).lower().strip() for d in docs]
    uncertainty = len(re.findall(r'\b(?:maybe|perhaps|possibly|uncertain|unsure|not sure|might|could)\b', text.lower()))
    revision = len(re.findall(r'\b(?:wait|actually|instead|alternatively|however|but)\b', text.lower()))
    return dict(zip(FEATURE_NAMES, [
        tok(q), len(words(q)), int(bool(re.search(r'\b(?:both|same|different|more|less|earlier|later|older|younger|taller|larger|longer|higher)\b', q.lower()))),
        sum(a['output_tokens'] for a in attempts), sum(a['thinking_tokens'] for a in attempts), tok(query),
        coverage(qc, sc), 100*uncertainty/max(1,len(w)), 100*revision/max(1,len(w)),
        1-len(set(grams))/max(1,len(grams)),
        int(r['query_parse_mode'] == 'fallback_question_as_query' or any(a['finish_type']=='length' for a in attempts)),
        len(set(titles))/len(titles), float(np.mean(pairs)), coverage(sc,dw[0]), coverage(sc,allw),
        qw1,qw5,qw5-qw1,coverage(sc,content(' '.join(titles))),r['root_full_context_tokens']
    ]))


def load_run(label, ids=None):
    p = RAW / 'input' / label
    roots = {r['sample_id']: r for r in rows(p / 'roots/roots.merged.jsonl.gz')}
    ids = sorted(roots) if ids is None else ids
    assert set(ids) == set(roots)
    features = [root_features(roots[s]) for s in ids]
    scored = {(r['sample_id'],r['condition_group']):r for r in rows(p/'analysis/scored_records.jsonl.gz') if r['condition_group'] in BUDGETS}
    assert len(scored)==len(ids)*len(BUDGETS)
    fields = ['answer_em','answer_f1','answer_acc','standalone_llm_input_tokens','standalone_llm_output_tokens',
              'continuation_llm_input_tokens','continuation_llm_output_tokens','search_calls','visible_evidence_tokens_total','local_fallback']
    arrays = {k:np.array([[float(scored[s,b][k]) for b in BUDGETS] for s in ids]) for k in fields}
    arrays['total_tokens']=arrays['standalone_llm_input_tokens']+arrays['standalone_llm_output_tokens']
    arrays['continuation_tokens']=arrays['continuation_llm_input_tokens']+arrays['continuation_llm_output_tokens']
    x = np.array([[f[k] for k in FEATURE_NAMES] for f in features])
    assert np.isfinite(x).all()
    # Lightweight terminal/raw checks use previously audited metrics, without
    # retokenizing 15,669 already checked generation calls a second time.
    raw_summary = Counter()
    for tr in rows(p/'trajectories/trajectories.merged.jsonl.gz'):
        b = 'FULL' if tr['exposure_mode']=='full_per_search' else str(tr['root_exposure_tokens'])
        if b not in BUDGETS:
            continue
        raw_summary['trajectories'] += 1
        assert (tr['sample_id'],b) in scored
        raw_summary['fallback'] += int(tr['answer_status']!='valid')
        raw_summary['any_length'] += int(any(a['finish_type']=='length' for e in tr['decision_events'] for a in e['attempts']))
        raw_summary['context_compactions'] += tr['context_compactions']
    return ids, {'label':label,'x':x,'features':features,'roots':roots,'arrays':arrays,'raw':dict(raw_summary)}


def ci(d, seed=20261004):
    d=np.asarray(d,float)
    if d.ndim==1:
        d=d[:,None]
    rng=np.random.default_rng(seed)
    draws=[]
    for start in range(0,BOOTSTRAPS,100):
        ix=rng.integers(0,len(d),(min(100,BOOTSTRAPS-start),len(d)))
        draws.append(d[ix].mean(1))
    return np.quantile(np.vstack(draws),[.025,.975],axis=0).T.tolist()


def mcnemar(gain,harm):
    n=gain+harm
    if not n:
        return 1.
    ll=[math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2) for k in range(min(gain,harm)+1)]
    peak=max(ll)
    return min(1.,2*math.exp(peak)*sum(math.exp(v-peak) for v in ll))


def selected(arrays, actions):
    ix=np.arange(len(actions))
    return {k:a[ix,actions] for k,a in arrays.items()}


def means(sel):
    return {k:float(np.mean(v)) for k,v in sel.items()}


def compare(a,b,interval=True):
    keys=['answer_em','total_tokens','answer_f1','answer_acc','continuation_tokens']
    ds=np.column_stack([a[k]-b[k] for k in keys])
    bounds=ci(ds) if interval else [None]*len(keys)
    out={k:{'delta':float(ds[:,i].mean()),'ci95':bounds[i]} for i,k in enumerate(keys)}
    out['total_token_saving_fraction']=1-float(np.mean(a['total_tokens'])/np.mean(b['total_tokens']))
    gain=int(np.sum((a['answer_em']==1)&(b['answer_em']==0)))
    harm=int(np.sum((a['answer_em']==0)&(b['answer_em']==1)))
    out['gain'],out['harm'],out['mcnemar_p']=gain,harm,mcnemar(gain,harm)
    return out


def rank(x):
    order=np.argsort(x,kind='stable')
    out=np.empty(len(x),float)
    start=0
    while start<len(x):
        end=start+1
        while end<len(x) and x[order[end]]==x[order[start]]:
            end+=1
        out[order[start:end]]=(start+end-1)/2
        start=end
    return out


def corr(a,b):
    if np.std(a)==0 or np.std(b)==0:
        return None
    return float(np.corrcoef(rank(a),rank(b))[0,1])


def choose_action(u,cost):
    quality=u.mean(0)
    eligible=np.flatnonzero(quality>=quality.max()-1e-12)
    return min(eligible,key=lambda i:(cost[:,i].mean(),i))


def fit_stump(x,em,cost,feature_ix,lam,min_leaf=80):
    u=em-lam*cost/1000
    a=int(choose_action(u,cost))
    best={'feature':None,'threshold':None,'left':a,'right':a,'training_value':float(u[:,a].mean())}
    for j in feature_ix:
        for threshold in np.unique(np.quantile(x[:,j],np.arange(.1,1,.1))):
            left=x[:,j]<=threshold
            if left.sum()<min_leaf or (~left).sum()<min_leaf:
                continue
            aa=int(choose_action(u[left],cost[left]));bb=int(choose_action(u[~left],cost[~left]))
            if aa==bb:
                continue
            val=(u[left,aa].sum()+u[~left,bb].sum())/len(x)
            if val>best['training_value']+1e-12:
                best={'feature':FEATURE_NAMES[j],'threshold':float(threshold),'left':aa,'right':bb,'training_value':float(val)}
    return best


def predict(model,x):
    if model['feature'] is None:
        return np.full(len(x),model['left'],int)
    j=FEATURE_NAMES.index(model['feature'])
    return np.where(x[:,j]<=model['threshold'],model['left'],model['right'])


def routing(source,target,ids):
    n=len(ids);results=[];predictions=[]
    ar=target['arrays']
    bases={b:selected(ar,np.full(n,BUDGETS.index(b))) for b in ['500','FULL']}
    for seed in SEEDS:
        order=np.random.default_rng(seed).permutation(n)
        folds=np.array_split(order,5)
        for lam in [0.,.02]:
            trained={}
            for family,names in FAMILIES.items():
                actions=np.full(n,-1,int); models=[]
                random_vectors={k:np.zeros(n) for k in ar}
                for fold,test in enumerate(folds):
                    train=np.setdiff1d(np.arange(n),test)
                    assert not set(np.array(ids)[train]) & set(np.array(ids)[test])
                    model=fit_stump(source['x'][train],source['arrays']['answer_em'][train],source['arrays']['total_tokens'][train],
                                    [FEATURE_NAMES.index(k) for k in names],lam)
                    actions[test]=predict(model,target['x'][test])
                    probs=np.bincount(actions[test],minlength=len(BUDGETS))/len(test)
                    # Expected outcome of random assignment of the same budget
                    # mix within this held-out fold; uses no outcome to assign.
                    for k in ar:
                        random_vectors[k][test]=ar[k][test]@probs
                    models.append({'fold':fold,'train_n':len(train),'test_n':len(test),**model})
                assert np.all(actions>=0)
                sel=selected(ar,actions)
                trained[family]=(sel,actions)
                item={'source':source['label'],'target':target['label'],'seed':seed,'lambda':lam,'family':family,
                      'means':means(sel),'actions':dict(Counter(BUDGETS[i] for i in actions)), 'fold_models':models,
                      'comparisons':{b:compare(sel,bases[b],seed==SEEDS[0]) for b in bases}}
                item['comparisons']['global']=compare(sel,trained['global'][0],seed==SEEDS[0])
                item['comparisons']['same_mix_expected']=compare(sel,random_vectors,seed==SEEDS[0])
                # McNemar is only defined for binary paired outcomes, not an
                # expectation over a fractional randomized policy.
                for k in ['gain','harm','mcnemar_p']:
                    item['comparisons']['same_mix_expected'].pop(k)
                results.append(item)
                if seed==SEEDS[0]:
                    predictions.extend({'sample_id':sid,'source':source['label'],'target':target['label'],
                                         'lambda':lam,'family':family,'budget':BUDGETS[a]} for sid,a in zip(ids,actions))
            print('routing',source['label'],'->',target['label'],seed,lam,flush=True)
    return results,predictions


def oracle_sets(old,new,ids):
    ae,be=old['arrays']['answer_em'],new['arrays']['answer_em']
    common=(ae==1)&(be==1)
    any_a=ae.max(1)==1;any_b=be.max(1)==1
    union=np.sum((ae==1)|(be==1),axis=1)
    intersect=common.sum(1)
    aopts=[np.flatnonzero(a==a.max()) for a in ae]
    result={'old_any_correct':int(any_a.sum()),'new_any_correct':int(any_b.sum()),
            'both_any_correct':int((any_a&any_b).sum()),'neither_any_correct':int((~any_a&~any_b).sum()),
            'any_shared_correct_budget':int((intersect>0).sum()),
            'both_any_but_no_shared_correct_budget':int(((any_a&any_b)&(intersect==0)).sum()),
            'success_set_jaccard_mean_nonempty_union':float(np.mean(intersect[union>0]/union[union>0])),
            'old_correct_budget_count':dict(Counter(map(int,ae.sum(1)))),
            'new_correct_budget_count':dict(Counter(map(int,be.sum(1)))),
            'tie_breaks':{},'success_multiplicity':[]}
    ao=old['arrays'];bo=new['arrays']
    chosen={}
    for name in ['min_realized_cost','smallest_budget','largest_budget','median_budget','full_if_optimal_else_smallest']:
        actions=[]
        for i,opt in enumerate(aopts):
            if name=='min_realized_cost':
                a=min(opt,key=lambda j:(ao['total_tokens'][i,j],j))
            elif name=='smallest_budget':a=opt[0]
            elif name=='largest_budget':a=opt[-1]
            elif name=='median_budget':a=opt[(len(opt)-1)//2]
            else:a=6 if 6 in opt else opt[0]
            actions.append(a)
        actions=np.array(actions);chosen[name]=actions
        oldsel=selected(ao,actions);newsel=selected(bo,actions)
        result['tie_breaks'][name]={'old':means(oldsel),'new':means(newsel),
            'retained_old_correct':int(np.sum((oldsel['answer_em']==1)&(newsel['answer_em']==1))),
            'vs_new_full':compare(newsel,selected(bo,np.full(len(ids),6))),
            'new_selected_budget_counts':dict(Counter(BUDGETS[i] for i in actions))}
    randomsel={k:np.array([v[i,opt].mean() for i,opt in enumerate(aopts)]) for k,v in bo.items()}
    result['tie_breaks']['uniform_old_optimal_set_expected']={'new':means(randomsel)}
    baseline_actions=chosen['min_realized_cost']
    byold=be[np.arange(len(ids)),baseline_actions]
    lost=any_a&(byold==0)
    result['lost_old_min_cost_winner']=int(lost.sum())
    result['lost_but_another_OLD_winner_still_correct']=int((lost&(intersect>0)).sum())
    for n in range(1,8):
        mask=ae.sum(1)==n
        result['success_multiplicity'].append({'old_correct_budgets':n,'n':int(mask.sum()),
            'old_min_cost_transfer_em':float(byold[mask].mean()) if mask.any() else None,
            'new_has_shared_correct_budget_rate':float((intersect[mask]>0).mean()) if mask.any() else None})
    result['diagnostic_two_run_robust_oracle_em']=float((intersect>0).mean())
    result['warning']='Uses both runs gold labels; descriptive upper bound, not executable online.'
    records=[{'sample_id':sid,'old_correct':[BUDGETS[j] for j in np.flatnonzero(ae[i])],
              'new_correct':[BUDGETS[j] for j in np.flatnonzero(be[i])],
              'shared_correct':[BUDGETS[j] for j in np.flatnonzero(common[i])],
              'old_min_cost_winner':BUDGETS[baseline_actions[i]]} for i,sid in enumerate(ids)]
    return result,records


def associations(old,new):
    result={'feature_stability':{},'within_run':{}}
    for j,name in enumerate(FEATURE_NAMES):
        result['feature_stability'][name]=corr(old['x'][:,j],new['x'][:,j])
    for run in [old,new]:
        ar=run['arrays']; em=ar['answer_em'];ct=ar['continuation_tokens']
        targets={'full_em':em[:,6],'mean_budget_em':em.mean(1),
                 'full_continuation_tokens':ct[:,6], 'full_search_calls':ar['search_calls'][:,6],
                 'full_minus_100_em':em[:,6]-em[:,0], 'full_minus_500_em':em[:,6]-em[:,4],
                 'full_minus_100_cost':ct[:,6]-ct[:,0]}
        group={}
        for j,name in enumerate(FEATURE_NAMES):
            vals=run['x'][:,j]
            bins=[]
            # Descriptive quartiles; tied values stay together and may produce
            # unequal bins. No gold-defined bin boundaries.
            q=np.quantile(vals,[.25,.5,.75]);ix=np.searchsorted(q,vals,side='left')
            for k in range(4):
                mask=ix==k
                if mask.any():
                    bins.append({'quartile':k+1,'n':int(mask.sum()),'feature_min':float(vals[mask].min()),
                                 'feature_max':float(vals[mask].max()),**{t:float(v[mask].mean()) for t,v in targets.items()}})
            group[name]={'spearman':{t:corr(vals,v) for t,v in targets.items()},'quartiles':bins}
        result['within_run'][run['label']]=group
    return result


def fixed_and_cost(run):
    a=run['arrays'];fixed={b:means(selected(a,np.full(500,j))) for j,b in enumerate(BUDGETS)}
    contrasts={}
    for b1,b2 in [('100','FULL'),('300','FULL'),('500','FULL'),('100','500')]:
        i,j=BUDGETS.index(b1),BUDGETS.index(b2)
        x,y=selected(a,np.full(500,i)),selected(a,np.full(500,j))
        contrasts[b1+'_vs_'+b2]=compare(x,y)
        contrasts[b1+'_vs_'+b2]['lower_evidence_but_more_total_cost_count']=int(np.sum((x['visible_evidence_tokens_total']<y['visible_evidence_tokens_total'])&(x['total_tokens']>y['total_tokens'])))
        contrasts[b1+'_vs_'+b2]['lower_evidence_but_more_output_count']=int(np.sum((x['visible_evidence_tokens_total']<y['visible_evidence_tokens_total'])&(x['standalone_llm_output_tokens']>y['standalone_llm_output_tokens'])))
    return {'fixed':fixed,'contrasts':contrasts,'raw_checks':run['raw']}


def self_test():
    assert np.allclose(rank(np.array([3,1,1,2])),[3,.5,.5,2])
    assert mcnemar(0,0)==1 and abs(mcnemar(10,0)-.001953125)<1e-12
    x=np.zeros((200,len(FEATURE_NAMES)));x[100:,0]=1
    em=np.zeros((200,7));em[:100,0]=1;em[100:,6]=1;cost=np.ones_like(em)
    m=fit_stump(x,em,cost,[0],0,min_leaf=80)
    assert np.array_equal(predict(m,x),np.r_[np.zeros(100,int),np.full(100,6)])
    assert fit_stump(x,em,cost,[],0)['feature'] is None
    print('SELF TEST PASSED',flush=True)


def main():
    self_test()
    ids,old=load_run('old500');_,new=load_run('new500',ids)
    save('FEATURES.json',{'feature_names':FEATURE_NAMES,'rows':[
        {'run':r['label'],'sample_id':s,**f} for r in [old,new] for s,f in zip(ids,r['features'])]})
    save('ASSOCIATIONS.json',associations(old,new))
    sets,records=oracle_sets(old,new,ids)
    save('ORACLE_SETS.json',sets);save('PER_QUESTION_SUCCESS_SETS.json',records)
    save('FIXED_BUDGETS.json',{r['label']:fixed_and_cost(r) for r in [old,new]})
    print('descriptive analysis complete',flush=True)
    a,p=routing(old,new,ids);b,q=routing(new,old,ids)
    save('CROSS_RUN_POLICIES.json',a+b);save('POLICY_CHOICES.json',p+q)
    manifest={'samples':len(ids),'budgets':BUDGETS,'features':FEATURE_NAMES,'seeds':SEEDS,
              'policies':len(a+b),'gold_feature_leakage':False,'question_disjoint':True,
              'new_model_calls':0,'ci_bootstraps':BOOTSTRAPS,
              'source_archive_sha256':'b76014d79f8282204ef8ac35acc22ef70d15ed30be1f963f3143c577f0133f25',
              'limitations':['exploratory, previously inspected questions','conditional bootstrap excludes training uncertainty',
                             'multiple comparisons unadjusted','same-mix baseline is expected randomized outcome',
                             'no wall-clock accounting for feature extraction','metrics F1/ACC retain legacy definitions',
                             'two chronological runs are confounded by unrecorded server differences; not independent controlled seeds']}
    save('MANIFEST.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
