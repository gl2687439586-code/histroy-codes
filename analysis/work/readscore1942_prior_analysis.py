"""Nested, question-grouped audit of READ-demand signals and budget policies.

All outcome-aware choices (model, shrinkage, operating threshold) are made
within an outer training fold. Gold metadata is used for evaluation only.
No remote requests are made. This remains exploratory across model families.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import binomtest, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SCORE_ROOT = ROOT / 'analysis_hotpot_read4213_partial_20260930_114059/hotpot_read4213_partial_20260930_114059'
STAGE = ROOT / 'work/hotpot_stage_export_review_20260928T141101Z'
DATASET = ROOT / 'work/hotpot_signal_search_v2/hotpot_dev_distractor_v1.json'
OUT = ROOT / 'work/readscore1942_prior_analysis'
BUDGETS = ['100', '200', '300', '400', '500', '600', 'FULL']
SEED = 20260930


def rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def words(s):
    return re.findall(r'[a-z0-9]+', s.lower())


def structural(q, query, basis):
    qw, rw = words(q), words(query)
    qs, rs = set(qw), set(rw)
    starts = ['who', 'what', 'which', 'where', 'when', 'how', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'has', 'have']
    cues = [
        {'both', 'same', 'either', 'or'},
        {'more', 'less', 'larger', 'smaller', 'higher', 'lower', 'longer', 'shorter'},
        {'older', 'younger', 'earlier', 'later', 'before', 'after'},
        {'first', 'last', 'earliest', 'latest'},
        {'year', 'date', 'born', 'died', 'founded', 'released', 'opened'},
        {'not', 'never', 'neither', 'without', 'except'},
        {'nationality', 'national', 'country', 'countries'},
        {'number', 'population', 'many', 'much', 'percent', 'distance', 'length'},
        {'director', 'directed', 'actor', 'actress', 'film', 'movie'},
        {'and', 'of', 'whose', 'that', 'which'},
    ]
    caps = re.findall(r"\b[A-Z][\w'’-]*(?:\s+(?:[A-Z][\w'’-]*|of|the|and|de|van)){0,5}", q)
    caps = [e for e in caps if e.lower() not in starts]
    hits = sum(set(words(e)) <= rs for e in caps)
    yesno = int(bool(re.match(r'^(is|are|was|were|do|does|did|has|have|had|can|could|would|will)\b', q.lower())))
    numeric = [len(qw), len(q), len(rw), len(query), len(qs & rs)/max(1,len(qs)), len(qs & rs)/max(1,len(qs|rs)), len(rw)/max(1,len(qw)), len(caps), hits, hits/max(1,len(caps)), len(re.findall(r'\d+',q)), len(re.findall(r'\d+',query)), yesno]
    numeric += [int(q.lower().startswith(x+' ')) for x in starts]
    numeric += [sum(w in group for w in qw) for group in cues]
    bw = set(words(basis))
    basis_cues = [
        {'localized','single','explicit','one'}, {'two','both','comparing','comparison'},
        {'bridge','indirect','relational','interdependent'}, {'multiple','several','distributed','sources','passages'},
        {'context','qualifications','restrictions','implicit'}, {'attribute','attributes','property','properties'},
        {'date','dates','chronological','temporal'}, {'number','numbers','numerical','quantities'},
        {'synthesis','synthesizing','integrating','integration','connecting'}, {'name','identity','identifying'},
    ]
    basis_values = [sum(w in group for w in bw) for group in basis_cues]
    if yesno:
        operator = 'yes_no'
    elif qs & (cues[0] | cues[1] | cues[2]):
        operator = 'comparison'
    elif qs & cues[7]:
        operator = 'numerical'
    elif qs & cues[4]:
        operator = 'temporal'
    elif q.lower().startswith('which '):
        operator = 'which'
    elif q.lower().startswith('who '):
        operator = 'who'
    else:
        operator = 'other'
    return numeric, basis_values, operator


def load_data(ratings=None):
    supplied_ratings=ratings is not None
    ratings = list(rows(SCORE_ROOT/'results/read_scores_partial.jsonl')) if ratings is None else ratings
    rating_map = {x['question_id']:x for x in ratings}
    assert len(ratings)==len(rating_map)
    if not supplied_ratings: assert len(ratings)==1942
    questions = {x['_id']:x['question'] for x in json.load(DATASET.open(encoding='utf-8'))}
    traj = {}
    for x in rows(STAGE/'analysis/scored_records.jsonl.gz'):
        if x['sample_id'] in rating_map:
            group = traj.setdefault(x['sample_id'],{})
            if x['condition_label'] in group:
                raise ValueError('Duplicate trajectory key')
            group[x['condition_label']] = x
    geometry = {}
    for x in rows(STAGE/'retrieval/embedding_enrichment/retrieval_scores.jsonl.gz'):
        if x['sample_id'] in rating_map and x['search_round']==1:
            geometry[x['sample_id']] = x['recomputed_similarity_scores']
    ids = sorted(rating_map)
    arrays = {k:[] for k in ['acc','em','f1','tokens','input_tokens','output_tokens','searches','fallback','recovered_acc','visible_tokens']}
    text_q, text_query, basis, types, operators, structs, bstructs, geoms, score, conf, score_tokens, meta = ([] for _ in range(12))
    for sid in ids:
        rating = rating_map[sid]
        assert sid in traj and all(b in traj[sid] for b in BUDGETS)
        assert questions[sid]==rating['query']
        base = traj[sid]['FULL']
        ev = base['per_search_round_evidence']
        query = ev[0]['query']
        assert all(traj[sid][b]['per_search_round_evidence'][0]['query']==query for b in BUDGETS)
        st, bst, op = structural(questions[sid],query,rating['read_basis'])
        s = np.asarray(geometry[sid],float)
        gaps = -np.diff(s)
        p = np.exp(s-s.max()); p/=p.sum()
        gf = [*s, s.mean(),s.std(),s[0]-s[-1],gaps.max(),gaps.mean(),-(p*np.log(p)).sum(),float(base['root_full_context_tokens'])]
        text_q.append(questions[sid]); text_query.append(query); basis.append(rating['read_basis'])
        types.append(base['type']); operators.append(op); structs.append(st); bstructs.append(bst); geoms.append(gf)
        score.append(min(3,rating['read_score'])); conf.append(rating['read_confidence']); score_tokens.append(rating['usage']['total_tokens'])
        meta.append({'sample_id':sid,'question':questions[sid],'first_query':query,'read_score':rating['read_score'],'read_confidence':rating['read_confidence'],'read_basis':rating['read_basis'],'type':base['type'],'operator':op,'score_tokens':rating['usage']['total_tokens'],'score_cost_usd':rating['usage'].get('cost',0),'root_title_recall':base['root_retrieved_title_recall'],'root_fuzzy_recall':base['root_support_sentence_fuzzy_recall']})
        for key in arrays:
            vals=[]
            for b in BUDGETS:
                r=traj[sid][b]
                mapping={'acc':'answer_acc','em':'answer_em','f1':'answer_f1','input_tokens':'standalone_llm_input_tokens','output_tokens':'standalone_llm_output_tokens','searches':'search_calls','fallback':'local_fallback','recovered_acc':'recovered_answer_acc','visible_tokens':'visible_evidence_tokens_total'}
                vals.append(float(r['standalone_llm_input_tokens']+r['standalone_llm_output_tokens']) if key=='tokens' else float(r[mapping[key]]))
            arrays[key].append(vals)
    embedding = np.load(ROOT/'work/deep_prior_signal_search/bert_question_query_embeddings.npz',allow_pickle=True)
    embed_map={sid:i for i,sid in enumerate(embedding['ids'])}
    data = {k:np.asarray(v,float) for k,v in arrays.items()}
    data.update(ids=np.asarray(ids),q=np.asarray(text_q),query=np.asarray(text_query),basis=np.asarray(basis),types=np.asarray(types),operators=np.asarray(operators),struct=np.asarray(structs,float),basis_struct=np.asarray(bstructs,float),geometry=np.asarray(geoms,float),score=np.asarray(score,float),confidence=np.asarray(conf,float),score_tokens=np.asarray(score_tokens,float),meta=meta,bert=embedding['embeddings'][[embed_map[sid] for sid in ids]])
    return data


def bootstrap(d,seed=SEED,reps=10000):
    rng=np.random.default_rng(seed); z=[]
    for start in range(0,reps,250):
        ix=rng.integers(0,len(d),size=(min(250,reps-start),len(d)))
        z.extend(np.asarray(d)[ix].mean(1))
    return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]


def evaluate(data, chosen, index=None, ci=False, charge=False):
    ix=np.arange(len(data['ids'])) if index is None else np.asarray(index)
    ch=np.asarray(chosen)[ix]
    out={'n':len(ix),'budgets':dict(Counter(BUDGETS[j] for j in ch))}
    for key in ['acc','em','f1','tokens','input_tokens','output_tokens','searches','visible_tokens','fallback','recovered_acc']:
        val=data[key][ix,ch]; ref=data[key][ix,6]
        if key=='tokens' and charge: val=val+data['score_tokens'][ix]
        diff=val-ref
        out[key]={'mean':float(val.mean()),'full':float(ref.mean()),'delta':float(diff.mean())}
        if ci and key in ['acc','em','f1','tokens']: out[key]['ci95']=bootstrap(diff,SEED+len(ix))
    a=data['acc'][ix,ch]; ref=data['acc'][ix,6]
    gain=int(((a==1)&(ref==0)).sum()); harm=int(((a==0)&(ref==1)).sum())
    out.update(gain=gain,harm=harm,mcnemar_p=float(binomtest(gain,gain+harm,.5).pvalue) if gain+harm else 1.0,intervention_rate=float((ch!=6).mean()),score_charged=charge)
    return out


def describe(data):
    n=len(data['ids']); all_ix=np.arange(n)
    oracle=np.asarray([max(range(7),key=lambda j:(data['acc'][i,j],data['f1'][i,j],-data['tokens'][i,j])) for i in all_ix])
    f1oracle=np.asarray([max(range(7),key=lambda j:(data['f1'][i,j],-data['tokens'][i,j])) for i in all_ix])
    regime=np.where(data['acc'].max(1)==0,'always_wrong',np.where(data['acc'].min(1)==1,'always_correct','sensitive'))
    out={'n':n,'complete_pairing':True,'question_text_exact_match':True,'root_query_identity':True,'budgets':BUDGETS,'score_counts':dict(Counter(x['read_score'] for x in data['meta'])),'types':dict(Counter(data['types'])),'regimes':dict(Counter(regime)),'mean_score_tokens':float(data['score_tokens'].mean()),'total_score_cost_usd':sum(x['score_cost_usd'] for x in data['meta']),'full':evaluate(data,np.full(n,6)),'acc_first_oracle':evaluate(data,oracle),'f1_first_oracle':evaluate(data,f1oracle),'fixed':{},'read_groups':{},'correlations':{}}
    for j,b in enumerate(BUDGETS): out['fixed'][b]=evaluate(data,np.full(n,j))
    for s in [1,2,3]:
        ix=np.flatnonzero(data['score']==s)
        group={'n':len(ix),'types':dict(Counter(data['types'][ix])),'regimes':dict(Counter(regime[ix])),'curves':{}}
        for j,b in enumerate(BUDGETS): group['curves'][b]=evaluate(data,np.full(n,j),ix,ci=True)
        out['read_groups'][s]=group
    for name,y in [('oracle_budget_index',oracle),('f1_oracle_budget_index',f1oracle),('any_success',data['acc'].max(1)),('full_acc',data['acc'][:,6]),('full_tokens',data['tokens'][:,6])]:
        r=spearmanr(data['score'],y); out['correlations'][name]={'rho':float(r.statistic),'p':float(r.pvalue)}
    return out


def numeric(data,kind):
    score=np.column_stack([data['score'],data['confidence'],data['score']*data['confidence'],(data['score']==1),(data['score']==2),(data['score']>=3)])
    if kind=='score': return score[:,:1]
    if kind=='score_conf': return score
    if kind=='structure': return data['struct']
    if kind=='geometry': return data['geometry']
    if kind=='lowcost': return np.column_stack([data['struct'],data['geometry']])
    base=np.column_stack([data['struct'],data['geometry']])
    if kind=='read_structure': base=data['struct']
    if kind=='read_geometry': base=data['geometry']
    if kind in ['read_all','read_structure','read_geometry','basis_all','bert_read','text_read','full_teacher']:
        v=np.column_stack([base,score,base*data['score'][:,None],base*data['confidence'][:,None]])
        if kind in ['basis_all','full_teacher']: v=np.column_stack([v,data['basis_struct']])
        return v
    if kind in ['text','bert']: return base
    raise ValueError(kind)


def matrices(data,train,test,kind,method):
    scaler=StandardScaler().fit(numeric(data,kind)[train])
    tr=scaler.transform(numeric(data,kind)[train]); te=scaler.transform(numeric(data,kind)[test])
    blocks_tr=[sparse.csr_matrix(tr)]; blocks_te=[sparse.csr_matrix(te)]
    if kind in ['text','text_read','full_teacher']:
        text=np.asarray(['QUESTION '+q+' QUERY '+r for q,r in zip(data['q'],data['query'])])
        vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=16000,sublinear_tf=True)
        blocks_tr.append(vec.fit_transform(text[train])); blocks_te.append(vec.transform(text[test]))
    if kind in ['basis_all','full_teacher']:
        vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=6000,sublinear_tf=True)
        blocks_tr.append(vec.fit_transform(data['basis'][train])); blocks_te.append(vec.transform(data['basis'][test]))
    if kind in ['bert','bert_read']:
        if method=='forest':
            pca=PCA(n_components=32,random_state=SEED).fit(data['bert'][train])
            emtr,emte=pca.transform(data['bert'][train]),pca.transform(data['bert'][test])
        else:
            sc=StandardScaler().fit(data['bert'][train]); emtr,emte=sc.transform(data['bert'][train]),sc.transform(data['bert'][test])
        blocks_tr.append(sparse.csr_matrix(emtr)); blocks_te.append(sparse.csr_matrix(emte))
    xtr=sparse.hstack(blocks_tr,format='csr'); xte=sparse.hstack(blocks_te,format='csr')
    if method=='forest': return xtr.toarray(),xte.toarray()
    return xtr,xte


def target(data):
    return np.concatenate([data['acc'][:,:6]-data['acc'][:,[6]],data['f1'][:,:6]-data['f1'][:,[6]],(data['tokens'][:,:6]-data['tokens'][:,[6]])/5000.],axis=1)


def predict(data,train,test,kind,method,param):
    y=target(data)
    if method=='group':
        # Score>=3 merged a priori because score4 n=4 is too small.
        if kind=='score_group': key=data['score'].astype(int).astype(str)
        elif kind=='score_operator': key=np.char.add(np.char.add(data['score'].astype(int).astype(str),'|'),data['operators'])
        else: raise ValueError(kind)
        global_mean=y[train].mean(0); out=[]
        for k in key[test]:
            members=train[key[train]==k]
            out.append((y[members].sum(0)+param*global_mean)/(len(members)+param))
        return np.asarray(out)
    if method=='knn':
        emb=normalize(data['bert'])
        sc=StandardScaler().fit(numeric(data,kind)[train])
        xtr=np.column_stack([emb[train],.1*sc.transform(numeric(data,kind)[train])]); xte=np.column_stack([emb[test],.1*sc.transform(numeric(data,kind)[test])])
        nn=NearestNeighbors(n_neighbors=min(param,len(train)),metric='euclidean').fit(xtr)
        dist,near=nn.kneighbors(xte)
        weights=1/np.maximum(dist,.1); weights/=weights.sum(1,keepdims=True)
        return (y[train][near]*weights[:,:,None]).sum(1)
    xtr,xte=matrices(data,train,test,kind,method)
    if method=='pairwise':
        outputs=[]
        for j in range(6):
            labels=y[train,j].astype(int)
            if len(set(labels))<2:
                outputs.append(np.full(len(test),labels.mean()))
                continue
            clf=LogisticRegression(C=param,max_iter=1500,solver='lbfgs',class_weight='balanced',random_state=SEED)
            clf.fit(xtr,labels)
            # Undo the training class-balancing prior before using probabilities.
            p=clf.predict_proba(xte)
            prior=np.asarray([(labels==c).mean() for c in clf.classes_])
            p*=prior[None,:]; p/=p.sum(1,keepdims=True)
            outputs.append(p@clf.classes_)
        aux=Ridge(alpha=100,solver='lsqr',tol=1e-4).fit(xtr,y[train,6:]).predict(xte)
        return np.column_stack(outputs+[aux[:,j] for j in range(12)])
    if method=='ridge': model=Ridge(alpha=param,solver='lsqr',tol=1e-4)
    elif method=='forest': model=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=param,max_features=.7,max_depth=10,n_jobs=4,random_state=SEED)
    else: raise ValueError(method)
    model.fit(xtr,y[train]); return model.predict(xte)


OPERATING=[(w,c,t) for w in [0,.25] for c in [0,.025,.075] for t in [0,.01,.025,.05]]


def choose(pred,op):
    w,c,t=op
    score=pred[:,:6]+w*pred[:,6:12]-c*pred[:,12:18]
    j=score.argmax(1); val=score[np.arange(len(j)),j]
    return np.where(val>t,j,6)


def quick(data,idx,ch):
    a=float((data['acc'][idx,ch]-data['acc'][idx,6]).mean())
    f=float((data['f1'][idx,ch]-data['f1'][idx,6]).mean())
    t=float((data['tokens'][idx,ch]-data['tokens'][idx,6]).mean())
    return a,f,t


FAMILIES={
    'read_score_only':('score_group','group',[25,100]),
    'read_confidence':('score_conf','ridge',[50,500]),
    'question_structure':('structure','ridge',[50,500]),
    'retrieval_geometry':('geometry','ridge',[50,500]),
    'structure_geometry':('lowcost','ridge',[50,500]),
    'read_structure':('read_structure','ridge',[50,500]),
    'read_geometry':('read_geometry','ridge',[50,500]),
    'read_structure_geometry':('read_all','ridge',[50,500]),
    'read_operator_groups':('score_operator','group',[25,100]),
    'question_query_text':('text','ridge',[10,100]),
    'read_question_query_text':('text_read','ridge',[10,100]),
    'read_basis_all':('basis_all','ridge',[10,100]),
    'all_teacher_signals':('full_teacher','ridge',[10,100]),
    'bert_structure_geometry':('bert','ridge',[100,1000]),
    'read_bert_structure_geometry':('bert_read','ridge',[100,1000]),
    'read_nonlinear':('read_all','forest',[15,40]),
    'read_bert_nonlinear':('bert_read','forest',[15,40]),
    'read_semantic_neighbor':('bert_read','knn',[50,150]),
    'semantic_neighbor_without_read':('bert','knn',[50,150]),
    'read_gain_harm':('read_all','pairwise',[.03,.3]),
    'read_text_gain_harm':('text_read','pairwise',[.03,.3]),
    'read_basis_gain_harm':('basis_all','pairwise',[.03,.3]),
}


def run_cv(data,families,seed=SEED,folds=5):
    n=len(data['ids']); idx=np.arange(n)
    strat=np.char.add(np.char.add(data['types'],'_'),data['score'].astype(int).astype(str))
    outer=StratifiedKFold(folds,shuffle=True,random_state=seed)
    chosen={k:np.full(n,-1,int) for k in families}
    chosen['nested_family_selector']=np.full(n,-1,int)
    fixed=np.full(n,-1,int); fold_id=np.full(n,-1,int); selections=[]
    for fold,(train,test) in enumerate(outer.split(idx,strat)):
        print(f'outer {fold+1}/{folds} train={len(train)} test={len(test)}',flush=True)
        fold_id[test]=fold
        fixed_j=max(range(7),key=lambda j:(data['acc'][train,j].mean(),data['f1'][train,j].mean(),-data['tokens'][train,j].mean()))
        fixed[test]=fixed_j
        inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+10).split(train,strat[train]))
        family_selections=[]
        for name in families:
            started=time.perf_counter(); kind,method,params=FAMILIES[name]; candidates=[]
            for param in params:
                inner_pred=np.zeros((len(train),18))
                for tr0,va0 in inner:
                    inner_pred[va0]=predict(data,train[tr0],train[va0],kind,method,param)
                for op in OPERATING:
                    ch=choose(inner_pred,op); a,f,t=quick(data,train,ch)
                    # Predeclared nonnegative ACC/F1 with >=100-token saving;
                    # rank feasible rules by ACC + 0.2 F1, then token saving.
                    if a>=0 and f>=0 and t<=-100:
                        candidates.append((a+.2*f,-t,param,op,a,f,t))
            if candidates:
                best=max(candidates,key=lambda x:(x[0],x[1]))
                _,_,param,op,a,f,t=best
                pr=predict(data,train,test,kind,method,param); ch=choose(pr,op)
                selected={'name':name,'param':param,'operating':op,'inner_acc_delta':a,'inner_f1_delta':f,'inner_token_delta':t,'inner_score':a+.2*f,'fallback':False}
            else:
                ch=np.full(len(test),6); selected={'name':name,'inner_score':0,'inner_token_delta':0,'fallback':True}
            chosen[name][test]=ch; family_selections.append(selected)
            print(f'  {name} inner={selected["inner_score"]:.4f} fallback={selected["fallback"]} time={time.perf_counter()-started:.1f}s',flush=True)
        best_family=max(family_selections,key=lambda x:(x['inner_score'],-x['inner_token_delta']))
        chosen['nested_family_selector'][test]=chosen[best_family['name']][test]
        selections.append({'fold':fold,'families':family_selections,'best_family':best_family['name'],'fixed_budget':BUDGETS[fixed_j]})
    chosen['nested_fixed_acc']=fixed
    report={'seed':seed,'n':n,'protocol':'5-fold outer / 3-fold inner; all outcome-aware choices on training only; retrospective exploration','fold_selections':selections,'policies':{}}
    for name,ch in chosen.items():
        report['policies'][name]=evaluate(data,ch,ci=True)
        report['policies'][name]['by_type']={ty:evaluate(data,ch,np.flatnonzero(data['types']==ty),ci=True) for ty in ['bridge','comparison']}
        if name.startswith('read_') or name=='all_teacher_signals': report['policies'][name]['with_online_score_cost']=evaluate(data,ch,ci=True,charge=True)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/f'cv_predictions_{seed}.npz',ids=data['ids'],fold_id=fold_id,**chosen)
    (OUT/f'nested_cv_{seed}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['describe','cv'],default='describe'); ap.add_argument('--families',default='all'); ap.add_argument('--seed',type=int,default=SEED); ap.add_argument('--folds',type=int,default=5); args=ap.parse_args()
    data=load_data(); OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='describe':
        report=describe(data)
        (OUT/'descriptive.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (OUT/'joined_questions.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in data['meta'])+'\n',encoding='utf-8')
        np.savez_compressed(OUT/'outcomes.npz',ids=data['ids'],**{k:data[k] for k in ['acc','em','f1','tokens','score','confidence','score_tokens','types','struct','geometry','basis_struct']})
        print(json.dumps({k:v for k,v in report.items() if k not in ['read_groups','fixed']},ensure_ascii=False,indent=2))
        for s,g in report['read_groups'].items():
            print('score',s,'n',g['n'],'curves',{b:round(v['acc']['mean'],4) for b,v in g['curves'].items()},flush=True)
    else:
        fs=list(FAMILIES) if args.families=='all' else args.families.split(',')
        with threadpool_limits(limits=4): report=run_cv(data,fs,args.seed,args.folds)
        for name,v in report['policies'].items(): print(name, {m:round(v[m]['delta'],4) for m in ['acc','f1','tokens']},flush=True)


if __name__=='__main__': main()
