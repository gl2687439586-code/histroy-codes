"""Single-feature follow-up requested by the user: title dispersion routing.

Primary routing definition, fixed before running this script:
  D = number of normalized unique titles in initial Top-5, groups 1-2/3-4/5.
  In source-run training questions select a budget per group by mean EM,
  breaking ties by mean total LLM tokens. Cost-aware sensitivity lambda=.02.
  Same five question-disjoint folds and three seeds as the previous audit.
Binary D==5 and prior one-split method are sensitivity analyses, not winners
to pick after seeing held-out results. No new generation.
"""
from collections import Counter
import json
import numpy as np
import analyze as a

DEST=a.BASE/'title_dispersion_results'


def save(name,obj):
    DEST.mkdir(exist_ok=True)
    (DEST/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def groups(x,method):
    v=x[:,a.FEATURE_NAMES.index('unique_title_fraction')]
    if method=='three_groups':return np.where(v<=.4,0,np.where(v<1,1,2))
    if method=='five_vs_rest':return (v==1).astype(int)
    if method=='global':return np.zeros(len(x),int)
    raise ValueError(method)


def fit(source,train,method,lam):
    em=source['arrays']['answer_em'][train];cost=source['arrays']['total_tokens'][train]
    if method=='stump_min80':
        return a.fit_stump(source['x'][train],em,cost,[a.FEATURE_NAMES.index('unique_title_fraction')],lam)
    labels=groups(source['x'][train],method)
    model={'method':method,'groups':{}}
    u=em-lam*cost/1000
    for g in np.unique(labels):
        mask=labels==g
        action=int(a.choose_action(u[mask],cost[mask]))
        model['groups'][int(g)]={'budget':a.BUDGETS[action],'action':action,'train_n':int(mask.sum()),
            'mean_em_by_budget':em[mask].mean(0).tolist(), 'mean_cost_by_budget':cost[mask].mean(0).tolist()}
    return model


def predict(model,x,method):
    if method=='stump_min80':return a.predict(model,x)
    return np.array([model['groups'][int(g)]['action'] for g in groups(x,method)],int)


def main():
    ids,old=a.load_run('old500');_,new=a.load_run('new500',ids)
    j=a.FEATURE_NAMES.index('unique_title_fraction')
    for run in [old,new]:
        assert all(len(r['documents'])==5 for r in run['roots'].values())
        assert set(np.round(run['x'][:,j]*5).astype(int)).issubset({1,2,3,4,5})
    policies=[];choices=[]
    for source,target in [(old,new),(new,old)]:
        ar=target['arrays'];n=len(ids)
        for seed in a.SEEDS:
            folds=np.array_split(np.random.default_rng(seed).permutation(n),5)
            for lam in [0.,.02]:
                for method in ['global','three_groups','five_vs_rest','stump_min80']:
                    acts=np.full(n,-1,int);models=[];random={k:np.zeros(n) for k in ar}
                    for fold,test in enumerate(folds):
                        train=np.setdiff1d(np.arange(n),test)
                        assert not set(np.array(ids)[train])&set(np.array(ids)[test])
                        model=fit(source,train,method,lam)
                        acts[test]=predict(model,target['x'][test],method)
                        probs=np.bincount(acts[test],minlength=7)/len(test)
                        for k in ar:random[k][test]=ar[k][test]@probs
                        models.append({'fold':fold,**model})
                    assert np.all(acts>=0)
                    selected=a.selected(ar,acts)
                    baselines={b:a.selected(ar,np.full(n,a.BUDGETS.index(b))) for b in ['500','FULL']}
                    baselines['same_mix_expected']=random
                    item={'source':source['label'],'target':target['label'],'seed':seed,'lambda':lam,'method':method,
                        'means':a.means(selected),'actions':dict(Counter(a.BUDGETS[k] for k in acts)),
                        'fold_models':models,'comparisons':{b:a.compare(selected,x,seed==a.SEEDS[0]) for b,x in baselines.items()}}
                    for k in ['gain','harm','mcnemar_p']:item['comparisons']['same_mix_expected'].pop(k)
                    policies.append(item)
                    if seed==a.SEEDS[0]:
                        choices.extend({'sample_id':sid,'source':source['label'],'target':target['label'],'lambda':lam,'method':method,
                            'D_target':int(round(target['x'][i,j]*5)),'budget':a.BUDGETS[acts[i]],
                            **{k:float(selected[k][i]) for k in ['answer_em','answer_f1','answer_acc','total_tokens']}}
                            for i,sid in enumerate(ids))
        print('completed',source['label'],'->',target['label'],flush=True)
    # Descriptive groups at FULL, not routed policies.
    profiles=[]
    for run in [old,new]:
        g=groups(run['x'],'three_groups')
        for k in [0,1,2]:
            mask=g==k
            profiles.append({'run':run['label'],'group':['1-2','3-4','5'][k],'n':int(mask.sum()),
                **{f:float(v[mask,6].mean()) for f,v in run['arrays'].items()}})
    save('POLICIES.json',policies);save('CHOICES.json',choices);save('FULL_GROUP_PROFILES.json',profiles)
    fixed={run['label']:{b:a.means(a.selected(run['arrays'],np.full(500,a.BUDGETS.index(b)))) for b in ['500','FULL']} for run in [old,new]}
    save('BASELINES.json',fixed)
    text=['首轮检索标题分散度单信号预算路由（离线回放）',
          '主方法：D=首轮Top-5不同页面标题数；固定三组1-2、3-4、5，仅在训练集为各组选预算。',
          '五折按题隔离，源运行400题训练、目标运行其余100题测试；两方向、三个固定划分seed。',
          '主目标EM，同分取训练组平均总tokens最低；成本敏感性EM-0.02*total/1000。没有按ACC或F1选预算。',
          '全部500题纳入；成本是单条选中轨迹的输入+输出，root一次，非枚举所有预算成本。',
          'ACC=规范化词法包含，不是LLM judge；F1沿用legacy token overlap；格式fallback按0。',
          '这是看到上一轮分组关联后追加的探索性测试，不是未接触过的新盲测；未证明真实在线推理可复现相同效果。',
          '以下为预先指定主seed20261004；附带二分组和阈值树仅作敏感性，不择优充当主结果。','',
          'SOURCE TARGET METHOD LAMBDA EM ACC F1 AVG_INPUT AVG_OUTPUT AVG_TOTAL']
    for r in policies:
        if r['seed']!=a.SEEDS[0]:continue
        m=r['means']
        text.append(f'{r["source"]} {r["target"]} {r["method"]} {r["lambda"]} {m["answer_em"]:.4f} {m["answer_acc"]:.4f} '
                    f'{m["answer_f1"]:.6f} {m["standalone_llm_input_tokens"]:.2f} {m["standalone_llm_output_tokens"]:.2f} {m["total_tokens"]:.2f}')
    text+=['','FULL分组关联（不是路由后的结果）：']
    for r in profiles:
        text.append(f'{r["run"]} D={r["group"]} n={r["n"]} EM={r["answer_em"]:.4f} ACC={r["answer_acc"]:.4f} F1={r["answer_f1"]:.6f} total={r["total_tokens"]:.2f}')
    text+=['','三组主方法全部seed（不要挑最优seed）：']
    for r in policies:
        if r['method']=='three_groups':
            text.append(f'{r["source"]}->{r["target"]} lambda={r["lambda"]} seed={r["seed"]} means={r["means"]}')
    text+=['','主方法分折的实际映射：']
    for r in policies:
        if r['method']=='three_groups' and r['seed']==a.SEEDS[0]:
            text.append(f'{r["source"]}->{r["target"]} lambda={r["lambda"]}')
            for f in r['fold_models']:
                text.append(f'  fold={f["fold"]}: '+str({k:(g['budget'],g['train_n']) for k,g in f['groups'].items()}))
            text.append('comparisons='+str(r['comparisons']))
    text+=['','不确定性：5000次题目配对bootstrap，仅条件于训练好的OOF策略；未计全部训练不确定性，未做多重检验校正。',
           '复现：python title_dispersion_only.py；依赖父分析脚本和已审计raw输入。新增模型调用=0。']
    (DEST/'REPORT_ZH.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    assert len(policies)==48 and all(sum(r['actions'].values())==500 for r in policies)
    for run in [old,new]:
        g=[r for r in profiles if r['run']==run['label']]
        assert sum(r['n'] for r in g)==500
        assert abs(sum(r['n']*r['answer_em'] for r in g)/500-fixed[run['label']]['FULL']['answer_em'])<1e-12
    print('CHECKS PASSED; results in',DEST)


if __name__=='__main__':main()
