"""Independent source-table/selection accounting checks and Chinese audit report."""
from pathlib import Path
import gzip,hashlib,json,zipfile
import numpy as np
import evaluate as e

LABELS={
 'question_soft':'问题长度与比较题提示', 'generation_soft':'首条query生成特征',
 'retrieval_soft':'首轮检索词法/分散度', 'all20_soft':'全部20项：成功频率预测',
 'all20_advantage':'全部20项：相对FULL收益预测', 'text_all20_advantage':'问题/query文本+20项：收益预测',
 'all20_cost01':'全部20项：显式成本权衡 λ=.01', 'all20_cost02':'全部20项：显式成本权衡 λ=.02',
 'train_fixed':'训练折选出的固定预算','shared_difficulty':'仅共享难度对照','nested_selector':'训练内部选择策略',
 **{'fixed_'+b:'固定 '+b for b in e.B}}

def table(items):
    out=['方法 | EM | 词法ACC | F1 | 平均input | 平均output | 平均总tokens | 相对FULL节省']
    for x in items:
        out.append(f"{LABELS.get(x.get('method'),x.get('method','历史标签特权策略'))} | {x['EM']:.2%} | {x['ACC']:.2%} | {x['F1']:.4f} | {x['input_tokens']:.1f} | {x['output_tokens']:.1f} | {x['total_tokens']:.1f} | {x['vs_FULL']['token_saving_fraction']:.2%}")
    return '\n'.join(out)

def cirow(x):
    out=[]
    for m in ['EM','ACC','F1','total_tokens']:
        d=x['vs_FULL'][m];scale=100 if m in ['EM','ACC'] else 1
        lo,hi=d['ci95'];out.append(f"{m}: delta={scale*d['delta']:+.4f}, 95%CI=[{scale*lo:.4f},{scale*hi:.4f}]"+(' 个百分点' if scale==100 else ''))
    return '\n'.join(out)

def main():
    data,X,texts=e.load();ids=data['ids'].tolist();Y,C=data['Y'],data['C']
    count=0;sourcehash={}
    for r,path in enumerate(e.PATHS):
        file=path/'analysis/scored_records.jsonl.gz';mp={(x['sample_id'],str(x['condition_group'])):x for x in e.rows(file)}
        sourcehash[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        for i,sid in enumerate(ids):
            for j,b in enumerate(e.B):
                x=mp[sid,b]
                np.testing.assert_allclose(Y[r,i,j],[x['answer_em'],x['answer_f1'],x['answer_acc']],atol=1e-12,rtol=0)
                assert C[r,i,j]==x['standalone_llm_input_tokens']+x['standalone_llm_output_tokens']
                count+=1
    folds=json.loads((e.BASE/'folds.json').read_text(encoding='utf-8'))
    for f in folds:
        assert len(f['training_ids'])==400 and len(f['test_ids'])==100
        assert not set(f['training_ids'])&set(f['test_ids'])
        assert set(f['training_ids'])|set(f['test_ids'])==set(ids)
    # Recompute primary choices from raw-derived features, independent of saved choice arrays.
    fd=folds[0];tr=np.array([ids.index(s) for s in fd['training_ids']]);te=np.array([ids.index(s) for s in fd['test_ids']])
    normal=e.fit_and_predict(tr,te,[2,3,4],X,texts,Y,C,e.METHODS+['shared_difficulty'])
    archived=np.load(e.BASE/'choices_20261009.npz')
    for m,p in normal.items():np.testing.assert_array_equal(p,archived[m][:,te])
    # Sentinel: mutate all target labels/costs and heldout question labels/costs in source runs.
    # Choices must remain identical. Feature data stays fixed.
    YY=Y.copy();CC=C.copy();YY[2:]=np.nan;CC[2:]=np.nan;YY[:2,te]=np.nan;CC[:2,te]=np.nan
    poisoned=e.fit_and_predict(tr,te,[2,3,4],X,texts,YY,CC,e.METHODS+['shared_difficulty'])
    for m in normal:np.testing.assert_array_equal(normal[m],poisoned[m])
    # Verify all saved rows agree with chosen archived quality and exact token accounting.
    chosen_rows=0;seen=set()
    with gzip.open(e.BASE/'selected_records_primary.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            x=json.loads(line);key=(x['method'],x['run'],x['sample_id']);assert key not in seen;seen.add(key)
            i=ids.index(x['sample_id']);r=e.RUNS.index(x['run']);j=e.B.index(x['budget'])
            assert archived[x['method']][r-2,i]==j
            np.testing.assert_allclose([x['EM'],x['F1'],x['ACC']],Y[r,i,j],atol=1e-12,rtol=0)
            assert x['total_tokens']==C[r,i,j]==x['input_tokens']+x['output_tokens']
            chosen_rows+=1
    assert chosen_rows==16500
    loo=json.loads((e.BASE/'secondary_loo_folds.json').read_text(encoding='utf-8'))
    for f in loo:
        assert f['target_run'] not in f['source_runs']
        assert len(f['source_runs'])==4
        assert not set(f['train_ids'])&set(f['test_ids'])
    verification={'passed':True,'source_records_rechecked':count,'selected_rows_rechecked':chosen_rows,
        'outer_primary_folds_checked':len(folds),'secondary_loo_folds_checked':len(loo),
        'prediction_reproduction':True,'heldout_label_and_cost_poison_invariance':True,
        'primary_quality_order':['EM','F1','ACC'],'source_hashes':sourcehash,
        'source_npz_sha256':hashlib.sha256((e.WORK/'analysis_five_runs_20261005/results/matched_arrays.npz').read_bytes()).hexdigest(),
        'new_LLM_calls':0,'source_files_modified':False}
    e.save('verification.json',verification)
    res=json.loads((e.BASE/'results.json').read_text(encoding='utf-8'))
    loo=json.loads((e.BASE/'secondary_loo_results.json').read_text(encoding='utf-8'))
    available=json.loads((e.BASE/'availability.json').read_text(encoding='utf-8'))
    primary=[x for x in res['policies'] if x['seed']==e.SEEDS[0]]
    hist_cost=(C[:2].sum(2)-6*data['RC'][:2]).sum(0).mean()
    lines=[
      'HotpotQA 五次匹配500题：次优预算的先验分配审计',
      '日期：2026-10-09；本次为实际离线拟合与轨迹回放，不是新增在线推理。',
      '', '结论',
      '现有可部署先验尚未稳定实现相对FULL的降本增质。多数策略节省约9%–15%的轨迹tokens，但质量没有一致提高。',
      '检索词法先验在主划分上EM/ACC略高、F1略低；固定B600已经提供相近或更好的均值。',
      '93.5%是共同成功预算的条件存在率，不是预测器的准确率。保留成功倾向值得研究，但尚未识别可泛化的预算匹配信号。',
      '', '一、93.5%的准确含义',
      json.dumps(res['overlap_935'],ensure_ascii=False),
      '在两轮都有正确预算的231题里，216题有非空交集，即216/231=93.51%。全体500题上为216/500=43.2%。',
      '计算这一交集需看到两轮答案标签，是集合存在性；并未指定如何在新题上选中其中一个成功预算。',
      '所以应把“Oracle空间不可复现”改成“单次赢家不稳定，但部分成功集合可迁移”。',
      '', '二、真正进行了什么测试',
      '共同七条件：100、200、300、400、500、600、FULL；B是每次搜索证据暴露上限，不是整题总成本。',
      '主测试：旧两轮作为训练来源，新三轮作为目标；每个外折训练400题，测试100题，测试题的全部旧轮记录也从训练中排除。',
      '训练目标为每题每预算在训练执行中的EM成功频率（0/.5/1），不是单个argmax预算标签。',
      '第一条搜索完成后，使用当前执行的首轮特征，每题只选一个预算；没有从两个候选中借助gold挑正确答案。',
      '主指标EM；同时报告词法ACC和legacy F1。ACC不是LLM Judge，F1不是官方HotpotQA yes/no特判实现。',
      '平均总tokens=所选完整轨迹input+output，含root一次、所有搜索轮次与重复输入历史；不把B标签冒充成本。',
      '额外CPU特征提取和预测开销未转换成LLM tokens。没有新增模型推理/检索。',
      '先前已多次查看这些数据，所以即使有按题隔离和内层选参，也只能称探索性回放，不是全新盲测。',
      '', '三、主划分：旧两轮训练，新三轮的500题测试均值',
      table(res['baselines']+primary),
      '', '四、关键差值及配对区间',
    ]
    for name in ['retrieval_soft','all20_soft','nested_selector']:
        x=next(x for x in primary if x['method']==name)
        lines += [LABELS[name],cirow(x),'各目标执行EM救回/损害：'+json.dumps(x['vs_FULL']['per_execution_rescue_harm'],ensure_ascii=False)]
    lines += ['固定B600',cirow(next(x for x in res['baselines'] if x['method']=='fixed_600')),
      '这些区间是10,000次按题配对bootstrap；同题的三次执行先平均，不将1500执行当作1500独立题。',
      '区间条件于当前OOF选择，没有每次重新拟合模型；多策略探索未做确认性多重检验。没有显著差异不等于证明质量非劣。',
      '', '五、同一批题的三套划分，不能只挑最好的seed',
      '方法 | seed | EM | ACC | F1 | 平均总tokens']
    for x in res['policies']:
        lines.append(f"{LABELS[x['method']]} | {x['seed']} | {x['EM']:.2%} | {x['ACC']:.2%} | {x['F1']:.4f} | {x['total_tokens']:.1f}")
    lines += ['', '检索特征三划分EM均值略高于FULL，但ACC并非都高，F1三套均低于FULL；不能概括为稳定降本增质。',
      '同预算比例随机分配对照保留每折预测出的预算比例，只去掉题目与预算的对应；若无明显差异，说明个性化匹配增益未证实。']
    x=next(x for x in primary if x['method']=='retrieval_soft')
    lines += ['检索策略 vs 同比例随机分配：'+json.dumps(x['vs_random_same_budget_mix'],ensure_ascii=False),
      '', '六、增加训练重复次数：四轮训练 -> 留一轮执行测试（次要分析）',
      '仍按题隔离：目标题其他四轮也不能进入训练。轮流留出五次执行；四次成功频率作监督，超参数不变。',
      '这不是时间顺序上的未来预测，而是回顾性leave-one-execution-out；此设计在看完主结果后增加，并完整报告。',
      table(loo['baselines']+[x for x in loo['policies'] if x['seed']==e.SEEDS[0]]),
      '四轮监督没有逆转核心结论：全部20项在主划分的EM/ACC/F1仍均低于FULL，尽管平均tokens降低。',
      '', '七、所有提到的信号：可用性与是否测试',
      'Query length：已用实际tokenizer计数，区分问题长度与首条query长度。',
      'Generation length：已用root输出/思考tokens；未来最终答案长度不可当免费先验，未偷用。',
      'Generation consensus：保存了重复执行的最终答案，但它们要先生成才能观察；未在单预算免费先验主测试中偷用。',
      'Entropy/交叉熵：五次root文件无logprob/logit/entropy数组，无法从文本重建真实概率。root不确定性词数不是熵。',
      f"另有白盒Gate100，它与这500题仅重合{available['whitebox_overlap_with_matched500']}题；原报告为结果富集诊断集，不能补齐本表或外推本表。",
      'Gradient norm / variance：本批未保存神经梯度，未计算；文本重复率或线性探针系数不能冒充这些信号。',
      '其他已用信号：标题分散度、文档词法重叠、query/问题覆盖、首条生成修订词和重复率、首轮FULL证据长度。',
      '额外文本实验只用问题和首条query词袋（词表在训练折建立），没有利用答案、gold证据或后续搜索。',
      '', '八、为何历史标签特权策略仍能很好，但不能当先验模型成功',
      table([res['privileged_diagnostic']]),
      '这个策略知道同一道测试题旧两次的正确/错误标签，并按成功频率、实际旧成本选预算；新三次不参与选择。',
      '它能验证“部分成功倾向可迁移”，但未解决未见题怎样从可用信号识别倾向的问题。',
      f'若为获得这份逐题历史而从头跑两轮完整七预算网格，已耗平均约{hist_cost:,.1f} tokens/题（每轮root共享一次）。主特权策略成本不含这笔枚举开销。',
      '该数字不是获取任何共识信号的必需下界；它只是当前完整历史网格的实际计算成本。不能把这批历史答案当免费廉价先验。',
      '', '九、边界与建议',
      '本次否定的是这些可用特征与有限候选学习器已能稳定降本增质，不是证明任何先验、深模型或梯度特征都不可能。',
      '固定预算均值略优不等于按题自适应的创新成功；也没有验证在线重新执行会逐条复现存档分支。',
      '若后续新增信号，应先冻结其采集时点、额外成本、主评价指标与强固定基线，用新题验证相对预算的收益，而非仅预测整题难度。',
      '若将现有轨迹用于可靠性校正蒸馏，那是另一条需要训练后验证的路线；本次先验负结果既未证明蒸馏失败，也没有证明蒸馏会成功。',
      '', '十、核验与文件',json.dumps({k:v for k,v in verification.items() if k!='source_hashes'},ensure_ascii=False),
      'results.json：主测试全部策略、指标、区间、救回损害和预算比例。',
      'secondary_loo_results.json：四轮监督测试。selected_records_primary.jsonl.gz：16500条主划分逐策略选择记录。',
      'choices_*.npz / folds.json：选择与按题划分。features.npz / availability.json：20项特征与信号库存。',
      '脚本依赖工作区已有analysis_raw_prior_signals_20261004模块和原始输入；压缩包是审计成果包，不是无需数据的服务器实验启动包。',
    ]
    (e.BASE/'REPORT_ZH.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    paths=[p for p in e.BASE.iterdir() if p.is_file() and p.suffix not in ['.zip'] and p.name!='file_sha256.json']
    e.save('file_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    archive=e.BASE/'hotpot_suboptimal_prior_audit_20261009.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths+[e.BASE/'file_sha256.json']:z.write(p,arcname=p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in json.loads(z.read('file_sha256.json')).items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    print(json.dumps({'passed':True,'source_records':count,'choices':chosen_rows,'zip':str(archive),'bytes':archive.stat().st_size},ensure_ascii=False))

if __name__=='__main__':main()
