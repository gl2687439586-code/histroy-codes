"""Produce a concise Chinese plain-text report and a portable results bundle."""
import hashlib
import json
from pathlib import Path
import zipfile
import analyze as a


def load(name):return json.loads((a.OUT/name).read_text(encoding='utf-8'))


def main():
    fixed=load('FIXED_BUDGETS.json');sets=load('ORACLE_SETS.json');ass=load('ASSOCIATIONS.json')
    policies=load('CROSS_RUN_POLICIES.json');sup=load('EXPLORATORY_SUPPLEMENT.json')
    t=[]
    def line(s=''):t.append(s)
    line('HotpotQA 匹配500题：原始轨迹 finding 与决策前信号审计')
    line('日期：2026-10-04；全部离线计算，新增模型调用 0。')
    line('结论：有值得继续研究的现象，但没有发现已验证、可稳定优于固定500预算的简单先验路由器。')
    line()
    line('一、范围与口径')
    line('两次运行完全相同的500题。主分析限制在双方全部题都有的100/200/300/400/500/600/FULL七种条件，共7000条轨迹。')
    line('本包提取1000个原始root的20项特征；沿用独立raw审计已逐条重算、对齐的答案分数。原包另有根长度特有条件，未混入公平比较。')
    line('budget为每一次Search后暴露的拼接前缀上限，不是全流程总预算；FULL是每次完整Top-5。')
    line('主指标EM：规范化精确匹配；格式失败/本地fallback按0。F1为现有legacy词重叠版本，ACC是词法包含，不是LLM judge。')
    line('平均成本=选中一条轨迹的LLM输入+输出tokens，含首条query/root一次；不含枚举所有备选预算的物理总成本，不等于计费价格/延时。')
    line('20项先验仅用问题、首条query生成、首轮原始Top-5；不输入gold答案、支持事实、后续query、最终答案或质量。')
    line('对所有Top-5做CPU词法检查意味着控制器可见检索结果，但不等于回答模型读完证据；其额外CPU开销尚未计时。')
    line()
    line('二、Finding 1：有时是胜者更替，而不是所有成功预算消失')
    for key,label in [('old_any_correct','旧轮至少一个预算EM正确'),('new_any_correct','新轮至少一个预算EM正确'),
                      ('both_any_correct','两轮都至少有一个正确预算'),('any_shared_correct_budget','至少有同一个预算两轮都正确'),
                      ('both_any_but_no_shared_correct_budget','两轮均有正确预算但正确集合无交集')]:
        line(f'{label}：{sets[key]}/500。')
    line(f'两轮都存在正确预算的231题中，216/231={216/231:.2%}有交集。非空成功集合并集上的平均Jaccard={sets["success_set_jaccard_mean_nonempty_union"]:.4f}。')
    line('旧轮EM优先、实际总tokens最少打破平局的赢家，242个旧正确中190个在新轮仍正确；52个丢失中26个仍有另一个“旧轮也正确”的预算在新轮正确。')
    line('因此不能只用单一argmax标签一致性衡量全部稳定性。但216/500=43.2%的双轮共赢Oracle使用了两轮gold，仅是诊断上界，不能当作部署结果。')
    line('旧Oracle同分规则敏感性（旧轮EM都为48.4%；新轮指标及平均tokens）：')
    for k,v in sets['tie_breaks'].items():
        line(f'  {k:38s} EM={v["new"]["answer_em"]:.4f} total={v["new"]["total_tokens"]:.2f}')
    line('这些规则都是事后诊断，不应根据新轮结果再挑最佳平局规则并声称通过泛化测试。')
    line()
    line('三、Finding 2：少暴露证据不等于同比例降低全流程成本')
    line('run budget EM F1_legacy lexical_ACC mean_input mean_output mean_total mean_evidence mean_searches')
    for run,r in fixed.items():
        for b,v in r['fixed'].items():
            line(f'{run:7s} {b:4s} {v["answer_em"]:.4f} {v["answer_f1"]:.4f} {v["answer_acc"]:.4f} '
                 f'{v["standalone_llm_input_tokens"]:.2f} {v["standalone_llm_output_tokens"]:.2f} {v["total_tokens"]:.2f} '
                 f'{v["visible_evidence_tokens_total"]:.2f} {v["search_calls"]:.3f}')
    line('新轮100 vs FULL：证据均值219.20 vs1205.17，但总成本3821.24 vs4504.18，仅节约15.16%；输出反而2766.70 vs2268.20，搜索2.192 vs1.526。')
    line('新轮有169/500题出现：100条件累计可见证据少于FULL，但总LLM tokens反而更多；291/500的输出tokens更多。旧轮对应161、271题。')
    line('新轮固定500与FULL的EM均38.2%，500节约10.36%总tokens；EM差95%配对bootstrap CI[-2.8,+2.6]百分点，不是证明两者严格等价。')
    line('统计上没有显著差异不等于证明非劣；正式非劣试验需预先设定容许差距并留盲测集。')
    line('可支持的研究表述：显式阅读成本与内生的后续搜索/生成成本耦合，最小化暴露tokens并不等于最小化总成本。')
    line('不能仅由这些轨迹证明给定题更少阅读导致了特定推理机制；后续query/生成及运行差异也是效应的一部分。')
    line()
    line('四、候选先验：检索集中度/首篇匹配度更像成本风险信号，不是已验证的预算答案')
    line('首轮Top-5的不同页面标题数使用NFC+小写精确标题统计；不同页面不必然意味着无关，重复页面也不保证证据完整。')
    line('以下均为FULL条件：')
    for run,g in sup['profiles'].items():
        for name in ['one_or_two_unique_titles','five_unique_titles']:
            x=g[name];v=x['budgets']['FULL']
            line(f'{run} {name}: n={x["n"]}, EM={v["answer_em"]:.4f}, total={v["total_tokens"]:.2f}, '
                 f'continuation={v["continuation_tokens"]:.2f}, searches={v["search_calls"]:.3f}, root_gold_title_recall={v["root_retrieved_title_recall"]:.4f}')
    line('为排除仅由两轮分组题目不同造成的表面复现，另取两轮都属于同一类的交集：')
    for name,x in sup['same_question_profile_replication'].items():
        line(f'{name}, same IDs n={x["n"]}: '+str(x))
    line('gold标题召回只用于事后解释，并未作为可部署特征。上述分组是本次查看20种特征后选出的探索性发现，存在选择偏差及题目难度混杂。')
    line('query-Top1词法覆盖率与FULL后续成本Spearman：旧-0.244，新-0.222；与FULL相对500的EM增益仅旧0.035、新0.010。')
    line('首条query输出长度与FULL后续成本Spearman：旧0.159、新0.082；与FULL正确性为旧-0.041、新-0.005。去掉root截断后结论仍弱。')
    line('必须区分“这一题可能很贵”与“给它哪个预算会更好”；这里前者有候选相关信号，后者尚未验证。')
    line()
    line('五、真正的预算路由测试：跨运行，并按题目隔离训练/测试')
    line('五折：旧轮400题训练 -> 新轮其余100题评估，循环覆盖500题；反向亦测。相同题号不会跨训练/测试折。')
    line('模型为一层阈值树，阈值仅从训练特征的十分位点产生，叶节点至少80题；训练目标EM-lambda*total/1000，lambda=0或0.02。')
    line('三个预先指定划分seed=20261004/05/06；下表是第一个seed全部特征族，没有挑最好的seed。')
    line('source -> target lambda family EM F1 mean_input mean_output mean_total delta_EM_vs500 95%CI')
    for r in policies:
        if r['seed']!=a.SEEDS[0]:continue
        m=r['means'];delta=r['comparisons']['500']['answer_em']
        line(f'{r["source"]}->{r["target"]} {r["lambda"]:.2f} {r["family"]:18s} {m["answer_em"]:.4f} {m["answer_f1"]:.4f} '
             f'{m["standalone_llm_input_tokens"]:.2f} {m["standalone_llm_output_tokens"]:.2f} {m["total_tokens"]:.2f} {delta["delta"]:.4f} {delta["ci95"]}')
    line('全部72组（2方向×3划分×2目标×6族）结果、每折实际阈值与动作都在JSON；未以最终答案或后续轨迹特征作为先验。')
    line('成本惩罚下root_generation在主划分旧->新得到EM37.8%、3857.25 tokens，比固定500便宜4.47%，但EM低0.4百分点；相同预算比例随机分配对照的EM差CI跨0。')
    line('换划分后该模型旧->新EM范围36.6%–37.8%；反向34.2%–36.6%，没有展示稳定降本增质。模型更复杂或新特征是否有效，本测试不能排除。')
    line('CI为5000次题目配对bootstrap、条件于已训练OOF预测；未覆盖训练随机性全部不确定性，未做多重检验校正。')
    line('两轮是时间不同的运行，不是服务器和batch完全控制的独立重复；这500题之前已被反复查看，因此这是探索性检查，不是新盲测。')
    line()
    line('六、下一步研究取舍')
    line('不建议继续把“首条query用了多少tokens -> 单题唯一最佳预算”作为核心方法：直接路由测试缺乏正证据。')
    line('更有数据依据的窄问题：面对首轮检索失配/分散风险，继续扩大当前阅读是否只是放大后续成本？与定向query修正或补全当前证据相比，哪种干预具有可重复的净收益？')
    line('另一条方法论线索：使用多次重复的成功预算集合/期望质量与成本作为训练标签，而非一次轨迹上按实际成本打破平局的argmax。')
    line('这两条都是待验证假设，不是本次已经实现的算法效果。新实验优先少量题、多次受控重复，冻结根和prompt并留独立题目测试，再扩大规模。')
    line('当前轨迹没有在同一状态下对“加读”和“改query”完整分叉，所以不能由本包直接计算两种新动作的因果收益。')
    line()
    line('七、相关工作边界（2026-10-04在线核查官方论文页面；不是系统综述）')
    line('Adaptive-RAG：按问题复杂度路由检索策略。https://arxiv.org/abs/2403.14403')
    line('DRAGIN：按生成时信息需求动态决定何时/检索什么。https://aclanthology.org/2024.acl-long.702/')
    line('TALE：依据复杂度调整推理token预算。https://arxiv.org/abs/2412.18547')
    line('Know Before You Fetch：校准置信度进行检索预算分配，并计入延时/token权衡；预印本。https://arxiv.org/abs/2606.29959')
    line('When Should Active RAG Retrieve：效用、阈值迁移、伤害率与成本分解；KDD2026 workshop，不是KDD主会。https://arxiv.org/abs/2607.24010')
    line('DF-RAG：按query动态选择检索多样性，Findings of EACL2026。https://arxiv.org/abs/2601.17212')
    line('故“动态预算/不确定性/检索多样性/成本分解”这些宽泛词本身不足以构成新颖性，不能承诺CCF-A或未被做过。')
    line()
    line('八、文件与复现')
    line('analyze.py为主分析；supplement.py为发现后的探索性分组；test_analysis.py包含4项测试；PLAN.txt记录计算前评估设计。')
    line('results/含原始先验特征、逐题成功预算集合、全部策略选择、置信区间和跨seed敏感性。源原始压缩包不在此分析ZIP中重复打包。')
    line('原始包SHA256：b76014d79f8282204ef8ac35acc22ef70d15ed30be1f963f3143c577f0133f25。')
    line('复现依赖同工作区analysis_matched500_raw_20261004/input、support/tokenizer.json及vendor；numpy与tokenizers。')
    line('执行：python analyze.py，然后 python supplement.py、python test_analysis.py、python make_report.py。')
    (a.BASE/'FINDINGS_AND_PRIORS_ZH.txt').write_text('\n'.join(t)+'\n',encoding='utf-8')
    dest=a.WORK/'deliverables/HotpotQA_matched500_findings_priors_20261004.zip'
    dest.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.BASE.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:
                z.write(f, str(f.relative_to(a.BASE)))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
    print(dest)
    print('bytes=',dest.stat().st_size,'sha256=',hashlib.sha256(dest.read_bytes()).hexdigest())


if __name__=='__main__':main()
