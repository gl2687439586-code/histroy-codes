"""Render the verified numerical findings as a Chinese plain-text report and compact review bundle."""
from pathlib import Path
import json,zipfile,hashlib
import numpy as np
import analyze as a

def read(n):return json.loads((a.OUT/n).read_text(encoding='utf-8'))
def pct(x):return f'{100*x:.2f}%'
def ci(x,percent=True):
    scale=100 if percent else 1
    return f"{x['mean']*scale:.2f} [{x['ci95_question_cluster'][0]*scale:.2f}, {x['ci95_question_cluster'][1]*scale:.2f}]"
def table(headers,rs):
    return '\n'.join([' | '.join(headers),' | '.join(['---']*len(headers))]+[' | '.join(map(str,r)) for r in rs])

def main():
    pr=read('per_run_metrics.json');sets=read('success_sets.json');tr=read('temporal_transfer.json')
    nt=read('nontrivial_budget_structure.json');st=read('set_transfer.json');pred=read('question_disjoint_prediction.json')
    raw=read('raw_protocol_summary.json');bounds=read('hindsight_bounds.json');curve=read('learning_curve.json')
    sens=read('transfer_sensitivity.json');meta=read('analysis_metadata.json');integ=read('archive_integrity.json')
    emtr={x['policy']:x for x in tr if x['objective']=='EM'};policy=emtr['old2_mean_per_question']
    lines=[
      'HotpotQA 匹配500题 × 五次重新执行：预算空间与成功预算集合诊断',
      '生成日期：2026-10-05；离线审计，不调用生成模型，不启动服务器，不改变原始实验。',
      '',
      '一、结论分级',
      '1. 预算优化空间跨运行存在：支持。五轮同一七档网格的 EM Oracle 为47.6%—50.0%，各轮均高于该轮最佳固定预算9.4—11.6个百分点。',
      '2. 每题存在部分可迁移的成功预算核心：支持，但并不是完整集合恒定不变。历史重复成功显著提升后续成功比例；绝对不能将它解释为所有题都可稳健选择。',
      '3. 使用重复结果的集合/频率目标比照搬一次赢家更合理：有实证支持。在同题有历史标签的特权测试中，旧两轮共同选出的单预算可在新三轮实现平均EM40.4%、总tokens3322.2。',
      '4. 首轮廉价特征已经能预测该集合并实现可部署增益：不支持。题目隔离、仅用旧两轮训练的20特征预测器，三种划分EM35.73%—37.20%，未稳定超过固定预算。',
      '准确表述：目标结构存在、重复标签有价值，但有效的无标签决策信号尚未找到。不是“无空间”，也不是“已有可部署算法”。',
      '',
      '二、可比性与统计口径',
      'R1=原500题；R2=全量7405题中对应的500题；R3/R4/R5=本次repeat_1/2/3。',
      f"原始轨迹共{meta['records_all_native']}条；主分析为500×5×7={meta['records_common_primary']}条。",
      '共同预算：100、200、300、400、500、600、FULL。更高数字预算的可用性随根检索长度变化，故主分析不把缺少档位当失败；原生完整网格单列敏感性结果。',
      '预算B表示每次搜索后顺序暴露的证据上限，不是整题推理token预算；FULL表示该轮Top-5全部结果。搜索自然进行，最多4次，首条Search计入。',
      '新三轮temperature=0，每轮独立重新执行完整流水线、重新生成root；不是固定root的随机采样，也不能假设统计独立。',
      'R2与新三轮主要配置一致：Qwen3-8B、Top-5、最大生成8192、context40960、client limit24497、双卡GPU2/3、每卡2worker。模型config、检索config及核心动作/提示词/检索/轨迹代码哈希一致；全量运行wrapper哈希不同。没有归档权重逐文件哈希及服务端完整数值执行轨迹，不能宣称bitwise环境一致。',
      'R1使用旧版Unicode证据前缀边界处理；481轮证据存在原始字符差异，NFC正规化后前缀一致。排除R1的敏感性结论同向。',
      'EM=严格归一化完全匹配；ACC=原工程normalized contains词法包含指标，不是LLM judge；F1沿用工程评分并逐条重算。主集合按EM=1定义，ACC另做同目标敏感性。',
      '每题总成本=该轨迹LLM输入tokens+输出tokens，共享root仅计一次。不把证据预算数字误当总成本。多候选集合计全部候选continuation成本+一次root；空集合也计当前root成本。',
      'bootstrap=10,000次按题目配对/聚类抽样；同一题的所有预算和运行保留在同一簇。后三轮先在题内平均，再对500题计算区间，不把1500条当独立题。区间描述这批运行的题目抽样不确定性，不是无限未来运行保证；探索性多指标未做家族错误率校正。',
      '',
      '三、原始审计与协议风险',
      table(['运行','原生轨迹','fallback(严格计0)','续写有length','涉及root length的轨迹','历史压缩','未闭合think内answer'],
            [[x['run'],x['native_records'],x['fallback'],x['any_length'],x['root_length'],x['compacted'],x['unclosed_think_answer']] for x in raw]),
      '注意root length列是受影响轨迹数，不是不同root数。',
      '500个ID/题目/gold对齐；所有raw prompt按模板重建通过；使用独立参考Qwen tokenizer校验输入输出和证据前缀，计数不一致为0；usage逐事件累加和scored字段一致；EM/F1/ACC逐条重算一致。参考tokenizer来自本地先前审计参考文件，不是服务器导出的tokenizer，不能把这个校验冒充服务器模型文件全同证明。',
      '新三轮6条答案标签出现在未闭合think内，原协议将其接受：这是协议风险，已单独输出案例。保守把这些记录全部计0，旧两轮选择策略的后三轮EM从40.40%变40.33%；主要结论不由这6条驱动。Search标签的同类情况另有记录，但并不自动等于最终答错。',
      '',
      '四、预算空间：同网格、同目标 Oracle',
      table(['运行','FULL EM','FULL F1','FULL ACC','FULL平均总tokens','该轮最佳固定EM','EM Oracle','EM Oracle选中轨迹平均总tokens'],
            [[x['run'],pct(x['fixed']['FULL']['EM']),f"{x['fixed']['FULL']['F1']:.4f}",pct(x['fixed']['FULL']['ACC']),f"{x['fixed']['FULL']['total_tokens']:.1f}",pct(max(v['EM'] for v in x['fixed'].values())),pct(x['oracles']['EM']['EM']),f"{x['oracles']['EM']['total_tokens']:.1f}"] for x in pr]),
      'Oracle先按指定质量指标最大化，并列时选该次运行实际总tokens最低者。因此其成本也是事后知道的下界，不是部署成本预测，也不包含枚举全部候选的计算费用。',
      table(['运行','EM目标 Oracle EM','F1目标 Oracle F1','ACC目标 Oracle ACC'],
            [[x['run'],pct(x['oracles']['EM']['EM']),f"{x['oracles']['F1']['F1']:.4f}",pct(x['oracles']['ACC']['ACC'])] for x in pr]),
      '这三列分别优化不同目标，不能合并为同一策略的三项指标。各策略完整交叉指标/输入输出成本均在per_run_metrics.json。',
      '',
      '五、成功集合的稳定性：稳定核心，不是原封不动',
      '定义S_i^(r)={B:EM_i^(r)(B)=1}。',
      f"五轮中至少一次有成功预算：{sets['EM']['ever_any_success']}/500；五轮每轮都有某个成功预算：{sets['EM']['any_success_in_every_run']}/500。",
      f"存在同一个预算在五轮都成功：{sets['EM']['same_budget_success_in_every_run']}/500，占前述每轮可解203题的{181/203:.1%}。",
      '其中66题全部35个题目×运行×预算结果都正确，不需要预算个性化；223题35个全部错误；另211题在运行/预算间有变化。181题稳定核心中有115题并非所有预算始终正确；35题同时存在“五次都对”的预算和“五次都错”的另一个预算。',
      '两两比较、两轮均有成功预算时，成功集合Jaccard为0.754—0.786。单赢家(质量后成本tie-break)一致率为45.2%—54.5%。两者指标定义不同，不能直接把77%和50%相减当“稳定性提升”。',
      '严格要求完整集合完全相同，在至少一轮集合非空的题中仅39.2%—43.5%。因此“整套集合固定不变”的强假设不成立，证据支持的是部分重叠与可迁移成功频率。',
      '',
      '六、关键时间顺序留出回放检验：只用旧两轮信息选择，新三轮评估',
      '对每题每预算计算旧两轮平均EM；选均值最高的预算；并列仅使用旧两轮平均实际总tokens，再按预算顺序。没有查看新三轮答案、质量或成本决定选择。旧两轮均失败的题仍保留并按规则选一个预算，不删除困难题。',
      table(['策略（后三轮平均）','EM','ACC','F1','输入tokens','输出tokens','总tokens'],
          [[name,pct(emtr[key]['aggregate']['EM']),pct(emtr[key]['aggregate']['ACC']),f"{emtr[key]['aggregate']['F1']:.4f}",f"{emtr[key]['aggregate']['input_tokens']:.1f}",f"{emtr[key]['aggregate']['output_tokens']:.1f}",f"{emtr[key]['aggregate']['total_tokens']:.1f}"] for key,name in [('FULL','FULL'),('fixed500','固定500（旧轮全局选择）'),('fixed600','固定600（新三轮EM最强固定档）'),('old1_winner','沿用旧第1轮赢家'),('old2_winner','沿用旧第2轮赢家'),('old2_mean_per_question','旧两轮成功频率→单预算')]]),
      f"相对FULL：EM差异 {ci(policy['vs_FULL']['EM'])} 个百分点；F1差异 {ci(policy['vs_FULL']['F1'],False)}；ACC差异 {ci(policy['vs_FULL']['ACC'])} 个百分点。",
      f"总tokens差异 {ci(policy['vs_FULL']['total_tokens'],False)}；平均节省{1-policy['aggregate']['total_tokens']/emtr['FULL']['aggregate']['total_tokens']:.2%}。",
      f"相对旧轮选择的全局固定500：EM差异 {ci(policy['vs_fixed500']['EM'])} 个百分点；平均节省{1-policy['aggregate']['total_tokens']/emtr['fixed500']['aggregate']['total_tokens']:.2%}。",
      f"相对新三轮最强固定600：EM差异 {ci(sens[0]['vs_fixed600']['EM'])} 个百分点。",
      table(['新测试轮','策略EM','FULL救回','FULL损害','McNemar精确p','策略总tokens'],[[v['run'],pct(v['EM']),v['vs_FULL_binary']['gain'],v['vs_FULL_binary']['harm'],f"{v['vs_FULL_binary']['mcnemar_exact_p']:.6f}",f"{v['total_tokens']:.1f}"] for v in policy['per_test_run']]),
      '三个新轮方向均正，但第三轮单独McNemar不显著。不要把合并显著说成每次运行都显著。',
      '这是同题历史标签可用的特权策略，不是对新题零标签泛化；历史预算枚举/标注费用未计入上述在线选中轨迹成本。可以作为目标可迁移性的证据，不能冒充可部署控制器。',
      '实现方式是从已归档新三轮读取所选预算对应轨迹，并非另外启动了一次在线部署试验。规则不使用测试轮结果选预算；本次分析并非实验前正式预注册。这里比较平均实现成本，不证明每道题都满足同一个硬性总token上限。',
      '只用历史成本不看质量：EM37.13%、总tokens3199.2；质量后成本策略较其EM+3.27个百分点[1.67,4.93]，说明不是纯粹选择便宜轨迹造成全部改善。并列只取最小名义预算时EM39.40%，说明tie-break确实影响效果，需如实固定和披露。',
      '',
      '七、成功频率的可迁移性与难度混淆',
      table(['旧两轮成功次数','题目×预算单元数','这些单元在新三轮的EM成功率'],[[x['old_success_count'],x['question_budget_cells'],pct(x['new3_cell_success_rate'])] for x in nt['historical_success_frequency_calibration']]),
      '90.73%是旧两轮都成功的1097个题目×预算单元的条件成功率，不是全部500题准确率。大量容易题有多个成功预算，不能仅凭这张表宣称强预算特异信号。',
      '更严格的题内对照：79题在旧两轮中同时有“连续两次都成功的预算”和“连续两次都失败的预算”。用旧数据定义这两类预算，在新三轮前者平均成功75.46%，后者25.26%；题内差值50.20个百分点，95%CI[41.90,58.76]。这支持确实存在预算身份相关的可迁移结构，不全是题目易难分组。仍是同题特权分析，而非真实无标签特征。',
      '',
      '八、候选集合上界与多候选成本',
      table(['旧数据选择规则','新三轮至少一个正确(Oracle-hit)','整组平均总tokens'],
            [[name,pct(st['EM'][key]['oracle_hit']['mean']),f"{st['EM'][key]['mean_bundle_tokens']['mean']:.1f}"] for key,name in [('old_ranked_top1','逐题Top-1'),('global_fixed_top1','全局固定Top-1'),('old_ranked_top2','逐题Top-2'),('global_fixed_top2','全局固定Top-2'),('old_ranked_top3','逐题Top-3'),('global_fixed_top3','全局固定Top-3')]]),
      'Top-2/3的命中需要知道哪一个答案正确，是候选集合内的Oracle上界，不是可直接输出的最终EM。没有验证器/可靠选择规则，不得把44.27%记成实际系统准确率。两候选5846.7 tokens也高于FULL4658.3，不能把“集合预测”天然等同降本。',
      '完整旧交集216题非空，其中187题新三轮每次至少有一个交集成员成功，181题有同一成员在新三轮全部成功。对交集为空的其余284题，需要真实策略的回退规则，不能只报告非空子集。',
      '',
      '九、是否已能用无标签先验预测集合？补做题目隔离测试',
      '复用此前锁定的20个首轮特征及L2=.03逻辑回归，不调参：题目长度/比较词、root输出与thinking tokens、query特征、标题分散度、词法覆盖/相似度等。不使用标准答案、supporting facts、后续质量、后续成本。',
      '5折按sample_id划分，每折旧两轮的400题训练，测试另100题的新三轮；同一测试题的全部旧轮标签均排除。3个固定划分seed。',
      table(['方法/主seed20261004','EM','ACC','F1','平均总tokens','题内预算AUC'],
          [[x['method'],pct(x['top1']['EM']),pct(x['top1']['ACC']),f"{x['top1']['F1']:.4f}",f"{x['top1']['total_tokens']:.1f}",f"{x['within_question_auc']:.4f}"] for x in pred['results'] if x['seed']==20261004]),
      'constant为每折训练数据选择的全局预算；shared_difficulty只改变题目难度估计，不改变预算排序，故其选择结果与constant一致。首轮标题分散分组没有改善。',
      '20特征预测器三个seed的EM分别35.73%、37.20%、36.13%；对应固定选择36.93%、37.00%、36.87%。主seed Top-2候选hit41.40%，训练选出的固定候选对42.00%，差值-0.60个百分点CI[-2.20,1.07]。未找到稳健优势，不以最好的一个seed汇报。',
      '因此：集合/成功频率目标可作为候选研究目标，但现有先验特征+简单预测器尚未实现它。这个负结果不能推出所有模型/特征都不可能，也不能用特权40.4%掩盖它。',
      '',
      '十、重复次数、留一运行与经验上界',
      table(['用于同题选择的历史运行数','留出运行EM','平均总tokens'],[[x['n_training_runs'],pct(x['heldout_EM']['mean']),f"{x['total_tokens']['mean']:.1f}"] for x in curve['EM']]),
      '上表平均全部可能训练/测试运行组合，始终不使用目标轮标签；仍有同一道题的其它轮标签，不等价于未见题泛化。',
      '五轮各自事后Oracle平均EM48.68%；用五轮同时选择“每题一个固定预算”并回评同五轮为46.16%，差2.52个百分点。后者也是训练集回评上界，不能替代留出结果41.60%。',
      '排除有Unicode边界差异的R1后，R2—R5各轮Oracle平均48.75%，每题固定预算回评46.85%；仅新三轮分别48.40%、46.93%。预算机会和部分稳定结构不依赖旧边界实现。',
      '',
      '十一、对假设的最终裁决与论文可用表述',
      '建议保留：不同证据暴露预算存在可复现的质量—成本差异；单次事后最优预算会漂移，但多次执行中反复成功的预算包含可迁移信息。',
      '建议收窄：不要学习一个唯一、无噪声的oracle标签；可将每题每预算的经验成功频率作为软目标，并单独衡量稳定成功核心及选中轨迹实际成本。五次只是粗糙经验频率，尤其本实验temperature=0，不可声称估计了独立同分布采样下的精确成功概率。',
      '尚缺的关键证据：不使用同题历史gold结果、在未见题上，只凭可获得的状态信号选择预算也能稳定增益。本次简单预测试验没有达成这一点。',
      '研究瓶颈已从“是否完全没有机会”转向“能否找到可部署的预算区分信号”。不建议继续把单个离线Oracle数字当成在线收益承诺。',
      '',
      '附录A：五轮全部固定预算结果',
      table(['运行','预算','EM','ACC','F1','输入tokens','输出tokens','总tokens','累计可见证据tokens','搜索次数'],
          [[x['run'],b,pct(v['EM']),pct(v['ACC']),f"{v['F1']:.4f}",f"{v['input_tokens']:.1f}",f"{v['output_tokens']:.1f}",f"{v['total_tokens']:.1f}",f"{v['evidence_tokens']:.1f}",f"{v['search_calls']:.3f}"] for x in pr for b,v in x['fixed'].items()]),
      '',
      '附录B：复核与来源',
      f"新上传包：{integ['archive']}",f"本次计算SHA256：{integ['sha256']}（未提供新包外部对照SHA文件，仅作本地指纹）",
      'inspect_and_audit.py：安全解包与五轮原始prompt/evidence/usage/metric重算；analyze.py：主统计；supplement.py：协议/并列/配置敏感性；predict_sets.py：题目隔离预测；test_results.py：四组独立断言测试通过。',
      '所有统计在results/*.json；matched_arrays.npz保存五轮×500题×7预算质量/成本数组；per_question_sets.json可逐题查看五次集合和根query。',
      '代码依赖本工作区先前audit_raw.py与prior-success-set方法；压缩包用于审阅，不包含全部原始大轨迹或vendor tokenizer，并非服务器一键运行包。',
      '复核顺序：python -X utf8 inspect_and_audit.py；analyze.py；predict_sets.py；supplement.py；test_results.py；make_report.py。需原始输入包和先前工作区依赖；本次没有新增模型调用。',
    ]
    report=a.BASE/'REPORT_ZH.txt';report.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    bundle=a.BASE.parent/'hotpot_matched500_five_runs_diagnosis_20261005.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(a.BASE.glob('*.py'))+sorted(a.BASE.glob('*.txt')):
            z.write(p,p.relative_to(a.BASE))
        for p in sorted(a.OUT.iterdir()):
            if p.name.endswith('_compact.jsonl.gz'):continue
            if p.is_file():z.write(p,p.relative_to(a.BASE))
    print(report);print(bundle);print('Bundle bytes',bundle.stat().st_size)

if __name__=='__main__':main()
