"""Summarize every set-prediction model without selecting a winning seed."""
import json
from pathlib import Path
import zipfile
import numpy as np
import analyze as a
import success_set_prediction as s


def load(name):return json.loads((s.DEST/name).read_text(encoding='utf-8'))


def main():
    results=load('ALL_MODELS.json');preds=load('PREDICTIONS_PRIMARY_SEED.json')
    text=[]
    def line(t=''):text.append(t)
    line('成功预算集合预测：匹配500题跨运行探索性检验')
    line('2026-10-04；新模型调用=0；代码实际运行，非伪代码。')
    line('结论：现有廉价先验能弱区分“是否有任何成功预算”，但没有稳定优于固定集合地识别成功预算成员。')
    line()
    line('目标与方法')
    line('S_i={100,200,300,400,500,600,FULL中，本轮EM=1的预算}；不是保证未来重复运行成功的集合。')
    line('旧轮400题训练->新轮100题测试，五折覆盖500题；反向亦测，三个固定划分seed。主seed20261004。')
    line('共七类预测器：按预算的全局常数；首轮标题数三组；问题特征；root生成特征；检索词法特征；全部20项；共享题目难度对照。')
    line('逻辑回归预测七个预算的EM概率，另有独立的“至少一档成功”头；L2=.03，无超参数搜索。')
    line('特征只用问题和首轮query/检索内容，不用gold、支持事实、后续轨迹或目标题的其他运行标签。')
    line('共享题目难度对照允许各题整体成功率不同，但所有题的预算偏好顺序一致，用来区分难度预测与预算匹配。')
    line('所有题都已在先前分析中被查看，因此这是探索性而非盲测；轻量模型的失败不能证明所有语义先验都无效。')
    line()
    line('一、预测是否存在成功预算：AUROC，0.5为随机排序参考')
    for r in results:
        if r['seed']==20261004:
            line(f'{r["source"]}->{r["target"]} {r["model"]:20s}: AUC={r["any_success_auc"]:.4f}')
    extra=[]
    for source,target in [('old500','new500'),('new500','old500')]:
        for name in ['all_prior','title_groups']:
            rr=sorted([r for r in preds if r['source']==source and r['model']==name],key=lambda r:r['sample_id'])
            y=np.array([bool(r['true_success_set']) for r in rr],int);p=np.array([r['p_any'] for r in rr])
            rng=np.random.default_rng(20261004);vals=[]
            for _ in range(2000):
                ix=rng.integers(0,len(y),len(y));v=s.auc(y[ix],p[ix])
                if v is not None:vals.append(v)
            folds=np.array_split(np.random.default_rng(20261004).permutation(500),5)
            row={'source':source,'target':target,'model':name,'auc':s.auc(y,p),
                 'auc_ci95':np.quantile(vals,[.025,.975]).tolist(),
                 'within_fold_auc_mean':float(np.mean([s.auc(y[f],p[f]) for f in folds]))}
            extra.append(row);line(str(row))
    line('注意：训练集常数基线在不同折采用不同频率，合并OOF时AUC可低于0.5；不能把相对该基线的约0.15提升当成很强能力。应看绝对AUC约0.59/0.60及相对0.5的区间。')
    line()
    line('二、同一题内，能否把成功预算排在失败预算前？')
    line('仅评估同时存在成功/失败预算的题：新轮148、旧轮132。全部成功或全部失败没有可比较的预算排序。')
    for r in results:
        if r['seed']==20261004:
            line(f'{r["source"]}->{r["target"]} {r["model"]:20s}: within-question AUC={r["within_question_budget_auc"]:.4f}; Brier={r["budget_brier"]:.4f}')
    line('全部先验模型的预算Brier在两个方向均略差于共享难度模型。因此分数改善更多体现题目整体难度，而不是已经学会个性化成功预算。')
    line()
    line('三、统一只预测两个候选预算：命中集合与tokens')
    line('候选命中=两个预算中至少有一个EM正确；不是最终可部署ACC。必须实际生成两条并额外解决答案选择，才可能实现相应收益。')
    line('tokens=两条continuation的输入+输出之和+root一次；不偷算成一条轨迹。以下仅离线估计，无新增生成。')
    for source in ['old500','new500']:
        chosen=[r for r in results if r['source']==source and r['seed']==20261004]
        baseline=chosen[0]['topk']['2']['training_selected_fixed_set_baseline']
        line(f'{source}: 无先验固定二预算集合（仅训练集选） '+str(baseline))
        for r in chosen:
            m=r['topk']['2']
            line(f'{r["model"]:20s}: hit={m["candidate_hit_rate"]:.4f}, precision={m["mean_set_precision"]:.4f}, '
                 f'nonempty-hit={m["candidate_hit_given_any_success"]:.4f}, bundle_total={m["mean_bundle_total_tokens"]:.2f}')
        r=next(r for r in chosen if r['model']=='all_prior')
        line('全特征 vs 固定集合 配对区间：'+str(r['topk']['2']['vs_fixed_set']))
    line('主划分旧->新：全特征与固定集合均42.2%命中（差CI[-2.2,+2.4]个百分点）；新->旧为41.6% vs41.2%（差CI[-1.8,+2.605]个百分点）。')
    line('旧->新全特征集合精度36.4%高于固定集合33.7%，但候选命中无增加；反向精度差未显著。不能只挑这个局部正结果。')
    line()
    line('四、如果只运行概率最高的一档：真实单条轨迹指标')
    line('ACC是词法包含，不是LLM judge；F1沿用legacy词重叠；平均tokens包含root一次。')
    for r in results:
        if r['seed']==20261004:
            m=r['topk']['1']['single_answer_metrics']
            line(f'{r["source"]}->{r["target"]} {r["model"]:20s}: EM={m["answer_em"]:.4f}, ACC={m["answer_acc"]:.4f}, '
                 f'F1={m["answer_f1"]:.4f}, input={m["standalone_llm_input_tokens"]:.2f}, output={m["standalone_llm_output_tokens"]:.2f}, total={m["total_tokens"]:.2f}')
    line()
    line('五、集合完全匹配率的陷阱')
    line('新轮251/500题七档均不正确；旧轮258/500。永远输出空集合，完全匹配率就有50.2%/51.6%，但没有选出任何成功预算。')
    line('本次阈值0.5规则的空集合比例、Jaccard（含空-空=1）、非空真集合Jaccard、微平均精确率和召回率都保存在JSON，不能只报精确集合匹配率。')
    for r in results:
        if r['seed']==20261004 and r['model']=='all_prior':line(r['source']+' '+str(r['threshold_set']))
    line()
    line('六、种子敏感性：不选最佳划分')
    for source in ['old500','new500']:
        for name in ['all_prior','title_groups']:
            for r in results:
                if r['source']==source and r['model']==name:
                    line(f'{source} {name} seed={r["seed"]}: any_auc={r["any_success_auc"]:.4f}, within_auc={r["within_question_budget_auc"]:.4f}, top2_hit={r["topk"]["2"]["candidate_hit_rate"]:.4f}')
    line()
    line('七、可支持/不可支持的判断')
    line('可支持：成功集合比单次cost-tiebroken赢家更丰富；当前20项廉价特征提供弱的整体可答性信息，但具体预算成员预测没有稳定的增益。')
    line('不可支持：已找到优质集合预测信号；42.2%就是上线ACC；用两个预算与只跑FULL相比不增加成本；换成集合标签自动解决Oracle迁移。')
    line('当前标签只反映每预算每轮一次生成。若研究“可靠成功集合”，应在固定协议下重复生成并定义成功概率/下置信界，而不是把两轮观察直接解释成真实概率。')
    line('后续可检验更具体的证据语义信号（实体/关系覆盖、第一跳是否成立、证据冲突），但本次没有测定它们，不能宣称它们有效。')
    line('配对CI以题为单位，条件于OOF预测；未包含全部训练不确定性，未多重校正。任何方法都应对照固定集合及共享难度基线。')
    s.save('ABSOLUTE_AUC_INTERVALS.json',extra)
    (s.DEST/'REPORT_ZH.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    dest=a.WORK/'deliverables/HotpotQA_success_budget_set_prediction_20261004.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(s.DEST.iterdir()):
            if f.is_file():z.write(f,'results/'+f.name)
        for name in ['SUCCESS_SET_PLAN.txt','success_set_prediction.py','report_success_sets.py','analyze.py']:
            z.write(a.BASE/name,name)
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    print(json.dumps(extra,ensure_ascii=False,indent=2))
    print('report and zip saved',dest)


if __name__=='__main__':main()
