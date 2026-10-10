"""Render a human-readable audit report from independently recomputed results."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
R = BASE / 'results'
s = json.loads((R / 'AUDIT_SUMMARY.json').read_text(encoding='utf-8'))
t = json.loads((R / 'SUPPLEMENT.json').read_text(encoding='utf-8'))
em = s['primary']['answer_em']
f1 = s['primary']['answer_f1']
acc = s['primary']['answer_acc']


def cost_table(section):
    lines = ['方案 | EM | F1（原口径） | 词法ACC | 平均输入token | 平均输出token | 平均总token | 平均可见证据token']
    for key, name in [('old_oracle', '旧轮本轮Oracle'), ('new_at_old_budget', '旧轮预算迁移到新轮'),
                      ('new_oracle', '新轮本轮Oracle'), ('new_FULL', '新轮FULL基线')]:
        x = section[key]
        lines.append(f"{name} | {x['answer_em']:.2%} | {x['answer_f1']:.4f} | {x['answer_acc']:.2%} | "
                     f"{x['mean_input_tokens']:.2f} | {x['mean_output_tokens']:.2f} | {x['mean_total_tokens']:.2f} | {x['mean_visible_tokens']:.2f}")
    return '\n'.join(lines)


text = f'''HotpotQA 重合500题：原始轨迹独立审计
审计日期：2026-10-04。没有发起模型、Retriever或LLM Judge调用。

一、结论先行

1. 这次上传确实含原始prompt、raw_output、处理后output、逐轮完整检索文档、可见证据和最终答案。
   压缩包SHA256及13个成员文件校验和一致。不是仅凭摘要指标推断模型行为。
2. 两轮同一500题分别4180和4173条轨迹；同题同预算4131对。其余49/42条各自保留。
   500题问题文本完全一致。根引用、预算条件完整性、调用用量对账全部通过。
3. 8353条轨迹的EM、原口径F1、词法ACC从final_answer独立复算，与存储分数完全一致。
4. 单题旧Oracle预算的跨轮退化确实存在；不能主要解释为格式错误，也不能说新轮机会消失。
5. 首条prompt在500题上完全相同，温度全部记录为0，生成上限全部8192；但490题根输出变化，
   307题首条query变化。差异在证据预算干预前已出现。这不是温度设置不同的证据。
6. 原始数据也显示旧轮含历史证据截取实现的留存结果，故不能仅凭最终源码包相同就断言历史
   执行完全相同。不过可观察文本差异主要是Unicode规范化；不能直接把它认定为Oracle退化主因。

二、计分与比较口径

所有主比较严格使用同一500题、共同七档预算：100、200、300、400、500、600、FULL。
各档是每次搜索的证据暴露上限，FULL为该轮完整Top-5；不是整个问题的固定总token上限。
模型最多搜索4次，自然决定继续搜索还是回答；第一条query由共享根阶段生成。

选择目标与评估目标一致：EM Oracle按EM选；F1 Oracle按F1选；ACC Oracle按词法ACC选。
质量并列时先取该轮standalone输入+输出token最少的轨迹，再按预算顺序打破并列。
因此Oracle也利用了事后已知的成本，不能把其成本视为在线保证。
local_fallback在全部主指标中计零，不因为末尾碰巧包含答案而恢复正确。

ACC是规范化后gold答案是否为prediction的子串，不是LLM Judge或语义准确率。
F1沿用原脚本的规范化词重叠口径，未声称它实现了HotpotQA官方评测的全部特殊处理。
EM正确并不等同于语义唯一正确；例如Australia/Oceania、England/United Kingdom可能涉及粒度差异。
未进行人工语义重标或LLM Judge，因此这些案例只能称为EM得分翻转。

平均总token=每题所选一条轨迹的共享根输入/输出+全部续推输入/输出，再对500题求平均。
包括反复输入历史证据的成本，不等于可见证据token；不包含枚举所有候选的研究开销、检索器计算量。
输出token沿用客户端对处理后output的重新计数，包含补齐stop分隔符，不声称等于服务端账单token。

三、EM选择和EM评估：主要结果

{cost_table(em)}

旧轮选中轨迹242题EM正确，新轮同预算仍正确190题：190/242={190/242:.2%}。
旧对新错52题、旧错新对5题；所以48.4%降为39.0%，而不是简单丢失52/500。
52题中41题在新轮共同七档内仍有EM正确轨迹，仅11题在该候选集合内没有正确轨迹。
这区分了“赢家更替”和“本次观察网格中的机会消失”，不是对所有可能推理轨迹的断言。

相对新轮FULL：原本FULL错的309题中救回31题；原本FULL对的191题中损害27题。
净增4/500=0.8个百分点，配对bootstrap 95% CI为[-2.2,+3.8]个百分点；
McNemar双侧精确p={em['full_to_transfer']['mcnemar_exact_p']:.4f}，未显示显著EM提升。
平均总token从4504.18降至3510.74，减少993.44，节省22.06%；
配对bootstrap 95% CI为[-1257.14,-734.18] token/题。
未显著不同不等于已证明质量等价或非劣；不能宣称已经实现稳定降本增质。
预算选择使用旧轮同题gold，故这仍是有监督的事后跨轮诊断，不是无gold在线可部署方法。

四、避免混用目标：此前39.4% / F1=0.5284属于F1 Oracle迁移

{cost_table(f1)}

F1目标迁移相对新轮FULL：F1增加{f1['transfer_minus_FULL']['quality_delta']:.4f}，
95% CI=[{f1['transfer_minus_FULL']['quality_delta_ci95'][0]:.4f},{f1['transfer_minus_FULL']['quality_delta_ci95'][1]:.4f}]；
平均总token减少{f1['transfer_minus_FULL']['savings_fraction']:.2%}。
按词法ACC独立选预算时：旧轮Oracle ACC={acc['old_oracle']['answer_acc']:.2%}，
迁移ACC={acc['new_at_old_budget']['answer_acc']:.2%}，新轮Oracle ACC={acc['new_oracle']['answer_acc']:.2%}，
新轮FULL ACC={acc['new_FULL']['answer_acc']:.2%}。不能用F1选出的39.4%去冒充EM Oracle迁移结果。

补充敏感性分析保留每题两轮共有的全部预算，不限于共同七档：
EM目标迁移为{s['per_question_common_grid']['answer_em']['new_at_old_budget']['answer_em']:.2%}，
F1目标迁移F1为{s['per_question_common_grid']['answer_f1']['new_at_old_budget']['answer_f1']:.4f}。
数值会随候选集合改变；完整定义和逐题选择见JSON，不混合使用不同集合。

五、原始输入和输出真正发生了什么

首条根prompt：500/500完全相同；根raw_output：只有10/500完全相同。
根query：193/500相同；根检索内容及顺序：214/500相同。
193题相同query的检索内容及顺序全部相同。
不仅是根：两轮全部续搜中的699个共有精确query字符串，检索内容及顺序也全部一致。
本数据未显示同query下的检索结果漂移；query本身变化是另一回事。

4131对同题同预算续推中，首个续推prompt文本1526对完全相同。
这些相同prompt对中，1354对的首次raw_output仍然不同，占88.73%。
公开参考tokenizer比较显示1613对首个输入token序列相同，其中1432对输出不同。
文本完全相同是无需假设tokenizer即可验证的结论；参考token序列结果的限定见第八节。

EM Oracle丢失的52题：
- 34题首个续推prompt已经不同（其根query均不同）。
- 18题首个续推prompt相同，其中17题第一次生成输出即分歧，1题在第二次生成才分歧。
- 首prompt相同组旧正确91题中失去18题，失效率19.78%；不同组151题中失去34题，22.52%。
这些是观测分层，不是各机制的因果贡献；相同根并不保证后续输出/新query相同。

根请求分配也不同：245题记录在同一端口，255题换了端口。
同端口组240/245根输出变化、152/245 query变化；异端口组250/255输出变化、155/255 query变化。
这不支持把问题简单归因于“换了GPU”；端口相同也不代表同一服务进程、批次或运行环境。

六、格式、截断、上下文和协议审计

根生成：旧8/500、新15/500 finish_reason=length，并以问题作为fallback query。
续推fallback：旧108/4180、新115/4173。
其中真实length终止：旧99、新106；stop但格式未解析：两轮各9。
这18条stop格式异常全部来自同一道题5a90a9a45542990a9849367e，末尾出现
类似“<answer подокарпус</answer>”的畸形起始标签；并非stop标签被正常省略的问题。
它们已正确按失败计零，不能借修复评估器悄悄加回主分数。

52个EM损失案例里，新续推截断并fallback仅8个，其余44个正常得到可解析答案。
即使把任一轮根截断也纳入，涉及异常的损失案例共10个，仍不能解释全部52个。
两轮context_compactions均0；逐轮保留证据数量及完整prompt重建全部通过，未发现隐性历史删减。

观察到旧265、新316次续搜在</think>前输出<search>，控制器据标签执行了搜索。
这些全部是search事件，不是answer；属于当前解析器允许的协议行为，不能自动计为失败。
未证实这些搜索是示例标签误触发；应与严格要求think闭合的协议区分。
stop分隔符由客户端按finish_reason补齐是预期行为，已核对raw_output与output的关系。

最重要的状态实现事实：1000个根prompt和14669个续推prompt均从本题question及保存的
query+可见证据完整重建一致。下一轮不追加之前的整段模型think，其他题的对话也不在prompt里。
因此“历史不压缩”指保留检索轮次的query/证据，不等于把每次原始思考文本持续放入上下文。
仅从结果文件不能证明服务端权重文件字节不变；这里检查的是已保存客户端输入。

七、确有旧实现留存，但不能夸大Unicode差异

旧轮7397轮证据中7395轮未记录新版prefix_boundary_policy，仅2轮带新版original-text-offset标记；
新轮7272轮全部带新版标记。这与旧轮保留历史成功记录、修补个别边界的实现路径相符。
这是历史记录字段事实；不能仅凭最终v4.3.3压缩包名称说所有历史记录都由同一实现生成。

旧轮481轮可见文本不是原文的逐字符前缀，但全部在NFC规范化后成为前缀；新轮此数为0。
公开Qwen3参考tokenizer的normalizer是NFC。本次重新分词核验：
- 15669次保存的模型调用：prompt输入token与处理后output输出token计数全匹配，0误差。
- 14669轮证据：可见token计数和原始完整证据token前缀全匹配，0错误。
- 同query、同文档但首续推prompt文本不同的87对，参考tokenizer的输入token序列全部相同。
因此版本留存需要披露，但当前观察的Unicode文本差异不能作为实质证据改变的证明。
没有发现“标成100 token却实际读入另一段内容”的预算结构错误。

八、仍无法确认的事项，以及原结论应如何收紧

可确认：所保存prompt相同、所记录温度均0、上限8192、Top-5、最多4搜、计分一致。
不能确认：两轮完整服务端启动参数/运行版本/模型权重hash/tokenizer文件hash/请求调度与批次，
以及每个请求服务端实际使用的完整sampling payload。模型路径相同不是权重字节相同的证明。
包中的新轮服务快照也不能替代旧轮日志或全程运行跟踪。

参考tokenizer来自先前独立审计保存的公开Qwen3 tokenizer，不是本次打包的服务器文件。
全部计数一致是强一致性检查，但不能据此声称服务器tokenizer文件字节身份已验证。

SGLang当前官方文档说明，即使temperature=0，动态批处理和GPU浮点归约顺序也可能令输出改变：
https://docs.sglang.io/docs/advanced_features/deterministic_inference
这是一个可检验的解释，不是本次已证实的原因；也不意味着旧版本支持当前文档的所有参数。
不能再直接说“温度不同导致不复现”或“已证实一定是CUDA非确定性”。

最稳妥结论：机会在两轮中仍存在；同题最优预算并非稳定标签。当前数据确认了整个agent执行
链路存在跨轮变化，但不能把这种变化全部归为阅读预算策略本身不泛化，更不能证明所有在线
策略无价值。反过来，也不能凭事后Oracle大就声称已经有可部署的准确率提升。

若要区分机制，下一项最小验证应是离线冻结相同根query、同序文档、实际证据token前缀与
完整prompt，在记录完整服务配置后重放。先检验同prompt生成稳定性，再谈跨轨迹预算学习。
本审计没有启动重放，也没有修改任何实验设置或主结果。

九、可追溯案例和文件

示例1：5a74167455429929fddd83dc，FULL，Pia Miller/Dolly所在大陆。
两轮首query和首续推prompt相同；答案由Australia变为Oceania，EM由1变0；新轮200预算又答Australia。
说明存在正常格式下的答案变化，同时提醒EM翻转包含地理粒度差异，不能直接当作语义能力失败。

示例2：5a736f7c5542991f29ee2e18，100预算，1839年签署宪法的部落政府。
query从“tribe signed constitution Oklahoma September 6 1839”变为
“Oklahoma tribe constitution signed September 6 1839”；答案由Cherokee Nation变为Chickasaw Nation；
新轮400预算存在正确答案。该例来自不同初始轨迹，而不是冻结状态上的纯预算重放。

RAW_CASES.json保存11个按审计条件确定性选出的完整案例（不是随机代表性样本），含两轮根、
prompt、输出、检索证据与gold标签。原始包保留全部记录；未在报告中展示全部思考文本。
AUDIT_SUMMARY.json：完整汇总、各目标、七档/逐题共同档敏感性分析、每题平均token成本。
per_question_answer_em/answer_acc/answer_f1.json：每题预算选择、质量、成本、输入差异及异常标记。
root_pairs.json与common_trajectory_pairs.json：500根和4131对轨迹的精确匹配检查。
SUPPLEMENT.json：共享query、格式失败案例、运行端口分层。

统计方法：以问题为配对单位，10000次bootstrap，随机种子20261003；
McNemar使用双侧精确二项检验。分组诊断为探索性，无因果解释，无多重比较校正。
本报告只关于同一500题，不把结果外推为全7405题的在线效果或独立新题泛化。
'''
(BASE / 'AUDIT_REPORT_ZH.txt').write_text(text, encoding='utf-8')
print(BASE / 'AUDIT_REPORT_ZH.txt')
