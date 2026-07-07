"""
阶段1任务：下载并跑通 MAST-Data（HuggingFace mcemri/MAD, MAD_full_dataset.json），
筛出标签为 "Step Repetition" 的轨迹，读几条样例，感受数据格式。

数据结构（已探明）：
- 顶层是 list，每条记录 keys: mas_name, llm_name, benchmark_name, trace_id, trace, mast_annotation
- mast_annotation 是 MAST 14 类失败模式的 0/1 标注字典，key 为论文中的编号，如 "1.1".."3.3"
- 已通过 MAST 论文 (arXiv:2503.13657) 确认：编号 "1.3" = Step Repetition
  （FC1 系统设计问题类下的第3项，"unnecessary reiteration of previously completed steps"）
"""

import json
from collections import Counter
from huggingface_hub import hf_hub_download

REPO_ID = "mcemri/MAD"
FILENAME = "MAD_full_dataset.json"
STEP_REPETITION_CODE = "1.3"

print(f"Downloading {FILENAME} from {REPO_ID} ...")
path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
print(f"Downloaded to: {path}")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"\n总记录数: {len(data)}")
print(f"单条记录字段: {list(data[0].keys())}")
print(f"mast_annotation 编码 (MAST 14类失败模式 0/1): {list(data[0]['mast_annotation'].keys())}")

step_rep_records = [r for r in data if r["mast_annotation"].get(STEP_REPETITION_CODE) == 1]
print(f"\n=== Step Repetition ({STEP_REPETITION_CODE}) 轨迹数: {len(step_rep_records)} / {len(data)} "
      f"({len(step_rep_records)/len(data):.1%}) ===")

print("\nmas_name 分布:", Counter(r["mas_name"] for r in step_rep_records))
print("benchmark_name 分布:", Counter(r["benchmark_name"] for r in step_rep_records))

# 同时统计"完全没有任何失败标注"的轨迹数，供后续做对照参考
clean_records = [r for r in data if sum(r["mast_annotation"].values()) == 0]
print(f"\n无任何失败标注的干净轨迹数: {len(clean_records)}")

# 保存一份精简样例（截断超长trajectory文本，只看结构和前2000字符）
def brief(rec, trunc=2000):
    r = dict(rec)
    r["trace"] = dict(rec["trace"])
    if isinstance(r["trace"].get("trajectory"), str):
        full_len = len(r["trace"]["trajectory"])
        r["trace"]["trajectory"] = r["trace"]["trajectory"][:trunc] + f"...[截断，全长{full_len}字符]"
    return r

sample_out = [brief(r) for r in step_rep_records[:20]]
out_path = "step_repetition_samples.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(sample_out, f, ensure_ascii=False, indent=2)
print(f"\n已把前20条 Step Repetition 样例（trajectory截断到{2000}字符）存到 {out_path}")

print("\n=== 第一条 Step Repetition 样例概览 ===")
first = step_rep_records[0]
print(f"mas_name={first['mas_name']}  llm_name={first['llm_name']}  benchmark={first['benchmark_name']}")
print(f"其余同时命中的失败模式: {[k for k, v in first['mast_annotation'].items() if v == 1]}")
print(f"trajectory 长度: {len(first['trace']['trajectory'])} 字符")
print(f"trajectory 前500字符:\n{first['trace']['trajectory'][:500]}")
