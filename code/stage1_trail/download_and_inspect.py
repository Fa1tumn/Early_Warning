"""
阶段1任务：下载 TRAIL benchmark（PatronusAI/TRAIL），读几条 span 级错误标注样例，
确认字段是否包含需要的时间/位置信息。

仓库结构（已探明，通过 HfApi.dataset_info 列出）：
- GAIA/<hash>.json, SWE Bench/<hash>.json          : 原始 agent 执行轨迹
- processed_annotations_gaia/<hash>.json           : 对应轨迹的 span 级错误标注（核心）
- processed_annotations_swe_bench/<hash>.json
- data/gaia-....parquet, data/swe_bench-....parquet: 打包好的 parquet 版本（含 trace+annotation）
"""

import json
from collections import Counter
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "PatronusAI/TRAIL"

files = list_repo_files(REPO_ID, repo_type="dataset")
gaia_traces = sorted(f for f in files if f.startswith("GAIA/"))
gaia_annos = sorted(f for f in files if f.startswith("processed_annotations_gaia/"))
swe_traces = sorted(f for f in files if f.startswith("SWE Bench/"))
swe_annos = sorted(f for f in files if f.startswith("processed_annotations_swe_bench/"))

print(f"GAIA 原始轨迹: {len(gaia_traces)} 条, 标注: {len(gaia_annos)} 条")
print(f"SWE-Bench 原始轨迹: {len(swe_traces)} 条, 标注: {len(swe_annos)} 条")

# 下载一条 GAIA 的 trace + 对应 annotation 做样例
sample_hash = gaia_traces[0].split("/")[1]  # e.g. 0035f455....json
trace_path = hf_hub_download(REPO_ID, gaia_traces[0], repo_type="dataset")
anno_path = hf_hub_download(REPO_ID, f"processed_annotations_gaia/{sample_hash}", repo_type="dataset")

with open(trace_path, encoding="utf-8") as f:
    trace = json.load(f)
with open(anno_path, encoding="utf-8") as f:
    anno = json.load(f)

print(f"\n=== 原始轨迹样例 ({gaia_traces[0]}) ===")
print(f"类型: {type(trace)}")
if isinstance(trace, dict):
    print("顶层keys:", list(trace.keys()))
elif isinstance(trace, list):
    print(f"list长度: {len(trace)}, 第一个元素keys: {list(trace[0].keys()) if isinstance(trace[0], dict) else type(trace[0])}")

print(f"\n=== span级标注样例 (processed_annotations_gaia/{sample_hash}) ===")
print(f"类型: {type(anno)}")
if isinstance(anno, dict):
    print("顶层keys:", list(anno.keys()))
    print(json.dumps(anno, ensure_ascii=False, indent=2)[:3000])
elif isinstance(anno, list):
    print(f"list长度: {len(anno)}")
    print(f"第一条 keys: {list(anno[0].keys())}")
    print(json.dumps(anno[0], ensure_ascii=False, indent=2)[:3000])

# 批量下载所有 GAIA 标注文件，统计错误类别分布 + 检查时间/位置字段
print("\n=== 批量拉取全部 GAIA 标注，统计错误类别分布 ===")
all_annos = []
parse_failures = []
for f in gaia_annos:
    p = hf_hub_download(REPO_ID, f, repo_type="dataset")
    with open(p, encoding="utf-8") as fh:
        try:
            all_annos.append((f, json.load(fh)))
        except json.JSONDecodeError as e:
            parse_failures.append((f, str(e)))
if parse_failures:
    print(f"\n跳过 {len(parse_failures)} 个JSON格式有问题的标注文件: {[f for f, _ in parse_failures]}")

category_counter = Counter()
location_fields_seen = set()
time_fields_seen = set()
total_errors = 0

for fname, a in all_annos:
    errors = a.get("errors") if isinstance(a, dict) else (a if isinstance(a, list) else [])
    if isinstance(errors, dict):
        errors = errors.get("errors", [])
    if not isinstance(errors, list):
        continue
    for err in errors:
        if not isinstance(err, dict):
            continue
        total_errors += 1
        cat = err.get("category") or err.get("error_category") or err.get("type")
        if cat:
            category_counter[cat] += 1
        for k in err.keys():
            lk = k.lower()
            if any(t in lk for t in ("span", "step", "index", "location", "position")):
                location_fields_seen.add(k)
            if any(t in lk for t in ("time", "timestamp")):
                time_fields_seen.add(k)

print(f"标注文件数: {len(all_annos)}, 累计错误条数: {total_errors}")
print(f"发现的位置类字段: {location_fields_seen}")
print(f"发现的时间类字段: {time_fields_seen}")
print("\n错误类别分布 (前20):")
for cat, cnt in category_counter.most_common(20):
    print(f"  {cnt:4d}  {cat}")

out_path = "gaia_annotations_sample.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump([a for _, a in all_annos[:10]], f, ensure_ascii=False, indent=2)
print(f"\n已把前10条标注样例存到 {out_path}")
