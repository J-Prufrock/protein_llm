import json
import re

input_tsv = "data/uniportData/anti.tsv"
output_json = "data/uniportData/anti_cleaned.json"

data = []

# 批量清洗正则
patterns = [
    re.compile(r'^FUNCTION:\s*', re.I),       # 去掉 FUNCTION:
    re.compile(r'\{ECO:.+?\}', re.S),         # 去掉所有 {ECO:...}
    #re.compile(r'\[Isoform \d+\]:\s*'),       # 去掉 [Isoform 1]: 等
    re.compile(r'\s+'),                      # 多个空格变单个
]

with open(input_tsv, 'r', encoding='utf-8') as f:
    next(f)  # 跳过表头
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        
        entry = parts[0].strip()
        seq = parts[1].strip()
        func = parts[2].strip()

        # 依次清洗
        for pat in patterns:
            func = pat.sub(' ', func)
        
        # 清理首尾标点和空格
        func = func.strip(' .;')
        func = func.strip()

        data.append({
            "entry": entry,
            "sequence": seq,
            "function": func
        })

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"完成！共 {len(data)} 条，已输出至 {output_json}")