import json, sys, datetime, os, re
p = r"D:\DKD\NJIT Courses\RAG Security\KidnapRag\dataset_builder\results_sample\chunk_00.json"
d = json.load(open(p, encoding='utf-8'))
for f in sys.argv[1:]:
    e = json.load(open(f, encoding='utf-8'))
    e["timestamp"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    d.append(e)
    json.dump(d, open(p, "w", encoding='utf-8'), indent=2, ensure_ascii=False)
print(len(d))
