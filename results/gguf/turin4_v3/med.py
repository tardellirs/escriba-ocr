import json, statistics as s, sys, os
root = sys.argv[1]
for v in ["f16", "q8", "q4f16", "q4q8", "q4q4"]:
    p = os.path.join(root, v, "meta.jsonl")
    if not os.path.exists(p): continue
    r = [json.loads(l) for l in open(p)]
    print(f"{v:6s} n={len(r):2d} lat {s.median(x['latency_s'] for x in r):5.1f}  enc {s.median(x['prompt_ms'] for x in r)/1000:5.1f}  gen {s.median(x['predicted_per_second'] for x in r):5.1f} tok/s  peak {max(x['peak_rss_mb'] for x in r)} MB  stop {sum(x['finish']=='stop' for x in r)}")
