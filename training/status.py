"""Print one line per finished run/checkpoint eval from /content/ft_eval/summary.jsonl (+ current activity)."""
import glob
import json
import os

p = "/content/ft_eval/summary.jsonl"
if os.path.exists(p):
    for line in open(p):
        r = json.loads(line)
        for k, v in (r.get("evals") or {}).items():
            f = lambda x, n=2: "-" if v.get(x) is None else round(v[x], n)  # noqa: E731
            print(f"RES {r['run']}@{k} cer={f('cer_ci')} cap={f('cer_cap')} pres={f('err_preserved', 1)} inv={f('invented')} "
                  f"strikeR={f('strike_R', 0)} insR={f('insert_R', 0)} lines_ok={f('lines_ok', 0)}")
        if r.get("error"):
            print(f"ERR {r['run']} {r['error']}")
logs = sorted(glob.glob("/content/ft_logs/*.train.log"), key=os.path.getmtime)
if logs:
    t = open(logs[-1], errors="ignore").read()
    if "Traceback" in t[-4000:]:
        print("TRACEBACK", os.path.basename(logs[-1]))
