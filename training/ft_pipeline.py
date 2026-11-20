"""Run a queue of fine-tune experiments: train -> vLLM eval (val50 + valfoto [+ val154]) -> score.

Run inside the Colab kernel (keeps the VM busy):
    colab exec -s vlm --timeout 7200 -f bench/colab/ft_pipeline.py   (reads /content/ft_queue.json)
Queue file: list of {"run": name, "args": [...ft_lightonocr args...], "eval": ["val50","valfoto"], "keep": false}
A run with "baseline": true skips training and evaluates the hub model.
Results: /content/ft_eval/<set>/<run>/{pred,meta.jsonl,run_info.json} and /content/ft_eval/summary.jsonl
"""
import json
import shutil
import subprocess
import time
from pathlib import Path

Q = Path("/content/ft_queue.json")
EV = Path("/content/ft_eval")
EV.mkdir(exist_ok=True)
Path("/content/ft_logs").mkdir(exist_ok=True)
SETS = {
    "val50": ["--ids-file", "/content/ocr-model/bench/val50.txt"],
    "valfoto": ["--ids-file", "/content/ocr-model/bench/val50.txt", "--image-root", "/content/valfoto"],
    "val154": ["--split", "val"],
}


def run(cmd, log):
    """Run a command, heartbeating so `colab exec` never sees 30 s of silence."""
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd="/content")
        n = 0
        while p.poll() is None:
            time.sleep(15)
            n += 1
            print("tick" if n % 20 else f"[pipe] {Path(log).name} {(time.time() - t0) / 60:.0f} min", flush=True)
    return p.returncode


def score(set_name, run_name, split="val"):
    r = subprocess.run(["python3", "-c", (
        "import sys,json; sys.path.insert(0,'/content/ocr-model');"
        "from bench.evaluate import score; from pathlib import Path;"
        f"print(json.dumps(score('{run_name}','{split}',Path('{EV / set_name}'))))")],
        capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.stderr[-500:]}


done = set()
summ = EV / "summary.jsonl"
if summ.exists():
    done = {json.loads(l)["run"] for l in summ.read_text().splitlines() if l.strip()}

while True:
    queue = json.loads(Q.read_text())  # re-read each time so the queue can be extended while running
    todo = [e for e in queue if e["run"] not in done]
    if not todo:
        break
    e = todo[0]
    name = e["run"]
    print(f"[pipe] === {name}", flush=True)
    rec = {"run": name, "args": e.get("args", []), "t_start": time.strftime("%H:%M:%S")}
    if e.get("cmd"):  # arbitrary job (e.g. contrastive decoding); scored sets listed in "score"
        rc = run(e["cmd"], f"/content/ft_logs/{name}.cmd.log")
        rec["cmd_rc"] = rc
        rec["evals"] = {}
        for set_name, rname in e.get("score", []):
            sc = score(set_name, rname, "val")
            rec["evals"][f"{rname}/{set_name}"] = sc
            print(f"[pipe] {rname} {set_name}: " + json.dumps({k: sc.get(k) for k in ("cer_ci", "err_preserved", "invented")}), flush=True)
        with open(summ, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done.add(name)
        continue
    if e.get("baseline"):
        model_path = "lightonai/LightOnOCR-3-1B"
        long_side = e.get("long_side", 1540)
        
    else:
        rc = run(["python3", "/content/ft_lightonocr.py", "--run", name, *e["args"]], f"/content/ft_logs/{name}.train.log")
        rec["train_rc"] = rc
        ft = Path("/content/ft") / name
        rec["train"] = json.loads((ft / "done.json").read_text()) if (ft / "done.json").exists() else None
        model_path = str(ft / "final")
        cfg = json.loads((ft / "config.json").read_text())
        long_side = cfg["long_side"]
        if rc != 0 or not (ft / "final").exists():
            rec["error"] = "training failed"
            with open(summ, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done.add(name)
            continue
    prompt = "dipl" if "dipl" in e.get("args", []) else "empty"
    # evaluate every saved checkpoint (curve) -> name@stepN / name@final
    if e.get("baseline"):
        ckpts = [("base", model_path)]
    else:
        ft = Path("/content/ft") / name
        ckpts = sorted(((d.name, str(d)) for d in ft.iterdir() if d.is_dir() and (d.name.startswith("step") or d.name == "final")),
                       key=lambda x: (x[0] == "final", int(x[0][4:]) if x[0].startswith("step") else 0))
    rec["evals"] = {}
    for tag, path in ckpts:
        rname = f"{name}@{tag}"
        for s in e.get("eval", ["val154"]):
            sets = SETS.get(s) or ["--ids-file", str(Path("/content/ft") / name / "train_pages.txt"), "--split", "train"]
            if s == "trainfit":
                sets = ["--ids-file", str(Path("/content/ft") / name / "train_pages.txt"), "--split", "train"]
            rc = run(["python3", "/content/run_vllm.py", path, rname, "--prompt", prompt, "--seq", str(e.get("seq", 0)),
                      "--long-side", str(long_side), "--rep-penalty", str(e.get("rep", 1.05)),
                      "--out-root", str(EV / s), *sets], f"/content/ft_logs/{rname}.{s}.log")
            split = "train" if s == "trainfit" else "val"
            sc = score(s, rname, split) if rc == 0 else {"error": f"vllm rc={rc}"}
            rec["evals"][f"{tag}/{s}"] = sc
            print(f"[pipe] {rname} {s}: " + json.dumps({k: sc.get(k) for k in ("cer_ci", "cer_lines", "err_preserved", "invented", "strike_R", "insert_R", "lines_ok")}), flush=True)
        if tag.startswith("step") and not e.get("keep"):
            shutil.rmtree(path, ignore_errors=True)
    rec["t_end"] = time.strftime("%H:%M:%S")
    with open(summ, "a") as f:
        f.write(json.dumps(rec) + "\n")
    done.add(name)
    if not e.get("keep") and not e.get("baseline"):
        shutil.rmtree(Path("/content/ft") / name / "final", ignore_errors=True)
print("[pipe] QUEUE_DONE", flush=True)
