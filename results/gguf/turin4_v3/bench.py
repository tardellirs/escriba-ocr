import base64, json, os, subprocess, sys, time, urllib.request
from pathlib import Path
name, tq, mq = sys.argv[1], sys.argv[2], sys.argv[3]
B = Path('/opt/cpubench'); OUT = B / 'out' / name; (OUT / 'pred').mkdir(parents=True, exist_ok=True)
G = Path('/opt/gguf')
import socket
s0 = socket.socket()
assert s0.connect_ex(('127.0.0.1', 8010)) != 0, 'port 8010 busy: stale server'
s0.close()
srv = subprocess.Popen(['/opt/llama.cpp/build/bin/llama-server', '-m', str(G / f'Escriba-OCR-1B-{tq}.gguf'), '--mmproj', str(G / f'mmproj-Escriba-OCR-1B-{mq}.gguf'),
                        '--port', '8010', '-c', '8192', '--temp', '0', '--repeat-penalty', '1.05', '--cache-ram', '0', '-np', '1', '-t', '4'],
                       stdout=open(OUT / 'server.log', 'w'), stderr=subprocess.STDOUT)
for _ in range(600):
    try: urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=2); break
    except Exception: time.sleep(1)
assert srv.poll() is None, 'server exited'
for pid in (B / 'ids.txt').read_text().split():
    img = base64.b64encode((B / 'pages' / f'{pid}.png').read_bytes()).decode()
    body = {"model": "x", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]}],
            "temperature": 0.0, "repetition_penalty": 1.05, "repeat_penalty": 1.05, "max_tokens": 2048}
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8010/v1/chat/completions', data=json.dumps(body).encode(),
                                                                 headers={'Content-Type': 'application/json'}), timeout=3600).read())
    lat = time.perf_counter() - t0
    (OUT / 'pred' / f'{pid}.txt').write_text((r['choices'][0]['message'].get('content') or '').strip() + '\n')
    rss = int([l for l in open(f'/proc/{srv.pid}/status') if l.startswith('VmHWM')][0].split()[1]) // 1024
    cur = int([l for l in open(f'/proc/{srv.pid}/status') if l.startswith('VmRSS')][0].split()[1]) // 1024
    tm = r.get('timings', {})
    row = {'variant': name, 'page': pid, 'latency_s': round(lat, 2), 'prompt_n': tm.get('prompt_n'), 'prompt_ms': tm.get('prompt_ms'),
           'predicted_n': tm.get('predicted_n'), 'predicted_ms': tm.get('predicted_ms'), 'predicted_per_second': tm.get('predicted_per_second'),
           'finish': r['choices'][0].get('finish_reason'), 'peak_rss_mb': rss, 'rss_mb': cur}
    open(OUT / 'meta.jsonl', 'a').write(json.dumps(row) + '\n'); print(json.dumps(row), flush=True)
srv.terminate(); srv.wait()
print('BENCH_DONE', flush=True)
