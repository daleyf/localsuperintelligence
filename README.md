# localsuperintelligence

other ai projects to bounce off:
1. https://github.com/daleyf/transparent-agent
2. https://github.com/daleyf/circle-it
3. https://github.com/daleyf/local-evals
4. https://github.com/daleyf/clipboard-agent
5. https://github.com/daleyf/personal_agent
6. https://github.com/daleyf/sf
7. https://github.com/daleyf/llm-tools-df

private, immediate, not-a-chatbot. Let’s ship a wild, runnable MVP on macOS that proves the 2027 vision now and lets you iterate without waiting on online models.

TimePilot — “While you live, it learns. While you sleep, it works.”
What it does (MVP scope)

Runs locally on your Mac all day, quietly logging only opt‑in activity metadata.

9:00 PM: pops a native macOS notification and opens a transparent Daily Update HTML page.

The report includes: Today in 7 lines, Codebase Intelligence (what changed, where you left off), Focus stats, Nudges, and a crisp Tomorrow plan.

Fully local by default using Qwen 2.5 (1.5B) in GGUF via llama.cpp Metal or via MLX on Apple Silicon. Optional cloud toggle for gpt‑5‑nano when you want extra polish. 
Hugging Face
GitHub
+1
OpenAI Platform

Why this is different (PC‑Agent inspired)

We borrow PC‑Agent’s cognition pipeline: (1) collect interaction traces, (2) transform them into cognitive trajectories, (3) run a tiny planning‑critic pair to produce advice and next actions. For macOS we implement a super‑light tracker and a two‑stage “plan → critique” pass. 
gair-nlp.github.io
GitHub

Alice (your user) will love it

Immediate: runs local, no spinner.

Transparent: every suggestion shows “why” with the underlying events.

Private: zero cloud by default; toggles per connector.

Not a chatbot: a nightly brief, not a DM thread.

Fast install (macOS, Apple Silicon)
Prereqs

macOS 14+ (M‑series recommended).

Python 3.11+.

Xcode CLT: xcode-select --install.

Optional: uv or pipx if you like.

If using llama.cpp Metal: we’ll compile with Metal acceleration. Docs: llama‑cpp‑python Metal install. 
llama-cpp-python.readthedocs.io

1) Create project

bash
Copy
mkdir -p ~/TimePilot && cd ~/TimePilot
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install jinja2 psutil schedule huggingface_hub
# Local engine A: llama.cpp w/ Metal
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir
# Optional local engine B: MLX (Apple)
pip install mlx-lm
# Optional cloud
pip install openai
2) Grab a small, fast Qwen

GGUF (for llama.cpp Metal): download Qwen2.5-1.5B-Instruct-Q6_K_L.gguf (great latency/quality tradeoff) from the official Qwen GGUF repo or Bartowski’s quants. Put the file path in GGUF_PATH. 
Hugging Face
+1

MLX (Apple’s framework): use MLX‑LM with Qwen MLX checkpoints (Qwen team ships MLX formats). Set MLX_REPO_ID to the MLX model id. 
GitHub
qwen.readthedocs.io
Reuters

3) Drop in this single‑file MVP (main.py)

This runs a tracker loop, a tiny PC‑style plan→critic pipeline, and renders an HTML report at 9:00 PM. Local by default; flip engines with an env var.

python
Copy
# main.py
import os, time, json, sqlite3, subprocess, datetime, psutil, schedule
from dataclasses import dataclass
from typing import List, Dict, Any
from jinja2 import Template

# ---------- Config ----------
DATA_DIR = os.path.expanduser("~/Library/Application Support/TimePilot")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "logs.sqlite")
REPORT_PATH = os.path.join(DATA_DIR, "daily.html")
ENGINE = os.getenv("ENGINE", "llamacpp")  # "llamacpp" | "mlx" | "openai"
GGUF_PATH = os.getenv("GGUF_PATH")  # e.g., ".../Qwen2.5-1.5B-Instruct-Q6_K_L.gguf"
MLX_REPO_ID = os.getenv("MLX_REPO_ID")  # e.g., "mlx-community/Qwen2.5-1.5B-Instruct-4bit-mlx"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")  # optional cloud
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------- Storage ----------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS events(
        ts INTEGER, app TEXT, title TEXT, kind TEXT, meta TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS facts(
        day TEXT PRIMARY KEY, json TEXT
    )""")
    return conn

# ---------- macOS Helpers ----------
def front_app_and_title():
    # AppleScript: get frontmost app + window title if accessible
    script = r'''
        tell application "System Events"
            set frontApp to name of first application process whose frontmost is true
        end tell
        set winTitle to ""
        try
            tell application frontApp
                try
                    set winTitle to name of front window
                end try
            end tell
        end try
        return frontApp & "||" & winTitle
    '''
    out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if out.returncode != 0: return ("Unknown", "")
    parts = out.stdout.strip().split("||")
    return (parts[0] if parts else "Unknown", parts[1] if len(parts) > 1 else "")

def notify(title, subtitle, text):
    # Native notification
    osa = f'display notification "{text}" with title "{title}" subtitle "{subtitle}"'
    subprocess.run(["osascript", "-e", osa])

# ---------- Engines ----------
class LLM:
    def complete(self, sys_prompt: str, prompt: str, max_tokens=600, temperature=0.2)->str:
        raise NotImplementedError

class LlamaCppEngine(LLM):
    def __init__(self):
        assert GGUF_PATH and os.path.exists(GGUF_PATH), "Set GGUF_PATH to your Qwen gguf"
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=GGUF_PATH,
            n_ctx=8192,
            n_gpu_layers=-1,  # Metal
            logits_all=False,
            verbose=False
        )
    def complete(self, sys_prompt, prompt, max_tokens=600, temperature=0.2):
        full = f"<|system|>\n{sys_prompt}\n</s>\n<|user|>\n{prompt}\n</s>\n<|assistant|>\n"
        out = self.llm(full, max_tokens=max_tokens, temperature=temperature, stop=["</s>"])
        return out["choices"][0]["text"].strip()

class MLXEngine(LLM):
    def __init__(self):
        assert MLX_REPO_ID, "Set MLX_REPO_ID to a Qwen MLX checkpoint"
        from mlx_lm import load
        self.model, self.tokenizer = load(MLX_REPO_ID)
        from mlx_lm import generate
        self._generate = generate
    def complete(self, sys_prompt, prompt, max_tokens=600, temperature=0.2):
        full = f"[SYSTEM]\n{sys_prompt}\n\n[USER]\n{prompt}\n\n[ASSISTANT]\n"
        return self._generate(self.model, self.tokenizer, full, max_tokens=max_tokens, temp=temperature)

class OpenAIEngine(LLM):
    def __init__(self):
        assert OPENAI_API_KEY, "Set OPENAI_API_KEY"
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
    def complete(self, sys_prompt, prompt, max_tokens=600, temperature=0.2):
        # Responses API
        r = self.client.responses.create(
            model=self.model,
            input=[{"role":"system","content":sys_prompt},
                   {"role":"user","content":prompt}],
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        return r.output_text.strip()

def get_engine() -> LLM:
    if ENGINE == "mlx": return MLXEngine()
    if ENGINE == "openai": return OpenAIEngine()
    return LlamaCppEngine()

# ---------- Tracker ----------
def track_loop():
    conn = db()
    while True:
        ts = int(time.time())
        app, title = front_app_and_title()
        evt = ("activity", {"cpu": psutil.cpu_percent(interval=None), "focus": app})
        conn.execute("INSERT INTO events VALUES(?,?,?,?,?)",
                     (ts, app, title[:300], evt[0], json.dumps(evt[1]) ))
        conn.commit()
        time.sleep(10)  # 10s sampling; adjust as needed

# ---------- PC-style cognition ----------
SYS = """You are TimePilot, a private, local coach. Explain *why* for each suggestion.
Be concise. Prefer bullet points. Avoid fluff. Produce JSON with keys:
{ "today_lines": [..7 lines..],
  "code_insights": [ { "repo": str, "files_changed": int, "next_step": str, "why": str } ],
  "focus": { "top_apps": [ { "app": str, "minutes": int } ], "observations": [str] },
  "nudges": [ { "tip": str, "why": str } ],
  "tomorrow": [ "3-5 concrete tasks" ]
}"""

def build_day_context(conn)->Dict[str,Any]:
    start = datetime.datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
    start_ts = int(start.timestamp())
    cur = conn.execute("SELECT ts, app, title, kind, meta FROM events WHERE ts>=? ORDER BY ts", (start_ts,))
    rows = cur.fetchall()
    # derive minutes per app
    buckets = {}
    last_ts = None; last_app = None
    for ts, app, title, kind, meta in rows:
        if last_ts and last_app:
            dt = min(120, max(0, ts - last_ts))  # cap gaps
            buckets[last_app] = buckets.get(last_app, 0) + dt/60
        last_ts, last_app = ts, app
    top = sorted([ (a,int(m)) for a,m in buckets.items() ], key=lambda x:-x[1])[:5]
    # tiny code intel: scan git repos under ~/Projects changed today
    repos = []
    projects = os.path.expanduser("~/Projects")
    if os.path.isdir(projects):
        for name in os.listdir(projects):
            path = os.path.join(projects, name)
            if os.path.isdir(os.path.join(path, ".git")):
                try:
                    out = subprocess.check_output(
                        ["git","-C",path,"log","--since=midnight","--pretty=%h %s","--name-only","--no-merges"],
                        text=True, timeout=2)
                    changed = [ln for ln in out.splitlines() if "/" in ln or ln.endswith(".py") or ln.endswith(".js")]
                    if out.strip():
                        repos.append({"repo": name, "files_changed": len(set(changed)), "raw": out[:1500]})
                except Exception:
                    pass
    return {"top_apps": [{"app":a,"minutes":m} for a,m in top], "repos": repos, "event_count": len(rows)}

def plan_and_critique(llm: LLM, ctx: Dict[str,Any])->Dict[str,Any]:
    # Planner
    plan_prompt = f"""Context:
Events today: {ctx['event_count']} | Top apps: {ctx['top_apps']}
Repos touched: {[{'repo':r['repo'], 'files_changed': r['files_changed']} for r in ctx['repos']]}
Write a JSON plan per SYS schema. Prefer concrete, verifiable facts. If TikTok/YouTube dominated, call it out gently and propose limits.
For code_insights.next_step, reference changed files or commit summaries (if provided)."""
    draft = llm.complete(SYS, plan_prompt, max_tokens=700)
    # Critic pass: add "why" and sanity-check
    critic_prompt = f"""You are a strict critic. Improve the JSON by:
- adding a "why" explanation on each item using explicit evidence from: {ctx['top_apps']} and repos={ [r['repo'] for r in ctx['repos']] }.
- keeping the structure identical.
Here is the JSON to refine:
{draft}"""
    final_json = llm.complete(SYS, critic_prompt, max_tokens=700)
    try:
        data = json.loads(final_json)
        return data
    except Exception:
        # fallback: minimal safe payload
        return {
            "today_lines": ["You built a first TimePilot draft."],
            "code_insights": [],
            "focus": {"top_apps": ctx["top_apps"], "observations": []},
            "nudges": [{"tip":"Pick 1 deep-work block for tomorrow","why":"Top app showed context switching"}],
            "tomorrow": ["Ship v0.1", "Write README", "Cut 1 scope"]
        }

# ---------- Report ----------
HTML = """<!doctype html><html><head>
<meta charset="utf-8"><title>TimePilot — Daily</title>
<style>
body{font-family:ui-sans-serif,system-ui;max-width:820px;margin:40px auto;padding:0 16px;color:#0a0a0a;background:#fff}
h1{font-size:28px;margin:8px 0}
h2{font-size:18px;margin:18px 0 6px}
.card{border:1px solid #eee;border-radius:10px;padding:14px;margin:10px 0}
.small{color:#777;font-size:12px}
ul{margin:8px 0 0 18px}
.code{background:#111;color:#eee;padding:10px;border-radius:8px;white-space:pre-wrap}
.tag{display:inline-block;background:#f5f5f5;border-radius:999px;padding:2px 8px;margin-right:6px}
</style></head><body>
<h1>TimePilot — Daily Update</h1>
<div class="small">{{today}}</div>
<div class="card"><h2>Today in 7 lines</h2><ul>{% for l in d.today_lines %}<li>{{l}}</li>{% endfor %}</ul></div>
<div class="card"><h2>Focus</h2>
<div>{% for a in d.focus.top_apps %}<span class="tag">{{a.app}} · {{a.minutes}}m</span>{% endfor %}</div>
<ul>{% for o in d.focus.observations %}<li>{{o}}</li>{% endfor %}</ul>
</div>
<div class="card"><h2>Codebase Intelligence</h2>
<ul>{% for c in d.code_insights %}<li><b>{{c.repo}}</b> · files changed: {{c.files_changed}}<br/>Next step: {{c.next_step}}<br/><span class="small">why: {{c.why}}</span></li>{% endfor %}</ul></div>
<div class="card"><h2>Nudges</h2><ul>{% for n in d.nudges %}<li>{{n.tip}}<br/><span class="small">why: {{n.why}}</span></li>{% endfor %}</ul></div>
<div class="card"><h2>Tomorrow</h2><ul>{% for t in d.tomorrow %}<li>{{t}}</li>{% endfor %}</ul></div>
<div class="small">Private, local. Engine: {{engine}}.</div>
</body></html>"""

def build_report(d: Dict[str,Any]):
    today = datetime.date.today().strftime("%A, %B %d, %Y")
    html = Template(HTML).render(d=d, today=today, engine=ENGINE)
    with open(REPORT_PATH, "w") as f: f.write(html)

def generate_daily():
    conn = db()
    ctx = build_day_context(conn)
    engine = get_engine()
    result = plan_and_critique(engine, ctx)
    build_report(result)
    # persist facts for tomorrow-aware prompts later
    day = datetime.date.today().isoformat()
    conn.execute("INSERT OR REPLACE INTO facts(day, json) VALUES(?,?)", (day, json.dumps(result)))
    conn.commit()
    notify("TimePilot", "Your daily update is ready", "Click to view")
    subprocess.run(["open", REPORT_PATH])

# ---------- Scheduler ----------
def main():
    # Start lightweight tracker in a child process
    if os.fork() == 0:
        track_loop()
        return
    # Nightly run at 21:00 local time
    schedule.every().day.at("21:00").do(generate_daily)
    print("TimePilot running. Local, private. 9pm daily update.")
    while True:
        schedule.run_pending()
        time.sleep(2)

if __name__ == "__main__":
    main()
Run it

bash
Copy
python main.py
# You may be prompted to grant Accessibility and Screen Recording so AppleScript can read frontmost window titles.
Swap engines in seconds
Local default (llama.cpp + Metal):
ENGINE=llamacpp GGUF_PATH="/path/to/Qwen2.5-1.5B-Instruct-Q6_K_L.gguf" python main.py
(llama‑cpp offers an OpenAI‑compatible server too if you ever want to route tools behind a local API.) 
GitHub

Local (MLX on Apple):
ENGINE=mlx MLX_REPO_ID="(your MLX Qwen checkpoint id)" python main.py
(MLX‑LM is Apple Silicon‑native; Qwen has MLX checkpoints.) 
GitHub
qwen.readthedocs.io

Optional cloud (fast polish pass):
ENGINE=openai OPENAI_API_KEY=sk-... OPENAI_MODEL=gpt-5-nano python main.py
(gpt‑5‑nano is a new tiny, fast version in the GPT‑5 family.) 
OpenAI Platform
OpenAI

How it stays transparent and private
Local‑first engines; no network calls unless you flip ENGINE=openai.

Each suggestion includes why with concrete evidence (“you spent 58m in Safari; last commit touched 3 files in sleep_model/”).

Connectors are opt‑in. You can add iMessage/Slack/Gmail later, but default MVP ships with activity + git diffs only.

How this borrows PC‑Agent’s superpowers (without the bloat)
PC Tracker → ours: a tiny, permissive tracker for frontmost app + durations (no keystrokes). The official PC‑Agent tracker today ships Windows build steps; we keep a Mac‑native path. 
GitHub

Cognition completion: we run a two‑pass plan then critic to turn raw events into “cognitive trajectories” and decisions. 
gair-nlp.github.io

Grounding: every item shows the raw evidence (“why”). Think “explanations by default.”

Auto‑start it at login (optional)
Create ~/Library/LaunchAgents/com.timepilot.daily.plist:

xml
Copy
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.timepilot.daily</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>-lc</string><string>cd ~/TimePilot && source .venv/bin/activate && python main.py</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict></plist>
Load it:

bash
Copy
launchctl load ~/Library/LaunchAgents/com.timepilot.daily.plist
Scale testing on big GPUs (when you want to go nuts)
vLLM serving for H100 or multi‑GPU, tensor/pipeline parallel. Easy to deploy on a GKE node pool; swap your ENGINE to hit that HTTP endpoint while the Mac stays the UX hub. 
VLLM Documentation
Google Cloud

NVIDIA 5090: 32 GB GDDR7 Blackwell—great for 7–14B quants at silly throughput if you want a home rig. 
NVIDIA

Iteration plan (1 week to “founder‑cred”)
Day 1

Run the MVP. Confirm events flowing and 9pm report renders.

Hard‑code 3 repos under ~/Projects and verify Codebase Intelligence.

Day 2–3

Add “Focus observations” heuristics (e.g., “>60 min social apps → propose 25‑min cap”).

Add “Tomorrow scaffolding” template that learns your cadence (3 tasks max).

Day 4

Add a repo memory file (.timepilot.json) where the agent writes brief design notes per repo every night. Feels like “it updates its knowledge on your codebase.”

Day 5–6

Add a “Conversations” connector (optional): you already know how to read chat.db; add a redaction pass + on‑device summarization.

Ship a menu‑bar icon that opens today’s report (SwiftUI or swiftbar).

Day 7

A/B test engines: Qwen GGUF Q6_K_L vs MLX 4‑bit vs gpt‑5‑nano polish pass. Measure latency and quality.

Why this gives you founder confidence
It’s show, don’t tell. People will finally see the idea.

It’s aligned to your constraints: Linux/AI chops, athlete discipline, privacy obsession, and “AI that’s not in your face.”

It’s agentic without the black box. The “why” column kills doubt.

If you want, I can:

generate a tiny repo scaffold with this main.py, a README, and a Makefile;

add a pc_agent adapter module so you can drop in their planner later;

wire a 60‑second benchmark script for GGUF Q6 vs MLX 4‑bit on your M‑series.

You’re not “waiting for 2027.” You’re building it tonight. 🚀

Sources you might want to skim later

PC‑Agent overview and code (tracker, cognition completion, multi‑agent planning/grounding). 
gair-nlp.github.io
GitHub

llama‑cpp‑python on macOS Metal and OpenAI‑compatible local server. 
llama-cpp-python.readthedocs.io
GitHub

Qwen 2.5 1.5B Instruct GGUF quantized models (fast local). MLX options for Apple Silicon. 
Hugging Face
+1
GitHub
qwen.readthedocs.io

GPT‑5 family; gpt‑5‑nano in the API (optional polish). 
OpenAI Platform
+1
OpenAI

vLLM for scaling on H100 / GKE. NVIDIA 5090 specs. 
VLLM Documentation
Google Cloud
NVIDIA

Want me to tailor the heuristics to your exact day (diving practice → gym → study blocks) and add a sleep‑research angle to the nudges?
