"""Poll GitHub Actions CI until the current commit passes."""
import subprocess, json, time, sys, os

REPO = "alpaca-C/Eduguide-Agent"
BRANCH = "test/add-unit-tests-p2"
SLEEP = 90  # seconds between checks
MAX_ATTEMPTS = 30

def api(path):
    r = subprocess.run(
        ["curl", "-s", f"https://api.github.com/repos/{REPO}/{path}"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return json.loads(r.stdout) if r.stdout.strip() else {}

def get_latest_run():
    data = api(f"actions/runs?branch={BRANCH}&per_page=1&status=completed")
    runs = data.get("workflow_runs", [])
    return runs[0] if runs else None

def get_current_commit():
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()[:7]

for i in range(1, MAX_ATTEMPTS + 1):
    run = get_latest_run()
    if not run:
        print(f"[{i}] No completed run yet...")
        time.sleep(SLEEP)
        continue

    cid = run.get("head_sha", "")[:7]
    conclusion = run.get("conclusion")
    html = run.get("html_url", "")
    cur = get_current_commit()

    if cid == cur:
        if conclusion == "success":
            print(f"[{i}] PASSED! {html}")
            sys.exit(0)
        elif conclusion == "failure":
            print(f"[{i}] FAILED: {html}")
            sys.exit(1)
        else:
            print(f"[{i}] {conclusion}: {html}")
    else:
        print(f"[{i}] Run {cid} != local {cur}, waiting for new run...")

    time.sleep(SLEEP)

print("Max attempts reached")
sys.exit(1)
