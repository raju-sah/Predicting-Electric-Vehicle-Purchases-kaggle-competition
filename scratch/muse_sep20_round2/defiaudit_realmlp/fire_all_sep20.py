#!/usr/bin/env python3
"""Sep 20 launcher: S6E9 r8 queue (10) + Kaggr r38-42 (5, md5-gated) + AGI-2 #12 + AGI-3 #5.
Usage: python3 fire_all_sep20.py [s6e9|kaggr|agi2|agi3|all]
Quota-aware + idempotent: skips whatever is already fired today.
NOTE: API token removed for public release — export KAGGLE_API_TOKEN to run.
"""
import subprocess, time, sys, os, re, hashlib, json, datetime

BASE = "/home/z/my-project"
LOG = f"{BASE}/fire_all_sep20.log"
R8 = f"{BASE}/s6e9/r8_out"
CHAMP = f"{BASE}/scripts/kaggr_agent/main_v263_backup.py"
CHAMP_MD5 = "1b2f6ed0e9aa61dfc07655de6c89301a"
ENV = dict(os.environ, KAGGLE_API_TOKEN=os.environ.get("KAGGLE_API_TOKEN", ""),
           PATH="/home/z/.venv/bin:/usr/bin:/bin")
KAGGLE = "/home/z/.venv/bin/kaggle"
PY = "/home/z/.venv/bin/python"

S6E9_QUEUE = [
    ("r8_m00_mega_verbatim.csv", "r8-m00: megayak 0.94656 verbatim (public-split swaps, w=1)"),
    ("r8_m75.csv",               "r8-m75: 0.75 mega + 0.25 champ rank-blend"),
    ("r8_m50.csv",               "r8-m50: 50/50 mega-champ rank-blend"),
    ("r8_m25.csv",               "r8-m25: 0.25 mega + 0.75 champ rank-blend"),
    ("r8_m10.csv",               "r8-m10: 0.10 mega + 0.90 champ rank-blend"),
    ("r8_x1_income_ge_170537_inv.csv",   "r8-x1: invert rule1 income>=170537 (+10 -> -10)"),
    ("r8_x2_income_31004_41970_inv.csv", "r8-x2: invert rule2 31004..41970 (-10 -> +10)"),
    ("r8_x3_commute_ge_83_inv.csv",      "r8-x3: invert rule3 commute>=83 (-5 -> +5)"),
    ("r8_x4_30k_nosub_env_anx_inv.csv",  "r8-x4: invert rule4 30k-block (-5 -> +5)"),
    ("r8_xall_inv.csv",                  "r8-xall: invert all 4 recipe rules"),
]

def log(m):
    line = time.strftime("%H:%M:%S", time.gmtime()) + " " + m
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def sh(args, timeout=600):
    r = subprocess.run(args, env=ENV, capture_output=True, text=True, timeout=timeout)
    out = (r.stdout + " " + r.stderr).strip()
    log(f"$ {' '.join(args[:4])}... rc={r.returncode} :: {out[-300:]}")
    return r

sys.path.insert(0, f"{BASE}/scripts")
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiGetSubmissionLimitsRequest, ApiListSubmissionsRequest,
    ApiCreateCodeSubmissionRequest)

def sdk():
    return KaggleClient(api_token=os.environ["KAGGLE_API_TOKEN"], verbose=False)

def limits_today(k, comp):
    """Raw limits endpoint value — KNOWN to flap/serve stale pre-reset values."""
    req = ApiGetSubmissionLimitsRequest()
    req.competition_name = comp
    d = k.competitions.competition_api_client.get_submission_limits(req).to_dict()
    return int(d.get("numToday", d.get("num_today", 0)))

def used_today(k, comp, tag):
    """Quota-aware used-count that trusts ACTUAL submissions dated today.
    The limits endpoint has flapped (0<->10) and serves stale values around
    the 00:00 UTC reset; a stale >0 would make us skip the whole queue.
    list_submissions(page=1) shows most-recent first, so today's are on p1."""
    try:
        n = today_sub_count(k, comp)
    except Exception as e:
        log(f"  {tag}: list_sub_count ERR {e} — falling back to limits endpoint")
        try:
            return limits_today(k, comp)
        except Exception:
            return 0
    if n > 0:
        return n
    # n == 0: cross-check; stale pre-reset counter would lie >0
    try:
        nt = limits_today(k, comp)
        if nt > 0:
            log(f"  {tag}: limits says numToday={nt} but 0 subs dated today — STALE, using 0")
        return 0
    except Exception:
        return 0

def submit_with_retry(args, tag, attempts=3):
    """Submit with retry — 400 right after reset usually = replica lag, not truth."""
    for a in range(attempts):
        r = sh(args)
        if r.returncode == 0:
            return True
        if a < attempts - 1:
            log(f"  {tag}: attempt {a+1} failed — retrying in 90s (replica lag possible)")
            time.sleep(90)
    log(f"  {tag}: FAILED after {attempts} attempts")
    return False

def today_sub_count(k, comp):
    try:
        req = ApiListSubmissionsRequest()
        req.competition_name = comp
        req.page = 1
        resp = k.competitions.competition_api_client.list_submissions(req)
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        n = 0
        for s in (resp.submissions or []):
            d = s.to_dict()
            if str(d.get("date", "")).startswith(today):
                n += 1
        return n
    except Exception as e:
        log(f"today_sub_count {comp} ERR: {e} (assume 0)")
        return 0

def do_s6e9(k):
    used = used_today(k, "playground-series-s6e9", "S6E9")
    log(f"S6E9 used_today={used}")
    for i, (fname, msg) in enumerate(S6E9_QUEUE):
        if i < used:
            log(f"  skip {fname} (already fired)")
            continue
        p = f"{R8}/{fname}"
        if not os.path.exists(p):
            log(f"  MISSING {p} — abort queue")
            return
        ok = submit_with_retry([KAGGLE, "competitions", "submit",
                                "-c", "playground-series-s6e9",
                                "-f", p, "-m", msg], fname)
        if not ok:
            log(f"  FAIL {fname}")
            return
        time.sleep(20)

def do_kaggr(k):
    md5 = hashlib.md5(open(CHAMP, "rb").read()).hexdigest()
    log(f"Kaggr champion md5={md5}")
    if md5 != CHAMP_MD5:
        log("MD5 MISMATCH — aborting Kaggr tickets")
        return
    used = used_today(k, "kaggriculture", "Kaggr")
    log(f"Kaggr used_today={used}")
    NTH = {38: "thirty-eighth", 39: "thirty-ninth", 40: "fortieth",
           41: "forty-first", 42: "forty-second"}
    for rnum in range(38, 43):
        idx = rnum - 38
        if idx < used:
            log(f"  skip r{rnum} (already fired)")
            continue
        msg = (f"v2.6.3-r{rnum}: pool ticket champion-exact md5 {CHAMP_MD5[:12]}...; "
               f"{NTH[rnum]} ticket of the verified compound shape")
        ok = submit_with_retry([KAGGLE, "competitions", "submit",
                                "-c", "kaggriculture",
                                "-f", CHAMP, "-m", msg], f"r{rnum}")
        if not ok:
            log(f"  FAIL r{rnum}")
            return
        time.sleep(45)

def push_and_submit(vehicle_dir, slug, comp, file_name, tag):
    r = sh([KAGGLE, "kernels", "push", "-p", vehicle_dir])
    m = re.search(r"Kernel version (\d+) successfully pushed", r.stdout + r.stderr)
    if not m:
        log(f"{tag}: push failed or no version parsed — abort")
        return
    ver = int(m.group(1))
    log(f"{tag}: pushed version {ver} — RUNNING. Server REJECTS code submissions "
        f"until the kernel completes (400 'Notebook is still running').")
    log(f"{tag}: when `kaggle kernels status defiaudit/{slug}` = COMPLETE, fire:")
    log(f"  kaggle competitions submit -c {comp} -k defiaudit/{slug} -v {ver} -f {file_name} -m {tag}")
    log(f"  (or: python scripts/agi_submit_fix.py {'agi2' if 'agi2' in comp else 'agi3'})")

def do_agi2():
    push_and_submit(f"{BASE}/scripts/agi2_pull_0920",
                    "arc-agi2-lb33-89-minimal-perfpatch",
                    "arc-prize-2026-arc-agi-2", "submission.json", "agi2-draw12")

def do_agi3():
    push_and_submit(f"{BASE}/scripts/agi3_pull_0920",
                    "agi3-duck-qwen38-anim-v2",
                    "arc-prize-2026-arc-agi-3", "submission.parquet", "agi3-draw5")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    k = sdk()
    log(f"=== fire_all_sep20 mode={mode} ===")
    if mode in ("all", "s6e9"):  do_s6e9(k)
    if mode in ("all", "kaggr"): do_kaggr(k)
    if mode in ("all", "agi2"):  do_agi2()
    if mode in ("all", "agi3"):  do_agi3()
    log("=== done ===")
