#!/usr/bin/env python3
"""Focused AGI code-submission diagnostic: kernel status + create submission with full errors."""
# NOTE: API token removed for public release — export KAGGLE_API_TOKEN to run.
import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/z/my-project/scripts")
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import ApiCreateCodeSubmissionRequest
try:
    from kagglesdk.kernels.types.kernel_api_service import ApiGetKernelSessionStatusRequest
except ImportError:
    try:
        from kagglesdk.types.kernel_api_service import ApiGetKernelSessionStatusRequest
    except ImportError:
        ApiGetKernelSessionStatusRequest = None

TOK = os.environ["KAGGLE_API_TOKEN"]

def sdk():
    return KaggleClient(api_token=TOK, verbose=False)

def status(k, slug):
    if ApiGetKernelSessionStatusRequest is None:
        return "status API unavailable in this SDK version"
    req = ApiGetKernelSessionStatusRequest()
    req.user_name = "defiaudit"
    req.kernel_slug = slug
    try:
        r = k.kernels.kernels_api_client.get_kernel_session_status(req)
        return r.to_dict()
    except Exception as e:
        return f"STATUS ERR: {str(e)[:200]}"

def submit(k, comp, slug, ver, fname, tag):
    req = ApiCreateCodeSubmissionRequest()
    req.competition_name = comp
    req.kernel_owner = "defiaudit"
    req.kernel_slug = slug
    req.kernel_version = ver
    req.file_name = fname
    req.submission_description = tag
    try:
        resp = k.competitions.competition_api_client.create_code_submission(req)
        d = resp.to_dict()
        print(f"{tag}: OK ref={d.get('ref') or d.get('id')} :: {str(d)[:150]}")
        return True
    except Exception as e:
        print(f"{tag}: ERR {type(e).__name__} :: {str(e)[:400]}")
        return False

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "agi2"
    k = sdk()
    if which == "agi2":
        print("status:", status(k, "arc-agi2-lb33-89-minimal-perfpatch"))
        submit(k, "arc-prize-2026-arc-agi-2", "arc-agi2-lb33-89-minimal-perfpatch", 11,
               "submission.json", "agi2-draw12")
    else:
        print("status:", status(k, "agi3-duck-qwen38-anim-v2"))
        submit(k, "arc-prize-2026-arc-agi-3", "agi3-duck-qwen38-anim-v2", 8,
               "submission.parquet", "agi3-draw5")
