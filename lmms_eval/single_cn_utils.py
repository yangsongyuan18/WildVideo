# ================== Single-CN ==================

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List
import yaml

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks.wildvideo.wildvideo_evals import WildVideoEvaluator

VIDEO_ROOT = "/home/yangsongyuan/project/WildVideo/wildvideo/video"

JUDGE_MODEL_NAME = os.getenv("MODEL_VERSION", "gpt-4o-mini")
API_TYPE = os.getenv("API_TYPE", "openai")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
else:
    API_URL = "YOUR_API_URL"
    API_KEY = "YOUR_API_KEY"

with open(Path(__file__).parent / "wildvideo_single_en.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
    config = yaml.safe_load("".join(safe_data))

sys_prompt = config.get("metadata", {}).get("sys_prompt", "You are an automatic evaluator for WildVideo.")

evaluator = WildVideoEvaluator(
    sys_prompt=sys_prompt,
    api_key=API_KEY,
    api_url=API_URL,
    model_name=JUDGE_MODEL_NAME,
)


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,:;!?\"'“”‘’()[]{}")
    return s

def wildvideo_doc_to_visual(doc: Dict[str, Any]) -> List[str]:
    video_id = doc.get("video_id")
    video_path = os.path.join(VIDEO_ROOT, f"{video_id}.mp4")
    return [video_path]

def wildvideo_single_cn_doc_to_text(doc: Dict[str, Any]) -> str:

    question = doc.get("question", "")
    prompt = (
        "请你根据给定的视频内容，尽可能准确、简洁地回答下面的问题。\n\n"
        f"问题：{question}\n"
        "回答："
    )
    return prompt


def wildvideo_single_cn_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:

    if doc.get("lang", "cn") != "cn":
        return {}

    pred = (results[0] if results else "").strip()

    judge_input = {
        "video_id": doc.get("video_id"),
        "question": doc.get("question", ""),
        "answer": doc.get("answer", ""),
        "prediction": pred,
        "lang": doc.get("lang", "cn"),
        "turn_type": doc.get("turn_type", "single"),
        "type": doc.get("type", None),
    }

    return {"wildvideo_single_cn_acc": judge_input}


def wildvideo_single_cn_aggregate(results, args):

    print("============= WildVideo Single-CN (judge model only) =============")

    wrapped_results = [{"judge_input": j} for j in results]

    overall_acc, extra_stats = evaluator.eval_result(
        wrapped_results, eval_method="model"
    )

    out_file = generate_submission_file("wildvideo_single_cn_results.json", args)
    with open(out_file, "w") as f:
        json.dump(
            {
                "overall_acc": overall_acc,
                "extra_stats": extra_stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return overall_acc