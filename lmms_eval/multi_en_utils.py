# lmms_eval/tasks/wildvideo/multi_en_utils.py

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

_CONFIG_PATH = Path(__file__).parent / "wildvideo_multi_en.yaml"
if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH, "r") as f:
        raw_data = f.readlines()
        safe_data = [line for line in raw_data if "!function" not in line]
        config = yaml.safe_load("".join(safe_data)) or {}
else:
    config = {}

sys_prompt = config.get(
    "metadata", {}
).get("sys_prompt", "You are an automatic evaluator for WildVideo multi-turn English QA.")

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


def _build_multiturn_prompt_en(rounds: List[Dict[str, Any]]) -> str:

    if not rounds:
        return "Answer the question based on the given video."

    rounds_sorted = sorted(rounds, key=lambda r: r.get("round", 0))
    history = rounds_sorted[:-1]
    last = rounds_sorted[-1]

    parts: List[str] = [
        "Below is the previous conversation and the current question. "
        "Please answer the final question based on the video.\n"
    ]

    for r in history:
        rid = r.get("round", "")
        q = r.get("question", "")
        a = r.get("answer", "")
        parts.append(f"[Turn {rid}] Q: {q}\n[Turn {rid}] A: {a}\n")

    parts.append(
        f"Current question (Turn {last.get('round', '')}): {last.get('question', '')}\n"
        "Answer:"
    )

    return "\n".join(parts)


def wildvideo_multi_en_doc_to_text(doc: Dict[str, Any]) -> str:
    """
    多轮版 doc_to_text：
    - 优先读取 doc['rounds']
    - 用 _build_multiturn_prompt_en 拼 prompt
    - 如果没有 rounds，就退化成单轮样式
    """
    rounds = doc.get("rounds") or []

    if rounds:
        return _build_multiturn_prompt_en(rounds)

    question = doc.get("question", "")
    prompt = (
        "Answer the question based on the given video as concisely and accurately as possible.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt


# 如果你想让 doc_to_target 也能自动从最后一轮取 answer，可以加这个：
def wildvideo_multi_en_doc_to_target(doc: Dict[str, Any]) -> str:
    """
    多轮的“标准答案”：默认用最后一轮的 answer。
    方便在 YAML 里写：doc_to_target: !function multi_en_utils.wildvideo_multi_en_doc_to_target
    """
    if "answer" in doc and doc["answer"]:
        return str(doc["answer"])

    rounds = doc.get("rounds") or []
    if not rounds:
        return ""

    rounds_sorted = sorted(rounds, key=lambda r: r.get("round", 0))
    last = rounds_sorted[-1]
    return str(last.get("answer", "")).strip()


def wildvideo_multi_en_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:

    if doc.get("lang", "en") != "en":
        return {}
    if doc.get("turn_type", "multi") != "multi":
        return {}

    pred = (results[0] if results else "").strip()

    rounds = doc.get("rounds") or []

    if rounds:
        rounds_sorted = sorted(rounds, key=lambda r: r.get("round", 0))
        last = rounds_sorted[-1]

        final_question = last.get("question", doc.get("question", ""))
        gold_answer = last.get("answer", doc.get("answer", ""))

        q_type = last.get("type", doc.get("type", "Unknown"))
    else:
        final_question = doc.get("question", "")
        gold_answer = doc.get("answer", "")
        q_type = doc.get("type", "Unknown")

    judge_input = {
        "video_id": doc.get("video_id"),
        "question": final_question,
        "answer": gold_answer,
        "prediction": pred,
        "lang": doc.get("lang", "en"),
        "turn_type": doc.get("turn_type", "multi"),
        "type": q_type,
    }

    return {"wildvideo_multi_en_acc": judge_input}


def wildvideo_multi_en_aggregate(results, args):

    print("============= WildVideo Multi-EN (judge model only) =============")

    wrapped_results = [{"judge_input": j} for j in results]

    overall_acc, extra_stats = evaluator.eval_result(
        wrapped_results, eval_method="model"
    )

    out_file = generate_submission_file("wildvideo_multi_en_results.json", args)
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
