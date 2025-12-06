# lmms_eval/tasks/wildvideo/multi_cn_utils.py

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
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.plus/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
else:
    API_URL = "YOUR_API_URL"
    API_KEY = "YOUR_API_KEY"

with open(Path(__file__).parent / "wildvideo_multi_cn.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
    config = yaml.safe_load("".join(safe_data))

sys_prompt = config.get(
    "metadata",
    {},
).get(
    "sys_prompt",
    "你是 WildVideo 的自动评测器，会判断视频问答模型的回答是否可以视为正确。",
)

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
    """
    多轮 / 单轮通用：根据 video_id 拼出本地 mp4 路径。
    """
    video_id = doc.get("video_id")
    video_path = os.path.join(VIDEO_ROOT, f"{video_id}.mp4")
    return [video_path]



def _build_multiturn_prompt_cn(rounds: List[Dict[str, Any]]) -> str:
    if not rounds:
        return "请根据视频内容回答问题：\n回答："

    rounds_sorted = sorted(rounds, key=lambda r: r.get("round", 0))
    history = rounds_sorted[:-1]
    last = rounds_sorted[-1]

    parts: List[str] = []
    parts.append(
        "下面是关于同一个视频的多轮问答记录。"
        "请结合「视频内容」和「历史对话」，只回答最后一个问题。\n"
    )

    for r in history:
        rid = r.get("round", "")
        q = r.get("question", "")
        a = r.get("answer", "")
        parts.append(f"[轮次 {rid}] 问：{q}")
        parts.append(f"[轮次 {rid}] 答：{a}\n")

    last_id = last.get("round", "")
    last_q = last.get("question", "")
    parts.append(f"当前问题（轮次 {last_id}）：{last_q}")
    parts.append("请直接用中文回答，不要复述问题。\n回答：")

    return "\n".join(parts)


def wildvideo_multi_cn_doc_to_text(doc: Dict[str, Any]) -> str:
    """
    给 llava_vid 用的多轮 CN prompt。
    """
    rounds = doc.get("rounds", []) or []
    return _build_multiturn_prompt_cn(rounds)



def wildvideo_multi_cn_process_results(
    doc: Dict[str, Any],
    results: List[str],
) -> Dict[str, Any]:

    if doc.get("lang", "cn") != "cn":
        return {}
    if doc.get("turn_type", "multi") != "multi":
        return {}

    rounds = doc.get("rounds", []) or []
    if not rounds:
        return {}

    rounds_sorted = sorted(rounds, key=lambda r: r.get("round", 0))
    last = rounds_sorted[-1]

    pred = (results[0] if results else "").strip()

    judge_input = {
        "video_id": doc.get("video_id"),
        "question": last.get("question", ""),
        "answer": last.get("answer", ""),
        "prediction": pred,
        "lang": doc.get("lang", "cn"),
        "turn_type": doc.get("turn_type", "multi"),
        "type": last.get("type", doc.get("type", None)),
        "path_id": doc.get("path_id", None),
        "round": last.get("round", None),
    }

    return {"wildvideo_multi_cn_acc": judge_input}


def wildvideo_multi_cn_aggregate(results, args):

    print("============= WildVideo Multi-CN (judge model only) =============")

    wrapped_results = [{"judge_input": j} for j in results]

    overall_acc, extra_stats = evaluator.eval_result(
        wrapped_results,
        eval_method="model",
    )

    out_file = generate_submission_file("wildvideo_multi_cn_results.json", args)
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