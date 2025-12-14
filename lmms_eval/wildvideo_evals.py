# lmms_eval/tasks/wildvideo/wildvideo_evals.py

import json
import random
import time
from typing import Any, Dict, List, Tuple

import requests


class WildVideoEvaluator:
    def __init__(self, sys_prompt: str, api_key: str, api_url: str, model_name: str):
        self.sys_prompt = sys_prompt
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name

    def build_prompt(self, item: Dict[str, Any]) -> str:
        question = item.get("question", "")
        gold = item.get("answer", "")
        pred = item.get("prediction", "")

        base_prompt = """
You will receive a video question, the ground-truth answer, and the prediction
from a video question answering model.
Your task is to decide whether the prediction can be regarded as correct,
based on the question and the ground-truth answer.

Consider semantic equivalence and factual consistency.
Minor wording differences are allowed.
If the prediction is correct, respond with exactly "Correct".
If the prediction is incorrect, respond with exactly "Incorrect".
Do not output anything else.
""".strip()

        q_part = f"Question:\n{question}\n\n" if question else ""
        prompt = (
            f"{base_prompt}\n\n"
            f"{q_part}"
            f"Ground-Truth Answer:\n{gold}\n\n"
            f"Model Prediction:\n{pred}\n"
        )
        return prompt

    def _call_judge_model_once(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model_name,
            "temperature": 0.0,
            "top_p": 1.0,
            "messages": [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        resp = requests.post(
            self.api_url,
            headers=headers,
            json=data,
            timeout=30,
        )
        resp.raise_for_status()

        out = resp.json()
        try:
            return out["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Bad response format from judge model: {out}") from e

    def _call_judge_model_with_retry(self, prompt: str, maxtry: int = 2) -> str:
        last_err: Exception | None = None

        for i in range(maxtry):
            try:
                return self._call_judge_model_once(prompt)
            except Exception as e:
                last_err = e
                print(
                    f"[WildVideo judge] request failed (try {i+1}/{maxtry}): {e}"
                )
                time.sleep(0.5)

        raise RuntimeError(f"Judge model failed after {maxtry} tries: {last_err}")

    @staticmethod
    def _output_to_score(text: str) -> int:
        t = text.strip()
        if t.startswith("Correct"):
            return 1
        if t.startswith("Incorrect"):
            return 0
        if ("Correct" in t) and ("Incorrect" not in t):
            return 1
        if "Incorrect" in t:
            return 0
        return 0

    def eval_result(
        self, results: List[Dict[str, Any]], eval_method: str = "model"
    ) -> Tuple[float, Dict[str, Any]]:

        total = 0
        correct = 0
        failed = 0

        type_total: Dict[str, int] = {}
        type_correct: Dict[str, int] = {}
        type_failed: Dict[str, int] = {}

        for idx, item in enumerate(results):
            j = item.get("judge_input")
            if j is None:
                print(f"[WildVideo judge] sample {idx} has no judge_input, skip.")
                continue

            q_type = j.get("type", "Unknown")

            prompt = self.build_prompt(j)

            try:
                raw = self._call_judge_model_with_retry(prompt)
                score = self._output_to_score(raw)
            except Exception as e:
                print(
                    f"[WildVideo judge] sample {idx} FAILED, "
                    f"treat as incorrect. Error = {e}"
                )
                failed += 1
                total += 1

                type_total[q_type] = type_total.get(q_type, 0) + 1
                type_failed[q_type] = type_failed.get(q_type, 0) + 1
                continue

            total += 1
            correct += score

            type_total[q_type] = type_total.get(q_type, 0) + 1
            if score == 1:
                type_correct[q_type] = type_correct.get(q_type, 0) + 1

            time.sleep(0.1)

        overall_acc = correct / total if total > 0 else 0.0

        per_type_acc = {
            t: (type_correct.get(t, 0) / type_total[t]) if type_total[t] > 0 else 0.0
            for t in type_total
        }

        per_type_detail = {
            t: {
                "total": type_total.get(t, 0),
                "correct": type_correct.get(t, 0),
                "failed": type_failed.get(t, 0),
            }
            for t in type_total
        }

        extra_stats: Dict[str, Any] = {
            "total_judged": total,
            "correct_judged": correct,
            "failed_judged": failed,
            "acc_raw": overall_acc,
            "per_type_acc": per_type_acc,
            "per_type_detail": per_type_detail,
        }

        print(
            f"[WildVideo judge] total={total}, correct={correct}, "
            f"failed={failed}, overall_acc={overall_acc:.4f}"
        )
        print("[WildVideo judge] per-type accuracy:")
        for t, acc in per_type_acc.items():
            print(
                f"  {t}: {acc:.4f} "
                f"(correct={type_correct.get(t, 0)}, "
                f"total={type_total.get(t, 0)}, "
                f"failed={type_failed.get(t, 0)})"
            )

        return overall_acc, extra_stats
