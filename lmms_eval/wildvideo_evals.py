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

        for idx, item in enumerate(results):
            j = item.get("judge_input")
            if j is None:
                print(f"[WildVideo judge] sample {idx} has no judge_input, skip.")
                continue

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
                continue

            total += 1
            correct += score
            time.sleep(0.1)

        overall_acc = correct / total if total > 0 else 0.0
        extra_stats: Dict[str, Any] = {
            "total_judged": total,
            "correct_judged": correct,
            "failed_judged": failed,
            "acc_raw": overall_acc,
        }

        print(
            f"[WildVideo judge] total={total}, correct={correct}, "
            f"failed={failed}, overall_acc={overall_acc:.4f}"
        )

        return overall_acc, extra_stats
