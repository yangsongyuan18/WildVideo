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
    Your task is to give a **score between 0 and 1** to indicate how correct the prediction is.

    - 1.0 means completely correct.
    - 0.0 means totally wrong.
    - Values in between (e.g., 0.3, 0.5, 0.8) mean partially correct.

    Important:
    - Only output a single NUMBER between 0 and 1 (inclusive), with at most 2 decimal places.
    - Do NOT output any other words or explanation.
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
        """
        轻量级 retry：最多试几次，不成功就交给外层 eval_result 去记为 failed。
        """
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
    def _output_to_score(text: str) -> float:
        """
        把判分模型的输出转成 0~1 区间的小数分数。
        约定：优先认为输出是一个数字（如 "0.8"），
        实在不是数字就 fallback 成 0 或 1 简单判断。
        """
        t = text.strip()

        try:
            val = float(t)

            if val < 0.0:
                val = 0.0
            if val > 1.0:
                val = 1.0
            return val
        except ValueError:
            pass

        m = re.search(r"([01](?:\.\d+)?)", t)
        if m:
            try:
                val = float(m.group(1))
                if val < 0.0:
                    val = 0.0
                if val > 1.0:
                    val = 1.0
                return val
            except ValueError:
                pass

        t_up = t.upper()
        if "CORRECT" in t_up and "INCORRECT" not in t_up:
            return 1.0
        if "INCORRECT" in t_up:
            return 0.0

        return 0.0

    def eval_result(
        self, results: List[Dict[str, Any]], eval_method: str = "model"
    ) -> Tuple[float, Dict[str, Any]]:
        """
        把每个样本的 score（0~1 小数或 0/1）做平均，
        同时按 type 统计 per-type 的平均分。
        """

        total = 0                 
        sum_score = 0.0         
        failed = 0              

        per_type_sum: Dict[str, float] = {}
        per_type_count: Dict[str, int] = {}
        per_type_failed: Dict[str, int] = {}

        for idx, item in enumerate(results):
            j = item.get("judge_input")
            if j is None:
                print(f"[WildVideo judge] sample {idx} has no judge_input, skip.")
                continue

            q_type = j.get("type") or "Unknown"

            prompt = self.build_prompt(j)

            score = 0.0
            ok = True
            try:
                raw = self._call_judge_model_with_retry(prompt)
                score = float(self._output_to_score(raw))
            except Exception as e:
                ok = False
                failed += 1
                print(
                    f"[WildVideo judge] sample {idx} FAILED, "
                    f"treat as score=0. Error = {e}"
                )

            total += 1
            sum_score += score

            per_type_sum[q_type] = per_type_sum.get(q_type, 0.0) + score
            per_type_count[q_type] = per_type_count.get(q_type, 0) + 1
            if not ok:
                per_type_failed[q_type] = per_type_failed.get(q_type, 0) + 1

            time.sleep(0.1)

        overall_acc = sum_score / total if total > 0 else 0.0

        per_type_acc: Dict[str, float] = {}
        per_type_detail: Dict[str, Any] = {}
        for t, s in per_type_sum.items():
            cnt = per_type_count.get(t, 0)
            if cnt == 0:
                continue
            avg = s / cnt
            per_type_acc[t] = avg
            per_type_detail[t] = {
                "total": cnt,
                "failed": per_type_failed.get(t, 0),
                "avg_score": avg,
                "sum_score": s,
            }

        extra_stats: Dict[str, Any] = {
            "total_judged": total,
            "sum_score": sum_score,  
            "failed_judged": failed,
            "acc_raw": overall_acc, 
            "per_type_acc": per_type_acc,
            "per_type_detail": per_type_detail,
        }

        print(
            f"[WildVideo judge] total={total}, sum_score={sum_score:.4f}, "
            f"failed={failed}, overall_acc={overall_acc:.4f}"
        )

        return overall_acc, extra_stats
