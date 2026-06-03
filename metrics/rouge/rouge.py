# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" ROUGE metric (Pure Python Standalone Version). """

import datasets
import evaluate
from collections import Counter

_CITATION = """
@inproceedings{lin-2004-rouge,
    title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
    author = "Lin, Chin-Yew",
    booktitle = "Text Summarization Branches Out",
    year = "2004",
    publisher = "Association for Computational Linguistics",
    pages = "74--81",
}
"""

_DESCRIPTION = """
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics and a software package used for evaluating automatic summarization and machine translation software in natural language processing.
"""

_KWARGS_DESCRIPTION = """
Calculates ROUGE scores.
Args:
    predictions: list of predictions to score.
    references: list of references for each prediction.
    rouge_types: list of ROUGE types to calculate (e.g. ["rouge1", "rouge2", "rougeL"]).
Returns:
    scores: dict containing the scores.
"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Rouge(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[],
            reference_urls=["https://en.wikipedia.org/wiki/ROUGE_(metric)"],
        )

    def _compute(self, predictions, references, rouge_types=None, **kwargs):
        if rouge_types is None:
            rouge_types = ["rouge1", "rouge2", "rougeL"]

        score_list = {t: [] for t in rouge_types}

        for pred, ref in zip(predictions, references):
            # 简单分词：默认输入已经是空格分隔的字符串
            # 例如: "战 马" -> ["战", "马"]
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            if "rouge1" in rouge_types:
                score_list["rouge1"].append(self._compute_rouge_n(pred_tokens, ref_tokens, n=1))
            if "rouge2" in rouge_types:
                score_list["rouge2"].append(self._compute_rouge_n(pred_tokens, ref_tokens, n=2))
            if "rougeL" in rouge_types:
                score_list["rougeL"].append(self._compute_rouge_l(pred_tokens, ref_tokens))

        # 计算平均分
        final_scores = {}
        for t in rouge_types:
            if len(score_list[t]) > 0:
                final_scores[t] = sum(score_list[t]) / len(score_list[t])
            else:
                final_scores[t] = 0.0
        
        return final_scores

    def _get_ngrams(self, tokens, n):
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _compute_rouge_n(self, pred_tokens, ref_tokens, n):
        """计算 ROUGE-N F1 Score"""
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
            
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        pred_counts = Counter(pred_ngrams)
        ref_counts = Counter(ref_ngrams)
        
        overlap = 0
        for ngram, count in pred_counts.items():
            overlap += min(count, ref_counts[ngram])
            
        if overlap == 0: return 0.0
        
        precision = overlap / len(pred_ngrams)
        recall = overlap / len(ref_ngrams)
        
        if precision + recall == 0: return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _compute_rouge_l(self, pred_tokens, ref_tokens):
        """计算 ROUGE-L F1 Score (基于最长公共子序列)"""
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0

        # 简单的 LCS 动态规划实现
        m, n = len(pred_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        lcs_len = dp[m][n]
        
        if lcs_len == 0: return 0.0
        
        precision = lcs_len / m
        recall = lcs_len / n
        
        if precision + recall == 0: return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1