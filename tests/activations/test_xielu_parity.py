# Usage parity-test:
# RUN_SLOW=1 pytest -s tests/activations/test_xielu_parity.py
#
# Usage throughput-test (be patient):
# RUN_SLOW=1 RUN_1K=1 pytest -s tests/activations/test_xielu_parity.py

import unittest
import pytest
import torch
from transformers.testing_utils import require_torch_gpu, slow, parse_flag_from_env
from transformers import AutoTokenizer
from transformers.models.swissai.configuration_swissai import SwissAIConfig
from transformers.models.swissai.modeling_swissai import SwissAIForCausalLM
import sys
from transformers.activations import ACT2FN, XIELUActivation

XIELU_METRICS = {}

class XIELUParityTest(unittest.TestCase):
    def _lcs(self, X, Y):
        m = len(X)
        n = len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])
        return L[m][n]

    def _calculate_rouge_l(self, output_strs_list1, output_strs_list2):
        rouge_l_scores = []
        for s1, s2 in zip(output_strs_list1, output_strs_list2):
            lcs_len = self._lcs(s1, s2)
            precision = lcs_len / len(s1) if len(s1) > 0 else 0
            recall = lcs_len / len(s2) if len(s2) > 0 else 0
            if precision + recall > 0:
                fmeasure = (2 * precision * recall) / (precision + recall)
            else:
                fmeasure = 0.0
            rouge_l_scores.append(fmeasure)
        return rouge_l_scores

    @torch.no_grad()
    def _benchmark_generation(self, model, inputs, max_new_tokens=20, n_warmup=1, n_runs=5):
        """
        Benchmark generation latency for a given max_new_tokens.
        """
        model.eval()
        for _ in range(n_warmup):
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=None)
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_time.record()
        for _ in range(n_runs):
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=None)
        end_time.record()
        torch.cuda.synchronize()
        return start_time.elapsed_time(end_time) / n_runs

    @pytest.mark.xielu_test
    @require_torch_gpu
    @slow
    def test_xielu_vector_loads_parity_and_speed(self):
        model_id = "/capstor/store/cscs/swissai/infra01/swiss-alignment/checkpoints/Apertus3-8B_iter_1678000-tulu3-sft/checkpoint-13446"
        prompt = "The ETH AI Center is"
        run_1k = parse_flag_from_env("RUN_1K", default=False)

        # Load config and tokenizer
        config = SwissAIConfig.from_pretrained(model_id)
        config.hidden_act = "xielu"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Check CUDA fused xIELU availability
        try:
            import xielu.ops  # noqa: F401
        except ImportError as e:
            pytest.skip(f"CUDA-fused xIELU not available, skipping test. Error: {e}\n"
						"For CUDA xIELU (experimental), `pip install git+https://github.com/nickjbrowning/XIELU`")

        # Load model with vector loads disabled
        ACT2FN["xielu"] = XIELUActivation
        model_nv = SwissAIForCausalLM.from_pretrained(model_id, config=config, torch_dtype=torch.bfloat16).to("cuda")

        # Load model with vector loads enabled
        ACT2FN["xielu"] = (XIELUActivation, {"with_vector_loads": True})
        model_vl = SwissAIForCausalLM.from_pretrained(model_id, config=config, torch_dtype=torch.bfloat16).to("cuda")

        # Compile models if supported by torch
        if hasattr(torch, "compile"):
            model_nv = torch.compile(model_nv)
            model_vl = torch.compile(model_vl)

        # Generate with both models
        with torch.no_grad():
            output_nv = model_nv.generate(
                **inputs, max_new_tokens=20, do_sample=False, output_scores=True, return_dict_in_generate=True, eos_token_id=None
            )
            output_vl = model_vl.generate(
                **inputs, max_new_tokens=20, do_sample=False, output_scores=True, return_dict_in_generate=True, eos_token_id=None
            )
        # Decode outputs
        logits_nv = torch.stack(output_nv.scores)
        logits_vl = torch.stack(output_vl.scores)
        text_nv = tokenizer.decode(output_nv.sequences[0], skip_special_tokens=True)
        text_vl = tokenizer.decode(output_vl.sequences[0], skip_special_tokens=True)

        # Parity check (deferred failure)
        parity_er = None
        try:
            torch.testing.assert_close(logits_vl, logits_nv, atol=1e-3, rtol=1e-3)
            assert text_vl == text_nv
        except AssertionError as e:
            parity_er = e

        # Calculate ROUGE-L
        rouge_score = self._calculate_rouge_l([text_vl], [text_nv])[0]

        # Benchmark generation
        time_nv_20 = self._benchmark_generation(model_nv, inputs, max_new_tokens=20)
        time_vl_20 = self._benchmark_generation(model_vl, inputs, max_new_tokens=20)
        if run_1k:
            time_nv_1k = self._benchmark_generation(model_nv, inputs, max_new_tokens=1024)
            time_vl_1k = self._benchmark_generation(model_vl, inputs, max_new_tokens=1024)
        else:
            time_nv_1k = None
            time_vl_1k = None

        # Update metrics for pytest summary
        XIELU_METRICS.update({
            "Prompt": prompt,
            "Text (no vector loads)": text_nv,
            "Text (with vector loads)": text_vl,
            "ROUGE-L": f"{rouge_score:.4f}",
            "lat_nv_20": time_nv_20,
            "lat_vl_20": time_vl_20,
            "lat_nv_1k": time_nv_1k,
            "lat_vl_1k": time_vl_1k,
        })
        # Final parity failure (marks test as failed after metrics recorded)
        if parity_er:
            pytest.fail(f"Parity checks failed: {parity_er}")
