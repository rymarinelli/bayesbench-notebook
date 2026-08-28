# BayesBench Notebook

A self-contained Jupyter notebook demo of **BayesBench** — Bayesian early stopping
for LLM benchmarking.

> Stop evaluating once the posterior evidence is already strong enough.

Open [`bayesbench_demo.ipynb`](bayesbench_demo.ipynb): it loads a small [BoolQ](https://huggingface.co/datasets/google/boolq)
subset, runs a real local LLM ([SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct))
against a majority-class baseline, and walks through a Beta-Bernoulli Bayesian
sequential test — updating the posterior after every example and stopping as
soon as `P(A > B)` (or `P(B > A)`) crosses a confidence threshold.

Everything runs on CPU with no API keys.

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook bayesbench_demo.ipynb
```

## Related

- [bayesbench-demo](https://github.com/rymarinelli/bayesbench-demo) — the full
  interactive Streamlit app (multi-task suites, model ranking, agent/tool-use
  benchmarks, custom HuggingFace datasets).
- [bayesbench](https://github.com/rymarinelli/bayesbench) — the underlying
  Python package.
