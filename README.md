# BayesBench Notebook

A self-contained Jupyter notebook demo of **BayesBench** — Bayesian early stopping
for online A/B tests — framed as an **MLOps recommender-system rollout decision**.

> Stop the test once the posterior evidence is already strong enough.

## The scenario

You run a movie streaming service. Production currently serves a **popularity-based
recommender**. Your team built a **candidate recommender** — an implicit-feedback ALS
matrix factorization model (the same algorithm family behind `implicit` and Spotify's
early recommender) — and wants to roll it out. Rather than committing to a fixed-size
A/B test, [`bayesbench_demo.ipynb`](bayesbench_demo.ipynb) runs a **Bayesian sequential
test**: it re-checks the evidence after every simulated user session and stops as soon
as one model is decisively better, so fewer users get exposed to the losing model and
the ship/no-ship call comes faster.

Using real [MovieLens ratings](https://huggingface.co/datasets/ashraq/movielens_ratings),
the notebook:

1. Splits users into an **offline training pool** (fits both models, like a nightly
   batch job) and a **live-traffic pool** (simulates incoming sessions).
2. Fits the **production model** (global popularity) and the **candidate model**
   (implicit-ALS matrix factorization, trained by alternating least squares — an
   actual optimized model, not a similarity heuristic) on the training pool.
3. Scores each live session with **Hit Rate@10** — did the model's top-10 list
   contain the movie the user actually rated highly? A clean 0/1 outcome per session.
4. **Folds each session's outcome back into the training pool and periodically
   retrains both models** — a continuous-training loop, and then directly measures
   (rather than just asserts) whether that retraining changes the outcome.
5. Runs a Beta-Bernoulli Bayesian sequential test over the session stream and stops
   as soon as the winner is decisive.

The notebook is also honest about a real risk with naive sequential testing:
while tuning the candidate model, some hyperparameter settings produced an early
confident call that pointed the *wrong* direction relative to the full-sample
result — a known failure mode when a stopping rule has no minimum-sample floor.

Everything runs on CPU in under a minute — no GPU, no API keys.

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
