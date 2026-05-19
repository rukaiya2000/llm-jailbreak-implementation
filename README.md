# PAIR: Prompt Automatic Iterative Refinement

Implementation of the PAIR jailbreak algorithm from [Chao et al. (2023)](https://arxiv.org/abs/2310.08419) for academic security research.

PAIR pits three LLMs against each other — an **attacker** that iteratively refines jailbreak prompts, a **target** that responds to them, and a **judge** that scores each attempt from 1–10.

---

## How it works

1. The **attacker LLM** generates a jailbreak prompt using chain-of-thought reasoning (outputs `{"improvement": "...", "prompt": "..."}`)
2. The **target LLM** receives only the prompt and responds
3. The **judge LLM** scores the response 1–10 (1 = refused, 10 = fully jailbroken)
4. If score < 10, the attacker sees the feedback and refines its prompt
5. This repeats for up to K iterations, across N parallel streams

---

## Setup

**Requirements:** Python 3.13+

### 1. Install dependencies

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### 2. Set your API key(s)

Create a `.env` file in the project root. Only the keys for the providers you use are required:

```
NAVIGATOR_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

---

## Providers

Three providers are supported:

| Provider | Value | Notes |
|----------|-------|-------|
| UF AI Navigator | `navigator` | Default — OpenAI-compatible proxy for OSS models |
| Anthropic | `anthropic` | Claude models |
| OpenAI | `openai` | GPT models |

Each of the three roles (attacker, target, judge) can use a different provider and model independently.

---

## Running

### Basic run

```bash
python main.py --goal "your goal here"
```

### With custom streams and iterations

```bash
python main.py --goal "your goal here" --n-streams 5 --k-iterations 10 --verbose
```

### With a specific provider/model

```bash
python main.py --goal "your goal here" \
  --attacker-provider navigator --attacker-model gpt-oss-120b \
  --target-provider navigator --target-model gpt-oss-120b \
  --judge-provider navigator --judge-model gpt-oss-120b
```

---

## All options

| Flag | Default | Description |
|------|---------|-------------|
| `--goal` | *(required)* | Target behavior to elicit from the model |
| `--n-streams` | `3` | Number of parallel conversation streams |
| `--k-iterations` | `2` | Max refinement iterations per stream |
| `--attacker-provider` | `navigator` | Provider for the attacker (`anthropic`, `openai`, `navigator`) |
| `--target-provider` | `navigator` | Provider for the target |
| `--judge-provider` | `navigator` | Provider for the judge |
| `--attacker-model` | `gpt-oss-120b` | Model used to generate jailbreak prompts |
| `--target-model` | `gpt-oss-120b` | Model being attacked |
| `--judge-model` | `gpt-oss-120b` | Model used to score responses |
| `--success-threshold` | `10` | Score >= this counts as a successful jailbreak |
| `--verbose` | `false` | Print per-stream, per-iteration progress |
| `--log-dir` | `logs/` | Directory to save JSON run logs |

---

## Output

Each run prints a summary and the best prompt found along with the target's response, then saves a full log to `logs/run_<timestamp>.json` containing every stream's iteration history (prompts, responses, scores, and improvement reasoning).

---

## Project structure

```
pair/
├── config.py       PAIRConfig dataclass
├── prompts.py      Attacker and judge system prompts (verbatim from paper)
├── models.py       LLM clients for Anthropic, OpenAI, and Navigator
├── attacker.py     Attacker agent — generates and refines jailbreak prompts
├── judge.py        Judge agent — scores prompt-response pairs
└── algorithm.py    Core PAIR loop: N async streams × K iterations
main.py             CLI entry point
```

---

## Reference

> Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G. J., & Wong, E. (2023).
> *Jailbreaking Black Box Large Language Models in Twenty Queries.*
> arXiv:2310.08419
