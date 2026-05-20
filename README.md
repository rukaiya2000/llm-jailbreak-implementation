# pair-jailbreak

> Implementation of the **PAIR** (Prompt Automatic Iterative Refinement) algorithm for automated LLM red-teaming and adversarial prompt research.

[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Based on](https://img.shields.io/badge/paper-arXiv%3A2310.08419-red)](https://arxiv.org/abs/2310.08419)

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [CLI Reference](#cli-reference)
- [Output](#output)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Ethical Use](#ethical-use)
- [Citation](#citation)
- [License](#license)

---

## Overview

`pair-jailbreak` is a Python implementation of the PAIR algorithm introduced by [Chao et al. (2023)](https://arxiv.org/abs/2310.08419). It orchestrates three LLMs in an adversarial loop — an **attacker**, a **target**, and a **judge** — to automatically discover jailbreak prompts through iterative refinement.

This tool is intended for **academic security research**, **AI safety evaluation**, and **red-teaming** of language models in authorized settings.

---

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                   PAIR Algorithm                    │
│                                                     │
│   ┌──────────┐    prompt    ┌──────────┐            │
│   │ Attacker │ ──────────► │  Target  │            │
│   │   LLM    │             │   LLM    │            │
│   └──────────┘             └──────────┘            │
│        ▲                        │                  │
│        │  feedback + score      │ response         │
│        │                        ▼                  │
│        └──────────────── ┌──────────┐             │
│                           │  Judge   │             │
│                           │   LLM    │             │
│                           └──────────┘             │
│                        scores 1–10                 │
└─────────────────────────────────────────────────────┘
```

1. The **attacker LLM** generates a jailbreak prompt using chain-of-thought reasoning, outputting `{"improvement": "...", "prompt": "..."}`
2. The **target LLM** receives only the prompt and responds
3. The **judge LLM** scores the response from 1 (refused) to 10 (fully jailbroken)
4. If score < threshold, the attacker receives feedback and refines the prompt
5. This repeats for up to **K iterations** across **N parallel streams**

---

## Features

- Fully async — runs N streams concurrently for fast iteration
- Role-independent providers: attacker, target, and judge can each use a different model and provider
- Structured JSON logging of every prompt, response, score, and improvement reasoning
- Configurable success threshold, stream count, and iteration budget
- Supports OpenAI-compatible APIs (UF AI Navigator) and OpenAI directly

---

## Prerequisites

- Python 3.13+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- API key(s) for the provider(s) you intend to use

---

## Installation

### Using uv (recommended)

```bash
git clone https://github.com/rukaiyakhan/pair-jailbreak.git
cd pair-jailbreak
uv sync
```

### Using pip

```bash
git clone https://github.com/rukaiyakhan/pair-jailbreak.git
cd pair-jailbreak
pip install -e .
```

---

## Configuration

Create a `.env` file in the project root. Only include keys for the providers you use:

```env
NAVIGATOR_API_KEY=your_navigator_key_here
OPENAI_API_KEY=your_openai_key_here
```

### Supported Providers

| Provider | Value | Notes |
|----------|-------|-------|
| UF AI Navigator | `navigator` | Default — OpenAI-compatible proxy for open-source models |
| OpenAI | `openai` | GPT model family |

Each of the three roles (attacker, target, judge) can use a different provider and model independently.

---

## Usage

### Quickstart

```bash
python main.py --goal "your target behavior here"
```

### Custom streams and iterations

```bash
python main.py --goal "your target behavior here" \
  --n-streams 5 \
  --k-iterations 10 \
  --verbose
```

### Mix providers per role

```bash
python main.py --goal "your target behavior here" \
  --attacker-provider navigator --attacker-model gpt-oss-120b \
  --target-provider openai    --target-model gpt-4o \
  --judge-provider navigator  --judge-model gpt-oss-120b
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--goal` | *(required)* | Target behavior to elicit from the model |
| `--n-streams` | `3` | Number of parallel conversation streams |
| `--k-iterations` | `2` | Max refinement iterations per stream |
| `--attacker-provider` | `navigator` | Provider for the attacker (`openai`, `navigator`) |
| `--target-provider` | `navigator` | Provider for the target |
| `--judge-provider` | `navigator` | Provider for the judge |
| `--attacker-model` | `gpt-oss-120b` | Model used to generate jailbreak prompts |
| `--target-model` | `gpt-oss-120b` | Model being attacked |
| `--judge-model` | `gpt-oss-120b` | Model used to score responses |
| `--success-threshold` | `10` | Judge score >= this value counts as a successful jailbreak |
| `--verbose` | `false` | Print per-stream, per-iteration progress |
| `--log-dir` | `logs/` | Directory to save JSON run logs |

---

## Output

Each run prints a summary to stdout and saves a full structured log to `logs/run_<timestamp>.json`.

**Terminal summary:**

```
============================================================
Best score:   10/10
Success rate: 2/3 streams
API calls:    42
Elapsed:      18.4s

--- Best Prompt (stream 1, iter 4, score 10/10) ---
<generated prompt>

--- Target Response ---
<truncated response>

Log saved to: logs/run_20240101_120000.json
```

**JSON log structure:**

```json
{
  "goal": "...",
  "timestamp": "20240101_120000",
  "result": {
    "best_score": 10,
    "success_count": 2,
    "total_api_calls": 42,
    "streams": [
      {
        "stream_id": 0,
        "best_score": 10,
        "iterations": [
          {
            "iteration": 1,
            "score": 4,
            "improvement": "...",
            "prompt": "...",
            "response": "..."
          }
        ]
      }
    ]
  }
}
```

---

## Project Structure

```
pair-jailbreak/
├── pair/
│   ├── __init__.py       Public API (PAIRConfig, RunResult, run_pair)
│   ├── config.py         PAIRConfig dataclass
│   ├── prompts.py        Attacker and judge system prompts (verbatim from paper)
│   ├── models.py         LLM clients for OpenAI and Navigator
│   ├── attacker.py       Attacker agent — generates and refines jailbreak prompts
│   ├── judge.py          Judge agent — scores prompt-response pairs
│   └── algorithm.py      Core PAIR loop: N async streams × K iterations
├── main.py               CLI entry point
├── pyproject.toml
└── .env                  API keys (not committed)
```

---

## Contributing

Contributions are welcome for research and educational purposes.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request

Please keep contributions scoped to research tooling, evaluation improvements, and documentation. Do not submit features designed to target production systems without authorization.

---

## Ethical Use

This tool is provided strictly for **authorized security research, AI safety evaluation, and academic study**. Use it only against models and systems you own or have explicit written permission to test.

Misuse of this tool to attack third-party systems without authorization may violate the Computer Fraud and Abuse Act (CFAA), terms of service agreements, and other applicable laws. The authors accept no liability for misuse.

---

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{chao2023jailbreaking,
  title   = {Jailbreaking Black Box Large Language Models in Twenty Queries},
  author  = {Chao, Patrick and Robey, Alexander and Dobriban, Edgar and
             Hassani, Hamed and Pappas, George J. and Wong, Eric},
  journal = {arXiv preprint arXiv:2310.08419},
  year    = {2023}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
