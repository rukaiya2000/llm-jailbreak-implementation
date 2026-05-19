import asyncio
import json
import time
from dataclasses import dataclass, field

from .attacker import Attacker
from .config import PAIRConfig
from .judge import Judge
from .models import make_client
from .prompts import ATTACKER_FEEDBACK_MSG, INITIAL_ATTACKER_MSG


@dataclass
class IterationRecord:
    iteration: int
    prompt: str
    response: str
    score: int
    improvement: str


@dataclass
class StreamResult:
    stream_id: int
    iterations: list[IterationRecord] = field(default_factory=list)

    @property
    def best_score(self) -> int:
        return max((r.score for r in self.iterations), default=0)

    @property
    def best_iteration(self) -> IterationRecord:
        return max(self.iterations, key=lambda r: r.score)


@dataclass
class RunResult:
    config: PAIRConfig
    streams: list[StreamResult]
    elapsed_seconds: float

    @property
    def best_score(self) -> int:
        return max((s.best_score for s in self.streams), default=0)

    @property
    def success_count(self) -> int:
        return sum(1 for s in self.streams if s.best_score >= self.config.success_threshold)

    @property
    def total_api_calls(self) -> int:
        # Each iteration: attacker + target + judge = 3 calls
        return sum(len(s.iterations) * 3 for s in self.streams)


async def _run_stream(
    stream_id: int,
    attacker: Attacker,
    target_client,
    judge: Judge,
    config: PAIRConfig,
) -> StreamResult:
    result = StreamResult(stream_id)
    attacker_msgs: list[dict] = [
        {"role": "user", "content": INITIAL_ATTACKER_MSG.format(goal=config.goal)}
    ]

    for k in range(config.k_iterations):
        improvement, prompt = await attacker.generate(attacker_msgs)
        if not prompt or not prompt.strip():
            break

        attacker_msgs.append(
            {"role": "assistant", "content": json.dumps({"improvement": improvement, "prompt": prompt})}
        )

        response = await target_client.chat([{"role": "user", "content": prompt}], "")
        score = await judge.score(prompt, response)

        result.iterations.append(IterationRecord(k + 1, prompt, response, score, improvement))

        if config.verbose:
            print(f"  [stream {stream_id:02d}] iter {k + 1}/{config.k_iterations}: score={score}")

        if score >= config.success_threshold:
            break

        attacker_msgs.append({
            "role": "user",
            "content": ATTACKER_FEEDBACK_MSG.format(
                response=response, goal=config.goal, score=score
            ),
        })

    return result


async def run_pair(config: PAIRConfig) -> RunResult:
    attacker_client = make_client(config.attacker_provider, config.attacker_model)
    target_client = make_client(config.target_provider, config.target_model)
    judge_client = make_client(config.judge_provider, config.judge_model)

    judge = Judge(judge_client, config.goal)

    if config.verbose:
        print(f"\nRunning {config.n_streams} streams × {config.k_iterations} iterations [{config.label}]")

    start = time.monotonic()
    streams = await asyncio.gather(
        *[
            _run_stream(i, Attacker(attacker_client, config.goal), target_client, judge, config)
            for i in range(config.n_streams)
        ]
    )
    elapsed = time.monotonic() - start

    return RunResult(config, list(streams), elapsed)
