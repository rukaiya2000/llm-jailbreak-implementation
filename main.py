import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from pair import PAIRConfig, RunResult, run_pair

load_dotenv()


def _print_results(result: RunResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"Best score:   {result.best_score}/10")
    print(f"Success rate: {result.success_count}/{result.config.n_streams} streams")
    print(f"API calls:    {result.total_api_calls}")
    print(f"Elapsed:      {result.elapsed_seconds:.1f}s")

    if not any(s.iterations for s in result.streams):
        print("(no iterations completed)")
        return

    best_stream = max(result.streams, key=lambda s: s.best_score)
    if not best_stream.iterations:
        return
    best_iter = best_stream.best_iteration

    print(f"\n--- Best Prompt (stream {best_stream.stream_id}, iter {best_iter.iteration}, score {best_iter.score}/10) ---")
    print(best_iter.prompt)
    print("\n--- Target Response ---")
    truncated = best_iter.response[:600]
    print(truncated + ("..." if len(best_iter.response) > 600 else ""))


def _to_log_dict(result: RunResult) -> dict:
    return {
        "n_streams": result.config.n_streams,
        "k_iterations": result.config.k_iterations,
        "attacker": f"{result.config.attacker_provider}/{result.config.attacker_model}",
        "target": f"{result.config.target_provider}/{result.config.target_model}",
        "judge": f"{result.config.judge_provider}/{result.config.judge_model}",
        "best_score": result.best_score,
        "success_count": result.success_count,
        "total_api_calls": result.total_api_calls,
        "elapsed_seconds": round(result.elapsed_seconds, 2),
        "streams": [
            {
                "stream_id": s.stream_id,
                "best_score": s.best_score,
                "iterations": [
                    {
                        "iteration": r.iteration,
                        "score": r.score,
                        "improvement": r.improvement,
                        "prompt": r.prompt,
                        "response": r.response,
                    }
                    for r in s.iterations
                ],
            }
            for s in result.streams
        ],
    }


async def _run(args: argparse.Namespace) -> None:
    config = PAIRConfig(
        goal=args.goal,
        n_streams=args.n_streams,
        k_iterations=args.k_iterations,
        attacker_model=args.attacker_model,
        target_model=args.target_model,
        judge_model=args.judge_model,
        attacker_provider=args.attacker_provider,
        target_provider=args.target_provider,
        judge_provider=args.judge_provider,
        success_threshold=args.success_threshold,
        verbose=args.verbose,
    )
    result = await run_pair(config)
    _print_results(result)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"run_{ts}.json")
    with open(log_path, "w") as f:
        json.dump({"goal": args.goal, "timestamp": ts, "result": _to_log_dict(result)}, f, indent=2)
    print(f"\nLog saved to: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PAIR: Prompt Automatic Iterative Refinement (Chao et al. 2023)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--goal", required=True, help="Target behavior to elicit from the model")
    parser.add_argument("--n-streams", type=int, default=3, dest="n_streams",
                        help="Number of parallel conversation streams")
    parser.add_argument("--k-iterations", type=int, default=2, dest="k_iterations",
                        help="Max refinement iterations per stream")
    parser.add_argument("--attacker-provider", default="navigator", choices=["openai", "navigator"], dest="attacker_provider")
    parser.add_argument("--target-provider", default="navigator", choices=["openai", "navigator"], dest="target_provider")
    parser.add_argument("--judge-provider", default="navigator", choices=["openai", "navigator"], dest="judge_provider")
    parser.add_argument("--attacker-model", default="gpt-oss-120b", dest="attacker_model")
    parser.add_argument("--target-model", default="gpt-oss-120b", dest="target_model")
    parser.add_argument("--judge-model", default="gpt-oss-120b", dest="judge_model")
    parser.add_argument("--success-threshold", type=int, default=10, dest="success_threshold",
                        help="Judge score >= this value counts as a successful jailbreak")
    parser.add_argument("--verbose", action="store_true", help="Print per-stream progress")
    parser.add_argument("--log-dir", default="logs", dest="log_dir",
                        help="Directory for JSON logs")

    args = parser.parse_args()

    providers = (args.attacker_provider, args.target_provider, args.judge_provider)
    if "openai" in providers and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Add it to a .env file or export it.")
        sys.exit(1)
    if "navigator" in providers and not os.getenv("NAVIGATOR_API_KEY"):
        print("Error: NAVIGATOR_API_KEY not set. Add it to a .env file or export it.")
        sys.exit(1)

    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
