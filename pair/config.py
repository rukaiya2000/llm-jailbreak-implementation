from dataclasses import dataclass, field


@dataclass
class PAIRConfig:
    goal: str
    n_streams: int = 30
    k_iterations: int = 3
    attacker_model: str = "claude-haiku-4-5-20251001"
    target_model: str = "claude-sonnet-4-6"
    judge_model: str = "claude-sonnet-4-6"
    provider: str = "anthropic"
    success_threshold: int = 10
    verbose: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"N={self.n_streams},K={self.k_iterations}"
