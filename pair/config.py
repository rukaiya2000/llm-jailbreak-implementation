from dataclasses import dataclass, field


@dataclass
class PAIRConfig:
    goal: str
    n_streams: int = 30
    k_iterations: int = 3
    attacker_model: str = "gpt-oss-120b"
    target_model: str = "gpt-oss-120b"
    judge_model: str = "gpt-oss-120b"
    attacker_provider: str = "navigator"
    target_provider: str = "navigator"
    judge_provider: str = "navigator"
    success_threshold: int = 10
    verbose: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"N={self.n_streams},K={self.k_iterations}"
