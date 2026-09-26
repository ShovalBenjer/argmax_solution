"""jev router: cost/latency-aware model routing.

Priority: local small LMs (Ollama) -> free tiers (GitHub Models, other free APIs)
-> paid escalation only when the task needs it.

Every routing decision can be audited: `route_with_decision` returns a
`RouteDecision` record (task hash, complexity, threshold, chosen route,
reasons) instead of a bare route, and decisions can be appended to a JSONL
ledger via `log_path` or the `JEV_DECISION_LOG` env var. Task text is never
written to the ledger — only its sha256 — so logs stay PII-free.

Usage:
    from jev.router import route, route_with_decision
    r = route("summarize this log")
    print(r.name, r.model, r.base_url)
    r, d = route_with_decision("summarize this log", log_path="state/jev_decisions.jsonl")
    print(d.to_json())
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


# --- Calibration constants -------------------------------------------------
# ESCALATE_THRESHOLD decides which tasks stay on the cheap local route and
# which escalate to larger/free-tier models. A mis-tuned threshold silently
# degrades quality (too low: wasted spend on easy tasks; too high: hard tasks
# served by a small local model with no trace). Per ADR-0009 this value must
# be calibrated against measured task outcomes, not vibes; every "kept local"
# decision is auditable via RouteDecision so drift is visible.
ESCALATE_THRESHOLD = 0.55

# Keyword hits that push a task toward escalation. The list is deliberately a
# named constant — not inline literals — so the heuristic's bias surface is
# reviewable in one place, and so adversarial tests can pin its behavior.
COMPLEXITY_KEYWORDS = (
    "prove", "security", "architecture", "refactor",
    "distributed", "concurrency", "formal", "cryptograph",
)

#: Env var naming an append-only JSONL ledger for route decisions.
DECISION_LOG_ENV = "JEV_DECISION_LOG"

_NO_ROUTE = "no model route available: start Ollama or set JEV_MODEL_TOKEN"


@dataclass
class Route:
    name: str
    kind: str  # ollama | github-models | free-api | paid
    model: str
    base_url: str
    api_key_env: str | None
    cost_per_1k: float
    latency_ms_p50: int
    max_context: int
    available: bool = field(default=False, compare=False)


@dataclass(frozen=True)
class RouteDecision:
    """Auditable record of one routing decision.

    `task_sha256` (never the task text) keeps the ledger correlatable with
    task logs elsewhere without storing PII or prompt content.
    """
    task_sha256: str
    task_len: int
    complexity: float
    threshold: float
    chosen: str  # route name, or "none" when no route was available
    kind: str
    budget: str
    local_available: bool
    free_available: bool
    reasons: tuple[str, ...]
    decided_at: str  # UTC ISO-8601

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["reasons"] = list(d["reasons"])
        return json.dumps(d, sort_keys=True)


def _ollama_alive(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def default_routes() -> list[Route]:
    return [
        Route("ollama-local", "ollama",
              os.getenv("JEV_LOCAL_MODEL", "qwen2.5:7b"),
              "http://localhost:11434", None, 0.0, 400, 32768),
        Route("github-models", "github-models",
              os.getenv("JEV_GH_MODEL", "gpt-4o-mini"),
              "https://models.github.ai/inference", "JEV_MODEL_TOKEN", 0.0, 1200, 128000),
    ]


def estimate_complexity(task: str) -> float:
    """0..1 heuristic: short/simple tasks stay local, hard ones escalate."""
    t = task.lower()
    score = min(len(task) / 4000, 1.0) * 0.4
    score += 0.15 * sum(1 for m in COMPLEXITY_KEYWORDS if m in t)
    return min(score, 1.0)


def _route_impl(task: str, budget: str) -> tuple[Route | None, RouteDecision]:
    complexity = estimate_complexity(task)
    routes = [dataclasses.replace(r) for r in default_routes()]

    local = routes[0]
    local.available = _ollama_alive(local.base_url)
    gh = routes[1]
    gh.available = bool(os.getenv(gh.api_key_env or ""))

    common = dict(
        task_sha256=hashlib.sha256(task.encode("utf-8")).hexdigest(),
        task_len=len(task),
        complexity=round(complexity, 4),
        threshold=ESCALATE_THRESHOLD,
        budget=budget,
        local_available=local.available,
        free_available=gh.available,
        decided_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    def decide(route_obj: Route, *reasons: str) -> tuple[Route, RouteDecision]:
        return route_obj, RouteDecision(
            chosen=route_obj.name, kind=route_obj.kind,
            reasons=tuple(reasons), **common)

    # Local small LM wins for simple tasks when it is up.
    if local.available and (complexity < ESCALATE_THRESHOLD or budget == "free"):
        if budget == "free":
            why = "kept local: budget=free forces local-first"
        else:
            why = (f"kept local: complexity {complexity:.2f} < "
                   f"threshold {ESCALATE_THRESHOLD}")
        return decide(local, why, f"ollama available at {local.base_url}")
    if gh.available:
        return decide(
            gh,
            (f"escalated: complexity {complexity:.2f} vs threshold "
             f"{ESCALATE_THRESHOLD} (local_available={local.available})"),
            f"github-models available via {gh.api_key_env}")
    if local.available:
        return decide(local, "fallback: local is the only available route",
                      f"ollama available at {local.base_url}")
    return None, RouteDecision(
        chosen="none", kind="none",
        reasons=("no route available: start Ollama or set JEV_MODEL_TOKEN",),
        **common)


def _maybe_log(decision: RouteDecision, log_path: str | None) -> None:
    path = log_path or os.getenv(DECISION_LOG_ENV)
    if not path:
        return
    p = Path(path)
    if str(p.parent) not in (".", ""):
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(decision.to_json() + "\n")


def route(task: str, budget: str = "free", log_path: str | None = None) -> Route:
    """Pick the cheapest route that can plausibly handle the task.

    budget: "free" (never paid), "balanced" (prefer free, allow paid when hard),
            "max-quality" (best capable route regardless of cost).
    log_path: optional JSONL ledger for the routing decision (or set
            JEV_DECISION_LOG). Failed decisions are logged too.
    """
    route_obj, decision = _route_impl(task, budget)
    _maybe_log(decision, log_path)
    if route_obj is None:
        raise RuntimeError(_NO_ROUTE)
    return route_obj


def route_with_decision(task: str, budget: str = "free",
                        log_path: str | None = None) -> tuple[Route, RouteDecision]:
    """Like route(), but also returns the auditable RouteDecision."""
    route_obj, decision = _route_impl(task, budget)
    _maybe_log(decision, log_path)
    if route_obj is None:
        raise RuntimeError(_NO_ROUTE)
    return route_obj, decision
