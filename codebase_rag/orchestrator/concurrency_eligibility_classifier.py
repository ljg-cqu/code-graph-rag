"""
Concurrency Eligibility Classifier module for automatic parallel execution detection.
Determines if a task can be safely parallelized without explicit user request.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from codebase_rag.config import settings
from codebase_rag.providers import get_provider_from_config
from codebase_rag.shared.query_router import QueryMode
from codebase_rag.utils.path_utils import get_all_code_files


@dataclass
class EligibilityResult:
    """Result of eligibility classification with fallback suggestion."""

    eligible: bool
    task_type: str
    confidence: float
    fallback_action: str | None = None

    def __iter__(self) -> Iterator[bool | str | float]:
        """Support backward-compatible 3-tuple unpacking."""
        return iter((self.eligible, self.task_type, self.confidence))


class ConcurrencyEligibilityClassifier:
    """
    Classifies user requests/tasks to determine if they are eligible for automatic parallel execution.
    Uses explicit user overrides, minimal safety rules, and LLM intent analysis (no rule-based pattern matching for eligible tasks).
    """

    # Enhanced LLM system prompt with codebase context and mode awareness
    # NOTE: {prompt} is NOT included here. pydantic_ai sends the user prompt
    # as a separate message via agent.run(prompt). Including {prompt} in the
    # system prompt would create a confusing double-prompt where the LLM sees
    # both "{prompt}" literally in the system message AND the actual user message.
    LLM_ELIGIBILITY_PROMPT = """You are a parallel task eligibility classifier for a codebase analysis system.
Evaluate if the user request (provided separately) can be split into independent,
non-overlapping subtasks that can be executed in parallel.

Codebase Context:
- Total files: {file_count}
- Primary languages: {languages}
- Repository size: {repo_size}
- Query mode: {query_mode}

Respond ONLY with a valid JSON object:
{{
  "eligible": boolean,
  "confidence": float between 0.0 and 1.0,
  "task_type": string,
  "reasoning": string,
  "fallback_action": string or null
}}

Safety Rules:
- NEVER parallelize tasks that modify, create, delete, or update files/code/config
- ALWAYS parallelize read-only tasks that analyze multiple files or entities
- For conceptual/theoretical questions, return eligible=false with task_type="conceptual_question"
- For explicit write requests ("fix this", "create a file", "refactor X"), return eligible=false with task_type="write_operation"
- When uncertain, prefer sequential execution (eligible=false)
"""

    @staticmethod
    def _coerce_float(value: object, default: float = 0.0) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def __init__(self):
        self.enabled: bool = getattr(settings, "CGR_AUTO_PARALLEL_ENABLED", True)
        # self.threshold is the BASE threshold (unchanged from current code).
        # It is set from CGR_PARALLEL_ELIGIBILITY_THRESHOLD and stays fixed.
        self.threshold: float = getattr(
            settings, "CGR_PARALLEL_ELIGIBILITY_THRESHOLD", 0.6
        )
        # self._effective_threshold is the dynamically adjusted threshold that
        # actually gets used in the eligibility decision. It starts equal to
        # self.threshold and may be adjusted based on historical success rates.
        # This replaces the previous approach of adjusting confidence scores
        # via multipliers, which produced opaque "effective threshold" changes.
        self._effective_threshold: float = self.threshold
        self.min_subtask_count: int = getattr(
            settings, "CGR_PARALLEL_MIN_SUBTASKS", 2
        )
        self.agent: Agent | None = None
        self._planner = None
        # Dynamic calibration state (tracks success rates per task type)
        self.success_rate_tracker: dict[str, list[bool]] = {}
        # Adaptive adjustment enabled via CGR_PARALLEL_ADAPTIVE_THRESHOLD config
        self.adaptive_adjustment_enabled: bool = getattr(
            settings, "CGR_PARALLEL_ADAPTIVE_THRESHOLD", True
        )
        # Cache for codebase context to avoid recomputation
        self._codebase_context_cache: dict[str, object] | None = None

    def _adjust_threshold_based_on_success(self, task_type: str) -> float:
        """Dynamically adjust the effective eligibility threshold based on historical success rates.

        This REPLACES the existing confidence-multiplier calibration in is_eligible()
        (which adjusted confidence by *1.1 or *0.9). Threshold-based adjustment is:
        - More intuitive: lowering the bar vs. boosting the score
        - More observable: log messages show explicit threshold changes
        - Less prone to double-effect: only one mechanism, not two

        The effective threshold is adjusted relative to self.threshold (the base):
        - High success rate (>0.85): lower threshold to enable more parallelization
        - Good success rate (>0.7): keep base threshold
        - Moderate success rate (>0.5): slightly raise threshold
        - Low success rate (<0.5): significantly raise threshold

        Args:
            task_type: The task type to adjust threshold for

        Returns:
            The effective threshold to use for this eligibility decision
        """
        if not self.adaptive_adjustment_enabled:
            return self.threshold

        if task_type not in self.success_rate_tracker:
            return self._effective_threshold

        successes = self.success_rate_tracker[task_type]
        # Aligned with existing code's minimum of 10 data points (not 5).
        # The previous spec used 5, but the existing code requires 10 for
        # more reliable calibration. Using fewer data points produces noisy
        # threshold adjustments that may over-correct on limited evidence.
        if len(successes) < 10:
            return self._effective_threshold

        success_rate = sum(successes) / len(successes)

        # Adjust effective threshold based on success rate
        if success_rate >= 0.85:
            # High success rate: lower threshold to enable more parallelization
            new_threshold = max(0.5, self.threshold * 0.8)
        elif success_rate >= 0.7:
            # Good success rate: keep base threshold
            new_threshold = self.threshold
        elif success_rate >= 0.5:
            # Moderate success rate: slightly raise threshold
            new_threshold = min(0.8, self.threshold * 1.1)
        else:
            # Low success rate: significantly raise threshold
            new_threshold = min(0.9, self.threshold * 1.3)

        if new_threshold != self._effective_threshold:
            logger.info(
                f"Adaptive threshold adjusted for {task_type}: "
                f"{self._effective_threshold:.2f} → {new_threshold:.2f} "
                f"(success rate: {success_rate:.2%}, base threshold: {self.threshold:.2f})"
            )
            self._effective_threshold = new_threshold

        return self._effective_threshold

    def _get_codebase_context(self) -> tuple[int, str, str]:
        """Get codebase context (file count, primary languages, repo size).

        Returns:
            Tuple of (file_count, languages, repo_size)
        """
        if self._codebase_context_cache is not None:
            return cast(
                tuple[int, str, str],
                tuple(self._codebase_context_cache.values()),
            )

        try:
            repo_path = Path(settings.TARGET_REPO_PATH)
            all_files = get_all_code_files(repo_path)
            file_count = len(all_files)

            # Detect primary languages from file extensions
            extensions = [f.suffix.lower() for f in all_files if f.suffix]
            lang_counts: dict[str, int] = {}
            for ext in extensions:
                lang = {
                    ".py": "Python",
                    ".js": "JavaScript",
                    ".ts": "TypeScript",
                    ".jsx": "JavaScript",
                    ".tsx": "TypeScript",
                    ".java": "Java",
                    ".cpp": "C++",
                    ".h": "C++",
                    ".hpp": "C++",
                    ".go": "Go",
                    ".rs": "Rust",
                    ".cs": "C#",
                    ".rb": "Ruby",
                    ".php": "PHP",
                    ".swift": "Swift",
                    ".kt": "Kotlin",
                    ".scala": "Scala",
                }.get(ext, "Other")
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

            primary_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            languages = ", ".join([lang for lang, count in primary_langs if count > 5])

            # Get repo size
            total_size = sum(f.stat().st_size for f in all_files if f.exists())
            repo_size = f"{total_size / (1024 * 1024):.1f}MB" if total_size > 0 else "unknown"

            self._codebase_context_cache = {
                "file_count": file_count,
                "languages": languages,
                "repo_size": repo_size,
            }

            return file_count, languages, repo_size

        except Exception as e:
            logger.warning(f"Failed to get codebase context: {e}")
            self._codebase_context_cache = {
                "file_count": 0,
                "languages": "unknown",
                "repo_size": "unknown",
            }
            return 0, "unknown", "unknown"

    async def _get_llm_eligibility(
        self,
        prompt: str,
        query_mode: QueryMode = QueryMode.CODE_ONLY,
    ) -> tuple[float, str]:
        """Run enhanced LLM analysis with codebase context."""
        if not self.agent:
            config = settings.active_orchestrator_config
            provider = get_provider_from_config(config)
            llm = provider.create_model(config.model_id)

            # Get codebase context (only if enabled and not already cached)
            if getattr(settings, "CGR_PARALLEL_CODEBASE_CONTEXT", True):
                file_count, languages, repo_size = self._get_codebase_context()
            else:
                file_count = 0
                languages = "unknown"
                repo_size = "unknown"

            # Format system prompt with codebase context and query mode
            # The user prompt is sent separately by pydantic_ai via agent.run(prompt)
            system_prompt = self.LLM_ELIGIBILITY_PROMPT.format(
                file_count=file_count,
                languages=languages,
                repo_size=repo_size,
                query_mode=query_mode.value,
            )

            self.agent = Agent(
                model=llm,
                system_prompt=system_prompt,
                output_type=dict,
                retries=settings.AGENT_RETRIES,
            )

        try:
            result = await self.agent.run(
                prompt, usage_limits=UsageLimits(request_limit=settings.AGENT_REQUEST_LIMIT)
            )
            result_data_raw = result.output
            if not isinstance(result_data_raw, dict):
                return 0.0, "llm_invalid_output"

            result_data = cast(dict[str, object], result_data_raw)
            confidence = max(
                0.0,
                min(1.0, self._coerce_float(result_data.get("confidence", 0.0))),
            )
            task_type = (
                result_data.get("task_type", "llm_analyzed")
                if result_data.get("eligible", False)
                else "llm_rejected"
            )
            # Log reasoning for debugging/auditability
            reasoning = result_data.get("reasoning", "")
            if reasoning:
                logger.debug(f"LLM eligibility reasoning: {reasoning}")

            return confidence, str(task_type)
        except Exception as e:
            logger.warning(
                f"LLM eligibility check failed: {str(e)}, falling back to sequential execution"
            )
            return 0.0, "llm_check_failed"

    async def is_eligible(
        self,
        prompt: str,
        subtask_count: int | None = None,
        has_write_operations: bool = False,
        query_mode: QueryMode = QueryMode.CODE_ONLY,
    ) -> EligibilityResult:
        """Determine if a task is eligible for automatic parallel execution.

        All semantic decisions are delegated to the LLM. No regex pre-filtering.

        Args:
            prompt: User's natural language request / task description
            subtask_count: Optional number of detected subtasks for this request
            has_write_operations: Whether the task includes any write/modify operations
            query_mode: Current query mode (affects eligibility for document queries)

        Returns:
            EligibilityResult with eligible, task_type, confidence, and fallback_action
        """
        if not self.enabled:
            return EligibilityResult(False, "concurrency_disabled", 0.0)

        if has_write_operations:
            return EligibilityResult(False, "write_operation", 0.0)

        if subtask_count is not None and subtask_count < self.min_subtask_count:
            logger.debug(
                f"Task not eligible for parallel execution: only {subtask_count} subtasks (min {self.min_subtask_count})"
            )
            return EligibilityResult(False, "insufficient_subtasks", 0.0)

        # Delegate ALL semantic decisions to the LLM
        confidence, task_type = await self._get_llm_eligibility(prompt, query_mode)
        effective_threshold = self._adjust_threshold_based_on_success(task_type)

        if confidence >= effective_threshold:
            logger.info(
                f"Task eligible for parallel execution: type={task_type}, confidence={confidence:.2f}, effective_threshold={effective_threshold:.2f}"
            )
            return EligibilityResult(True, task_type, confidence)

        logger.debug(
            f"Task not eligible for parallel execution: LLM confidence {confidence:.2f} below effective threshold {effective_threshold:.2f} (base: {self.threshold:.2f})"
        )
        return EligibilityResult(False, task_type, confidence)

    def record_execution_result(self, task_type: str, success: bool) -> None:
        """Record execution result for dynamic threshold calibration."""
        if task_type not in self.success_rate_tracker:
            self.success_rate_tracker[task_type] = []
        self.success_rate_tracker[task_type].append(success)
        # Keep only last 100 results per task type for calibration
        if len(self.success_rate_tracker[task_type]) > 100:
            self.success_rate_tracker[task_type] = self.success_rate_tracker[task_type][
                -100:
            ]
