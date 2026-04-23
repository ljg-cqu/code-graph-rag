"""Quota status reporting for LLM providers.

This module provides user-facing quota status display and warnings
for LLM rate limiting and quota management.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .rate_limiter import QuotaStatus, get_rate_limiter

if TYPE_CHECKING:
    from .config import AppConfig


class QuotaStatusReporter:
    """Reports quota status to users.

    Displays current quota status for all configured LLM providers
    with visual indicators for health, warnings, and exhaustion.
    """

    def __init__(self, console: Console | None = None):
        """Initialize the reporter.

        Args:
            console: Rich console for output. Creates new one if not provided.
        """
        self.console = console or Console()

    def print_status(self, settings: AppConfig | None = None) -> None:
        """Print current quota status for all providers.

        Args:
            settings: Application settings to get provider configuration.
        """
        limiter = get_rate_limiter()

        table = Table(title="LLM Provider Quota Status")
        table.add_column("Provider", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("Status", style="bold")
        table.add_column("Usage", justify="right")
        table.add_column("Reset", style="dim")

        # Collect providers from settings
        providers_to_check: list[tuple[str, str]] = []
        if settings:
            # Primary provider
            if settings.PRIMARY_LLM_PROVIDER and settings.PRIMARY_LLM_MODEL:
                providers_to_check.append(
                    (settings.PRIMARY_LLM_PROVIDER, settings.PRIMARY_LLM_MODEL)
                )
            # Fallback providers (use default model mapping)
            model_map = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-haiku",
                "doubao": "doubao-seed-2-0-pro-260215",
                "ollama": "llama3",
                "google": "gemini-pro",
            }
            for provider in settings.FALLBACK_PROVIDERS:
                model = model_map.get(provider, "default")
                providers_to_check.append((provider, model))

        # If no providers configured, show from limiter's tracked providers
        if not providers_to_check:
            for key in limiter._quota_info:
                if "/" in key:
                    provider, model = key.split("/", 1)
                    providers_to_check.append((provider, model))

        # Deduplicate
        seen = set()
        for provider, model in providers_to_check:
            key = f"{provider}/{model}"
            if key in seen:
                continue
            seen.add(key)

            info = limiter._quota_info.get(key)
            status = limiter.check_quota(provider, model)

            status_color = {
                QuotaStatus.HEALTHY: "green",
                QuotaStatus.WARNING: "yellow",
                QuotaStatus.CRITICAL: "red",
                QuotaStatus.EXHAUSTED: "red",
            }.get(status, "white")

            usage_str = "N/A"
            if info:
                if info.total_requests:
                    usage_str = f"{info.used_requests}/{info.total_requests}"
                if info.total_tokens:
                    usage_str += f" ({info.used_tokens}/{info.total_tokens} tokens)"
            else:
                usage_str = "Not tracked"

            reset_str = "Unknown"
            if info and info.reset_time:
                reset_str = info.reset_time.strftime("%Y-%m-%d")

            table.add_row(
                provider,
                model,
                f"[{status_color}]{status.name}[/{status_color}]",
                usage_str,
                reset_str,
            )

        if not providers_to_check:
            self.console.print(
                "[yellow]No LLM providers configured. "
                "Set CGR_PRIMARY_LLM_PROVIDER and CGR_PRIMARY_LLM_MODEL.[/yellow]"
            )
            return

        self.console.print(table)

    def print_warning(self, provider: str, model: str) -> None:
        """Print a warning if quota is critical or exhausted.

        Args:
            provider: Provider name to check.
            model: Model name to check.
        """
        limiter = get_rate_limiter()
        status = limiter.check_quota(provider, model)

        if status == QuotaStatus.EXHAUSTED:
            self.console.print(
                Panel(
                    "[bold red]Quota Exceeded[/bold red]\n\n"
                    f"The LLM provider ({provider}/{model}) has exceeded its quota.\n"
                    "The system will attempt to use fallback providers if configured.\n\n"
                    "[dim]Check status with: cgr quota[/dim]",
                    title="LLM Quota Warning",
                    border_style="red",
                )
            )
        elif status == QuotaStatus.CRITICAL:
            self.console.print(
                Panel(
                    "[bold yellow]Quota Critical[/bold yellow]\n\n"
                    f"The LLM provider ({provider}/{model}) is near quota limit.\n"
                    "Consider reducing query frequency or switching providers.",
                    title="LLM Quota Warning",
                    border_style="yellow",
                )
            )

    def get_status_summary(self) -> dict[str, str]:
        """Get a summary of quota status for all tracked providers.

        Returns:
            Dict mapping provider/model to status name.
        """
        limiter = get_rate_limiter()
        summary = {}
        for key in limiter._quota_info:
            if "/" in key:
                status = limiter.check_quota(key.split("/")[0], key.split("/")[1])
                summary[key] = status.name
        return summary


def format_quota_reset_time(reset_time: datetime | None) -> str:
    """Format quota reset time for display.

    Args:
        reset_time: Datetime when quota resets.

    Returns:
        Human-readable string like "in 3 days" or "tomorrow".
    """
    if reset_time is None:
        return "Unknown"

    now = datetime.now(UTC)
    if reset_time.tzinfo is None:
        reset_time = reset_time.replace(tzinfo=UTC)

    delta = reset_time - now
    total_seconds = int(delta.total_seconds())

    if total_seconds < 0:
        return "Should reset soon"
    if total_seconds < 3600:
        minutes = total_seconds // 60
        return f"in {minutes} minute{'s' if minutes != 1 else ''}"
    if total_seconds < 86400:
        hours = total_seconds // 3600
        return f"in {hours} hour{'s' if hours != 1 else ''}"
    days = total_seconds // 86400
    if days == 1:
        return "tomorrow"
    return f"in {days} days"
