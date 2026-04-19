from __future__ import annotations

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(description="Message role: system, user, assistant, tool, or function")
    content: str = Field(description="Message content")


class TaskState(BaseModel):
    current_objective: str = Field(
        default="",
        description="The user's ultimate goal in this conversation",
    )
    completed_steps: list[str] = Field(
        default_factory=list,
        description="Key steps already taken or discoveries made",
    )
    pending_questions: list[str] = Field(
        default_factory=list,
        description="Unresolved questions or open issues",
    )
    active_code_elements: list[str] = Field(
        default_factory=list,
        description="Specific functions, classes, files under discussion",
    )
    key_decisions: list[str] = Field(
        default_factory=list,
        description="Architectural or approach decisions made during conversation",
    )
    error_states: list[str] = Field(
        default_factory=list,
        description="Errors encountered and their resolution status",
    )
    tool_results_summary: str = Field(
        default="",
        description="Summary of critical tool/query outputs that changed state",
    )
    user_preferences: list[str] = Field(
        default_factory=list,
        description="Constraints or preferences the user expressed",
    )


class CompressedState(BaseModel):
    task_state: TaskState = Field(description="Structured state extracted from conversation history")
    recent_messages: list[Message] = Field(
        default_factory=list,
        description="Most recent 2-4 messages preserved verbatim",
    )
    compression_rationale: str = Field(
        default="",
        description="Brief explanation of what was kept and why",
    )
