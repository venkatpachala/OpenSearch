"""
Tool base classes, registry, and schema error handling.

The LLM cannot execute arbitrary code. It MUST choose from the registered
typed tools. Every tool call is validated by Pydantic before execution.
Schema violations produce structured feedback the agent can use to repair
its arguments — this is Failure Type 2 (tool_schema_error).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel, ValidationError


# ---------------------------------------------------------------------------
# Base types
# ---------------------------------------------------------------------------


class ToolRequest(BaseModel):
    """Base class for all typed tool request models."""
    pass


class ToolResponse(BaseModel):
    """Structured response returned by every tool execution."""

    success: bool
    error: str | None = None
    data: dict[str, Any] = {}

    @classmethod
    def ok(cls, **data: Any) -> "ToolResponse":
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> "ToolResponse":
        return cls(success=False, error=error)


# ---------------------------------------------------------------------------
# Schema error — Failure Type 2
# ---------------------------------------------------------------------------


class ToolSchemaError(Exception):
    """
    Raised when the LLM provides arguments that fail Pydantic validation.

    Rather than crashing, the error is converted to structured feedback
    that the agent can use to repair its arguments on the next step.
    """

    def __init__(self, tool_name: str, validation_error: ValidationError) -> None:
        self.tool_name = tool_name
        self.validation_error = validation_error
        self.field_errors = validation_error.errors()
        super().__init__(
            f"Schema error for tool '{tool_name}': "
            f"{len(self.field_errors)} field error(s)"
        )

    def to_feedback(self) -> dict[str, Any]:
        """
        Structured feedback the agent can read to repair its arguments.

        Example output:
            {
              "error_type": "TOOL_SCHEMA_ERROR",
              "tool_name": "run_experiment",
              "field_errors": [
                {"field": "experiment_id", "message": "str type expected", "type": "string_type"}
              ]
            }
        """
        return {
            "error_type": "TOOL_SCHEMA_ERROR",
            "tool_name": self.tool_name,
            "field_errors": [
                {
                    "field": ".".join(str(loc) for loc in err["loc"]),
                    "message": err["msg"],
                    "type": err["type"],
                }
                for err in self.field_errors
            ],
        }


# ---------------------------------------------------------------------------
# Abstract Tool
# ---------------------------------------------------------------------------


class Tool(ABC):
    """
    Abstract base class for all ResearchRepro tools.

    Subclasses must define:
        name: str               — unique identifier
        description: str        — shown to the LLM in the tool list
        request_model: Type     — Pydantic model for argument validation
        execute(request) -> ToolResponse
    """

    name: str
    description: str
    request_model: Type[ToolRequest]

    @abstractmethod
    def execute(self, request: ToolRequest) -> ToolResponse:
        ...

    def validate_and_execute(self, raw_args: dict[str, Any]) -> ToolResponse:
        """
        Validate raw_args against request_model, then execute.
        Raises ToolSchemaError if validation fails.
        """
        try:
            request = self.request_model(**raw_args)
        except ValidationError as e:
            raise ToolSchemaError(self.name, e)
        return self.execute(request)

    def schema(self) -> dict[str, Any]:
        """Return the tool's JSON schema — shown to the LLM as the tool list."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.request_model.model_json_schema(),
        }


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """
    The authoritative set of tools available to the agent.

    The LLM sees `list_available()` and chooses a tool name.
    The registry validates and dispatches the call.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' not registered. "
                f"Available: {sorted(self._tools.keys())}"
            )
        return self._tools[name]

    def list_available(self) -> list[dict[str, Any]]:
        """Return schemas for all registered tools — fed to the planner."""
        return [t.schema() for t in self._tools.values()]

    def execute(self, name: str, raw_args: dict[str, Any]) -> ToolResponse:
        """
        Validate and execute a tool by name.
        Raises ToolSchemaError on bad arguments, KeyError on unknown tool.
        """
        return self.get(name).validate_and_execute(raw_args)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
