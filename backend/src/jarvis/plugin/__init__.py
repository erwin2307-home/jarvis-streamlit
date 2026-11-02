"""Plugin Tool decorator + Registry. No real tools ship in Phase 1."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, ClassVar


ToolFunc = Callable[..., Awaitable["ToolResult"]]


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    summary: str = ""


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict[str, dict[str, Any]]
    func: ToolFunc
    confirm: bool = False
    timeout_s: float = 10.0


class _RegistryImpl:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def add(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} already registered")
        self._tools[spec.name] = spec

    def clear(self) -> None:
        self._tools.clear()

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(ok=False, data=None, summary=f"unknown tool: {name}")
        try:
            return await asyncio.wait_for(spec.func(**args), timeout=spec.timeout_s)
        except asyncio.TimeoutError:
            return ToolResult(ok=False, data=None, summary=f"tool {name} timeout")
        except Exception as exc:
            return ToolResult(ok=False, data=None, summary=f"tool error: {exc}")

    def json_schemas(self) -> list[dict[str, Any]]:
        """OpenAI-compatible tool list for LLM function-calling."""
        out: list[dict[str, Any]] = []
        for spec in self._tools.values():
            properties: dict[str, dict[str, Any]] = {}
            required: list[str] = []
            for arg_name, meta in spec.schema.items():
                m = dict(meta)
                if m.pop("required", False):
                    required.append(arg_name)
                properties[arg_name] = m
            out.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return out


Registry = _RegistryImpl()


class Tool:
    """Decorator namespace for registering tools."""

    @staticmethod
    def register(
        *,
        name: str,
        description: str,
        schema: dict[str, dict[str, Any]],
        confirm: bool = False,
        timeout_s: float = 10.0,
    ) -> Callable[[ToolFunc], ToolFunc]:
        def deco(func: ToolFunc) -> ToolFunc:
            Registry.add(ToolSpec(
                name=name,
                description=description,
                schema=schema,
                func=func,
                confirm=confirm,
                timeout_s=timeout_s,
            ))
            return func
        return deco
