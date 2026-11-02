"""Plugin Tool + Registry tests."""
import asyncio

import pytest

from jarvis.plugin import Registry, Tool, ToolResult


@pytest.fixture(autouse=True)
def fresh_registry():
    Registry.clear()
    yield
    Registry.clear()


async def test_register_and_call_tool():
    @Tool.register(name="echo", description="echo input", schema={"text": {"type": "string"}})
    async def echo(text: str) -> ToolResult:
        return ToolResult(ok=True, data={"text": text}, summary=text)

    assert "echo" in Registry.names()
    result = await Registry.call("echo", {"text": "hi"})
    assert result.ok
    assert result.summary == "hi"


async def test_call_unknown_tool_returns_error():
    result = await Registry.call("missing", {})
    assert not result.ok
    assert "unknown" in result.summary.lower()


async def test_tool_timeout():
    @Tool.register(name="slow", description="sleeps", schema={}, timeout_s=0.1)
    async def slow() -> ToolResult:
        await asyncio.sleep(1)
        return ToolResult(ok=True, data=None, summary="")

    result = await Registry.call("slow", {})
    assert not result.ok
    assert "timeout" in result.summary.lower()


async def test_tool_exception_caught():
    @Tool.register(name="boom", description="raises", schema={})
    async def boom() -> ToolResult:
        raise RuntimeError("kapow")

    result = await Registry.call("boom", {})
    assert not result.ok
    assert "kapow" in result.summary


def test_json_schemas_format():
    @Tool.register(
        name="weather",
        description="get weather",
        schema={"city": {"type": "string", "required": True}},
    )
    async def weather(city: str) -> ToolResult:
        return ToolResult(ok=True, data={}, summary="")

    schemas = Registry.json_schemas()
    assert len(schemas) == 1
    s = schemas[0]
    assert s["type"] == "function"
    assert s["function"]["name"] == "weather"
    assert s["function"]["description"] == "get weather"
    assert "city" in s["function"]["parameters"]["properties"]
    assert s["function"]["parameters"]["required"] == ["city"]


def test_register_duplicate_raises():
    @Tool.register(name="dup", description="", schema={})
    async def dup1() -> ToolResult:
        return ToolResult(ok=True, data=None, summary="")

    with pytest.raises(ValueError, match="already registered"):
        @Tool.register(name="dup", description="", schema={})
        async def dup2() -> ToolResult:
            return ToolResult(ok=True, data=None, summary="")
