"""LLM tests using fake stream."""
import pytest

from jarvis.llm import FakeLlm, LlmEvent, OpenAiCompatLlm, select_cloud_provider


async def test_fake_llm_streams_chunks():
    fake = FakeLlm(chunks=["Hallo ", "Erwin, ", "wie geht's?"])
    out = []
    async for ev in fake.chat([{"role": "user", "content": "hi"}]):
        out.append(ev)
    assert [e.kind for e in out] == ["chunk", "chunk", "chunk", "done"]
    assert "".join(e.text for e in out if e.kind == "chunk") == "Hallo Erwin, wie geht's?"
    assert out[-1].kind == "done"
    assert out[-1].text == "Hallo Erwin, wie geht's?"


async def test_fake_llm_empty_chunks_still_yields_done():
    fake = FakeLlm(chunks=[])
    out = [ev async for ev in fake.chat([{"role": "user", "content": "x"}])]
    assert len(out) == 1
    assert out[0].kind == "done"
    assert out[0].text == ""


def test_openai_compat_llm_rejects_empty_key():
    with pytest.raises(ValueError, match="non-empty api_key"):
        OpenAiCompatLlm(api_key="", model="llama-3.1-8b-instant")


def test_openai_compat_llm_works_with_groq_style_args():
    # Same class serves DeepSeek and Groq — just different base_url/model.
    llm = OpenAiCompatLlm(
        api_key="fake-groq-key",
        model="llama-3.1-8b-instant",
        base_url="https://api.groq.com/openai/v1",
    )
    assert llm._model == "llama-3.1-8b-instant"


def test_select_cloud_provider_prefers_groq_when_both_present():
    assert select_cloud_provider(groq_key="groq-key", deepseek_key="ds-key") == "groq"


def test_select_cloud_provider_falls_back_to_deepseek():
    assert select_cloud_provider(groq_key="", deepseek_key="ds-key") == "deepseek"


def test_select_cloud_provider_none_when_neither_present():
    assert select_cloud_provider(groq_key="", deepseek_key="") is None
