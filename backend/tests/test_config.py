"""Config loading + validation tests."""
import os
from pathlib import Path

import pytest

from jarvis.config import Settings, get_settings


def test_defaults():
    s = Settings()
    assert s.ws_port == 8765
    assert s.sample_rate == 16000
    assert s.wake_word == "jarvis"
    assert s.llm_model == "llama3.2:1b"
    assert s.tts_voice == "de_DE-kerstin-low"
    assert s.tts_length_scale == 1.3
    assert "Illumina" in s.whisper_initial_prompt
    assert "BeadChip" in s.whisper_initial_prompt
    assert s.tts_backend == "edge"
    assert s.tts_edge_voice == "de-DE-KatjaNeural"
    assert s.tts_sample_rate == 24000
    assert s.coding_model == "qwen2.5-coder:14b"
    assert s.coding_model_small == "qwen2.5-coder:1.5b"
    assert s.coding_model_medium == "qwen2.5-coder:7b"
    assert s.coding_default_tier == "small"
    assert s.coding_max_tokens == 2000
    assert s.coding_scripts_dir.name == "Jarvis_Scripts"
    assert s.whisper_model == "small"
    assert s.whisper_lang == "de"
    assert s.vad_threshold == 0.65
    assert s.wake_threshold == 0.5
    assert s.llm_max_tokens == 200
    assert s.min_transcript_words == 2
    assert s.groq_model == "llama-3.1-8b-instant"
    assert s.groq_base_url == "https://api.groq.com/openai/v1"
    assert s.use_cloud is True
    assert s.route_strategy == "complexity"
    assert s.reaction_max_tokens == 40
    assert s.max_listen_s == 15.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("JARVIS_WS_PORT", "9000")
    monkeypatch.setenv("JARVIS_LLM_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("JARVIS_GROQ_MODEL", "llama-3.3-70b-versatile")
    s = Settings()
    assert s.ws_port == 9000
    assert s.llm_model == "qwen2.5:3b"
    assert s.groq_model == "llama-3.3-70b-versatile"


def test_use_cloud_can_be_disabled(monkeypatch):
    monkeypatch.setenv("JARVIS_USE_CLOUD", "0")
    assert Settings().use_cloud is False


def test_coding_default_tier_valid_value_passes_through(monkeypatch):
    monkeypatch.setenv("JARVIS_CODING_DEFAULT_TIER", "medium")
    assert Settings().coding_default_tier == "medium"


def test_coding_default_tier_invalid_value_falls_back_to_small(monkeypatch):
    # A stale/typo'd env var must not propagate — app.py's coding_models
    # dict is keyed only by TIER_ORDER's names, so an unrecognized tier
    # here would otherwise KeyError on every coding request.
    monkeypatch.setenv("JARVIS_CODING_DEFAULT_TIER", "smal")
    assert Settings().coding_default_tier == "small"


def test_get_settings_cached():
    a = get_settings()
    b = get_settings()
    assert a is b
