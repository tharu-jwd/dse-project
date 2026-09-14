from app.streaming.commands import HOTWORDS
from app.streaming.inference import build_transcribe_kwargs


def test_dictation_mode_never_applies_hotwords():
    # Regardless of whether the installed faster-whisper supports
    # `hotwords`, dictation mode must stay completely unbiased.
    assert build_transcribe_kwargs("dictation", hotwords_supported=True) == {}
    assert build_transcribe_kwargs("dictation", hotwords_supported=False) == {}


def test_command_mode_uses_hotwords_when_supported():
    kwargs = build_transcribe_kwargs("command", hotwords_supported=True)
    assert kwargs == {"hotwords": HOTWORDS}


def test_command_mode_falls_back_to_initial_prompt_when_unsupported():
    kwargs = build_transcribe_kwargs("command", hotwords_supported=False)
    assert kwargs == {"initial_prompt": HOTWORDS}


def test_command_mode_respects_the_disable_flag():
    # The A/B toggle: hotwords support is available, but the flag is off.
    kwargs = build_transcribe_kwargs(
        "command", hotwords_supported=True, hotwords_enabled=False
    )
    assert kwargs == {}
