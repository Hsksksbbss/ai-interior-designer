from app.ai_generator import format_openai_error, get_api_key


def test_get_api_key_uses_gemini_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert get_api_key() == "gemini-test-key"


def test_format_openai_error_for_insufficient_quota():
    error = RuntimeError("Error code: 429 - {'error': {'code': 'credit_balance_exhausted'}}")

    message = format_openai_error(error)

    assert "credits" in message.lower()
    assert "quota" in message.lower()


def test_format_openai_error_for_invalid_key():
    error = RuntimeError("Incorrect API key provided")

    message = format_openai_error(error)

    assert "api key" in message.lower()
