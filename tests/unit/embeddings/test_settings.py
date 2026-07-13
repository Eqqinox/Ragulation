from rag_flagship.embeddings.settings import EmbeddingSettings


def test_defaults_point_at_local_ollama() -> None:
    settings = EmbeddingSettings(_env_file=None)

    assert settings.base_url == "http://localhost:11434"
    assert settings.dense_model_name == "bge-m3"


def test_settings_are_overridable_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.invalid:11434")
    monkeypatch.setenv("OLLAMA_DENSE_MODEL_NAME", "some-other-model")

    settings = EmbeddingSettings(_env_file=None)

    assert settings.base_url == "http://example.invalid:11434"
    assert settings.dense_model_name == "some-other-model"
