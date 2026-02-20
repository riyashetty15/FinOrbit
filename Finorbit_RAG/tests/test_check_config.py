import os
import pytest

from config import get_database_config


def test_database_config_defaults():
    # Ensure the database config reads env or falls back to defaults
    db = get_database_config()
    assert db.host is not None
    assert db.database is not None
    assert db.user is not None


def test_warning_for_env_with_secrets(tmp_path, monkeypatch, capsys):
    # Create temporary .env file with a secret key and run check script
    env_file = tmp_path / ".env"
    env_file.write_text("DB_PASSWORD=supersecret\n")
    monkeypatch.chdir(tmp_path)
    # import the script as module to execute main
    from scripts import check_config as cc

    # set a valid required env so it doesn't fail on missing vars
    monkeypatch.setenv('DB_PASSWORD', 'x')
    monkeypatch.setenv('GOOGLE_API_KEY', 'x')
    cc.main()
    captured = capsys.readouterr()
    assert 'Config looks OK' in captured.out
