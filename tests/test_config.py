from skillpath import config


def test_paths_live_inside_repo():
    for path in (config.DATA_DIR, config.RAW_DIR, config.PROCESSED_DIR, config.CHROMA_DIR, config.DB_PATH):
        assert config.ROOT_DIR in path.parents


def test_repo_root_is_detected():
    assert (config.ROOT_DIR / "pyproject.toml").exists()
