import pytest

from skillpath.profile import Profile
from skillpath.recsys import skill_gap as sg
from skillpath.recsys.base import JOBS


@pytest.fixture
def patched(monkeypatch, toy_jobs):
    monkeypatch.setattr(sg, "load_jobs", lambda: toy_jobs)
    sg.role_requirements.cache_clear()
    yield
    sg.role_requirements.cache_clear()


def test_role_requirements_shares_exclude_soft_skills(patched):
    req = sg.role_requirements("Data Engineer").set_index("skill")["share"]
    assert req["Python"] == 1.0                  # in all 3 Data Engineer jobs
    assert req["Apache Spark"] == pytest.approx(2 / 3, abs=1e-3)
    assert "Communication" not in req.index      # soft skill


def test_skill_gap_statuses_and_priority(patched):
    p = Profile(skills=["Python", "SQL"], skill_levels={"SQL": "beginner"})
    gap = sg.skill_gap(p, "Data Engineer", min_share=0).set_index("skill")
    assert gap.loc["Python", "status"] == "Matched" and gap.loc["Python", "priority"] == 0
    assert gap.loc["SQL", "status"] == "Improve"
    assert gap.loc["Apache Spark", "status"] == "Missing"
    boosted = sg.skill_gap(p, "Data Engineer", min_share=0, boost={"Snowflake": 1.0}).set_index("skill")
    assert boosted.loc["Snowflake", "priority"] == pytest.approx(2 * boosted.loc["Snowflake", "share"], abs=1e-3)


def test_prioritized_gaps_and_match_rate(patched):
    p = Profile(skills=["Python"])
    gaps = sg.prioritized_gaps(p, "Data Engineer", min_share=0)
    assert gaps["status"].isin(["Missing", "Improve"]).all()
    assert gaps.iloc[0]["skill"] in ("Apache Spark", "SQL")   # 2/3 share, the highest after Python
    assert 0 < sg.match_rate(p, "Data Engineer", min_share=0) < 1


def test_skill_gap_needs_a_role(patched):
    with pytest.raises(ValueError):
        sg.skill_gap(Profile(skills=["Python"]))


@pytest.mark.skipif(not JOBS.exists(), reason="processed jobs not built")
def test_real_skill_gap_for_data_engineer():
    sg.role_requirements.cache_clear()
    gaps = sg.prioritized_gaps(Profile(skills=["Python", "SQL"]), "Data Engineer")
    assert len(gaps) > 3 and "Python" not in gaps["skill"].tolist()
