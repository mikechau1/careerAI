import pytest

from skillpath.skills.build import map_onet_name
from skillpath.skills.seed import EXACT_ONLY
from skillpath.skills.taxonomy import SKILLS_CSV, Taxonomy, skill_key


@pytest.fixture(scope="module")
def tax():
    return Taxonomy.from_seed()


def test_skill_key():
    assert skill_key("  Problem-Solving Skills ") == "problem-solving"
    assert skill_key("Amazon Web Services (AWS)") == "amazon web services aws"
    assert skill_key("R&D") == "r and d"


@pytest.mark.parametrize("raw,canonical", [
    ("k8s", "Kubernetes"),
    ("Kubernetes", "Kubernetes"),
    ("ReactJS", "React"),
    ("react.js", "React"),
    ("Amazon Web Services (AWS)", "AWS"),
    ("golang", "Go"),
    ("CI/CD pipelines", "CI/CD"),
    ("Spring Boot", "Spring"),
    ("problemsolving skills", "Problem Solving"),          # hyphen dropped in the job data
    ("objectoriented programming", "Object-Oriented Programming"),
    ("Salesforce software", "Salesforce"),                  # generic suffix fallback
    ("Django framework", "Django"),
    ("c/c++", "C++"),
    ("ts/sci clearance", "Security Clearance"),
])
def test_alias_normalization(tax, raw, canonical):
    assert tax.normalize(raw) == canonical


@pytest.mark.parametrize("raw", ["basket weaving", "bachelor's degree", "paid time off", ""])
def test_unknown_terms_return_none(tax, raw):
    assert tax.normalize(raw) is None


def test_normalize_list_dedupes_and_drops_unknowns(tax):
    assert tax.normalize_list(["Python", "python3", "PYTHON programming", "SQL", "paid time off"]) == ["Python", "SQL"]


def test_duplicate_alias_is_rejected():
    with pytest.raises(ValueError, match="claimed by both"):
        Taxonomy.from_records([("Go", "Languages", ["golang"]), ("Golang Tools", "Tools", ["golang"])])


def test_exact_only_aliases_exist_and_are_excluded_from_prose(tax):
    assert EXACT_ONLY <= set(tax.index), f"unknown EXACT_ONLY entries: {EXACT_ONLY - set(tax.index)}"
    assert "r" not in tax.prose_aliases("R")
    assert "r programming" in tax.prose_aliases("R")


def test_soft_skills_are_not_technical(tax):
    assert tax.is_technical("Docker")
    assert not tax.is_technical("Communication")


@pytest.mark.parametrize("onet_name,canonical", [
    ("Amazon Web Services AWS software", "AWS"),
    ("Structured query language SQL", "SQL"),
    ("Oracle Java", "Java"),
    ("Atlassian JIRA", "Jira"),
    ("Border Gateway Protocol BGP", "Routing and Switching"),
    ("Microsoft Azure Data Factory", "Azure Data Factory"),
    # look-alikes that must NOT map
    ("Apache Hadoop YARN", None),
    ("Schlumberger ECLIPSE", None),
    ("Corel Presentation", None),
    ("Pathogen Modeling Program PMP", None),
])
def test_onet_name_mapping(tax, onet_name, canonical):
    assert map_onet_name(tax, onet_name) == canonical


@pytest.mark.skipif(not SKILLS_CSV.exists(), reason="run `python -m skillpath.skills.build` first")
def test_built_csv_matches_seed():
    built, seed = Taxonomy.from_csv(), Taxonomy.from_seed()
    assert set(built.skills) == set(seed.skills)
    assert len(built.skills) >= 300
    for skill in seed.skills:  # the CSV only adds aliases, never drops seed ones
        assert set(seed.aliases_of[skill]) <= set(built.aliases_of[skill])
