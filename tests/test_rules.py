import pandas as pd
import pytest

from skillpath.recsys.rules import RULES, gap_boost, load_rules, related_skills

TOY_RULES = pd.DataFrame({
    "antecedents": [frozenset({"Docker"}), frozenset({"Docker", "AWS"}), frozenset({"React"}), frozenset({"Docker"})],
    "consequent": ["Kubernetes", "Terraform", "TypeScript", "AWS"],
    "support": [0.1, 0.05, 0.08, 0.1], "confidence": [0.6, 0.5, 0.7, 0.4], "lift": [3.0, 2.5, 2.0, 1.6],
})


def test_related_skills_needs_all_antecedents_and_skips_known():
    rel = related_skills(["Docker", "AWS"], rules=TOY_RULES)
    assert rel["skill"].tolist() == ["Kubernetes", "Terraform"]     # React rule doesn't fire; AWS already known
    assert rel.iloc[0]["because"] == ["Docker"]
    assert related_skills(["Excel"], rules=TOY_RULES).empty


def test_role_rules_take_precedence_and_global_rules_fill_in():
    rules = pd.concat([TOY_RULES.assign(role_family="ALL"),
                       pd.DataFrame({"role_family": ["Cloud/DevOps Engineer"], "antecedents": [frozenset({"Docker"})],
                                     "consequent": ["Helm"], "support": [0.2], "confidence": [0.3], "lift": [1.2]})])
    rel = related_skills(["Docker", "AWS"], role_family="Cloud/DevOps Engineer", rules=rules)
    assert rel.iloc[0]["skill"] == "Helm" and rel.iloc[0]["scope"] == "Cloud/DevOps Engineer"
    assert set(rel["skill"]) == {"Helm", "Kubernetes", "Terraform"}


@pytest.mark.skipif(not RULES.exists(), reason="run `python -m skillpath.recsys.rules` first")
def test_real_rules_are_sane():
    rules = load_rules()
    glob, role = rules[rules["role_family"] == "ALL"], rules[rules["role_family"] != "ALL"]
    assert len(glob) > 50 and role["role_family"].nunique() == 9
    assert (glob["lift"] >= 1.5).all() and (role["lift"] >= 1.1).all() and (rules["confidence"] >= 0.4).all()
    boost = gap_boost(["Docker", "AWS"])
    assert boost and all(0 < v <= 1 for v in boost.values())
    ml = related_skills(["Machine Learning", "Python"], role_family="AI/ML Engineer")["skill"].tolist()
    assert {"PyTorch", "TensorFlow"} <= set(ml)
