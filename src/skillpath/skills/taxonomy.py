"""Skill taxonomy: map any skill string to its canonical skill.

    from skillpath.skills.taxonomy import load_taxonomy
    tax = load_taxonomy()
    tax.normalize("k8s")             # -> "Kubernetes"
    tax.normalize("ReactJS")         # -> "React"
    tax.normalize("Basket weaving")  # -> None
    tax.category("Kubernetes")       # -> "DevOps & Infrastructure"
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from skillpath.config import DATA_DIR
from skillpath.skills.seed import EXACT_ONLY, SEED

SKILLS_CSV = DATA_DIR / "skills.csv"
SOFT = "Soft Skills"

# Generic suffixes tried as a fallback: "Salesforce software" -> "salesforce", "Django framework" -> "django"
_SUFFIXES = (" software", " framework", " programming", " language", " platform", " tools", " tool")


def skill_key(s: str) -> str:
    """Normalize a skill string for lookup: lowercase, unified dashes and spaces, no 'skills' suffix."""
    s = str(s).lower().replace("&", " and ").replace("–", "-").replace("—", "-")
    s = re.sub(r"[()\[\]{},;:!?\"']", " ", s)       # brackets and stray punctuation -> space
    s = re.sub(r"\s+", " ", s).strip(" .-*/")
    s = re.sub(r"\s+skills?$", "", s)
    return s


@dataclass
class Taxonomy:
    category_of: dict[str, str]                     # canonical -> category
    aliases_of: dict[str, list[str]]                # canonical -> alias keys (canonical key included)
    index: dict[str, str] = field(default_factory=dict)  # alias key -> canonical

    def __post_init__(self):
        for skill, aliases in self.aliases_of.items():
            for key in aliases:
                owner = self.index.setdefault(key, skill)
                if owner != skill:
                    raise ValueError(f"alias {key!r} is claimed by both {owner!r} and {skill!r}")

    # --- construction ---------------------------------------------------------------
    @classmethod
    def from_records(cls, records: list[tuple[str, str, list[str]]]) -> "Taxonomy":
        category_of, aliases_of = {}, {}
        for skill, category, aliases in records:
            if skill in category_of:
                raise ValueError(f"duplicate skill {skill!r}")
            category_of[skill] = category
            keys = [skill_key(skill)] + [skill_key(a) for a in aliases]
            # job data often drops hyphens: "object-oriented" appears as "objectoriented"
            keys += [k.replace("-", "") for k in keys if "-" in k] + [k.replace("-", " ") for k in keys if "-" in k]
            aliases_of[skill] = list(dict.fromkeys(k for k in keys if k))
        return cls(category_of, aliases_of)

    @classmethod
    def from_seed(cls) -> "Taxonomy":
        return cls.from_records(parse_seed(SEED))

    @classmethod
    def from_csv(cls, path=SKILLS_CSV) -> "Taxonomy":
        df = pd.read_csv(path, keep_default_na=False)
        return cls.from_records([(r.skill, r.category, r.aliases.split("|") if r.aliases else [])
                                 for r in df.itertuples()])

    # --- lookup ---------------------------------------------------------------------
    def normalize(self, term: str) -> str | None:
        """Canonical skill for a skill string, or None if it is not in the taxonomy."""
        key = skill_key(term)
        if key in self.index:
            return self.index[key]
        for suffix in _SUFFIXES:
            if key.endswith(suffix) and key[: -len(suffix)] in self.index:
                return self.index[key[: -len(suffix)]]
        return None

    def normalize_list(self, terms) -> list[str]:
        """Canonical skills for a list of skill strings, deduplicated, order kept, unknowns dropped."""
        return list(dict.fromkeys(s for s in map(self.normalize, terms) if s))

    def category(self, skill: str) -> str | None:
        return self.category_of.get(skill)

    def is_technical(self, skill: str) -> bool:
        return self.category_of.get(skill) not in (None, SOFT)

    def prose_aliases(self, skill: str) -> list[str]:
        """Aliases safe to search for in free text (EXACT_ONLY ones removed)."""
        return [a for a in self.aliases_of[skill] if a not in EXACT_ONLY]

    @property
    def skills(self) -> list[str]:
        return list(self.category_of)


def parse_seed(seed: dict[str, str]) -> list[tuple[str, str, list[str]]]:
    records = []
    for category, block in seed.items():
        for line in block.strip().splitlines():
            name, _, aliases = line.partition("|")
            records.append((name.strip(), category, [a.strip() for a in aliases.split(",") if a.strip()]))
    return records


@lru_cache(maxsize=1)
def load_taxonomy() -> Taxonomy:
    """The built taxonomy (data/skills.csv, includes mined aliases), or the seed if not built yet."""
    return Taxonomy.from_csv() if SKILLS_CSV.exists() else Taxonomy.from_seed()
