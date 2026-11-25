"""Find canonical skills mentioned in free text (resumes, job descriptions, course pages).

    from skillpath.skills.extract import SkillExtractor
    SkillExtractor().extract("Built REST services in Python and deployed them on k8s")
    # -> ['Python', 'Kubernetes']        ("REST" is EXACT_ONLY, so prose won't match it)

How it works: the text is tokenized, and at each position the longest run of up to
MAX_NGRAM tokens that matches a prose-safe alias wins. This is a dict lookup per n-gram,
fast enough for all 47k job descriptions, and unlike a giant regex alternation it doesn't
backtrack. Tokens joined by "/" or "-" ("python/django", "python-based") are also tried
piece by piece.
"""

import re
from collections import Counter

from skillpath.skills.taxonomy import Taxonomy, load_taxonomy

MAX_NGRAM = 6
_TOKEN = re.compile(r"\.?[a-z0-9+#][a-z0-9+#./\-]*")


def tokenize(text: str) -> list[str]:
    text = (text or "").lower().replace("&", " and ")
    return [t.rstrip(".-/") for t in _TOKEN.findall(text)]


class SkillExtractor:
    def __init__(self, taxonomy: Taxonomy | None = None):
        tax = taxonomy or load_taxonomy()
        self.taxonomy = tax
        self.index = {alias: skill for skill in tax.skills for alias in tax.prose_aliases(skill)}
        self.max_n = min(MAX_NGRAM, max(len(a.split()) for a in self.index))

    def _match_at(self, tokens: list[str], i: int) -> tuple[str | None, int]:
        for n in range(min(self.max_n, len(tokens) - i), 0, -1):
            skill = self.index.get(" ".join(tokens[i:i + n]))
            if skill:
                return skill, n
        return None, 1

    def extract_counts(self, text: str) -> Counter:
        """How often each canonical skill is mentioned."""
        tokens, counts, i = tokenize(text), Counter(), 0
        while i < len(tokens):
            skill, n = self._match_at(tokens, i)
            if skill:
                counts[skill] += 1
            elif re.search(r"[/-]", tokens[i]):  # "python/django" -> try "python", "django"
                for part in re.split(r"[/-]", tokens[i]):
                    if part in self.index:
                        counts[self.index[part]] += 1
            i += n
        return counts

    def extract(self, text: str, technical_only: bool = False) -> list[str]:
        """Distinct canonical skills in order of first mention."""
        skills = list(self.extract_counts(text))
        return [s for s in skills if self.taxonomy.is_technical(s)] if technical_only else skills
