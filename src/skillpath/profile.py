"""The user profile that every recommender takes as input."""

from dataclasses import dataclass, field


@dataclass
class Profile:
    text: str = ""                                   # resume or free-text description of the user
    skills: list[str] = field(default_factory=list)  # canonical skill names (see skills.taxonomy)
    target_role: str | None = None                   # role family, e.g. "Data Engineer" (see data.roles)
    title: str | None = None                         # current or most recent job title / resume headline
    highest_degree: str | None = None                # "PhD", "Master's", "Bachelor's", "Associate"
    years_experience: float | None = None
    skill_levels: dict[str, str] = field(default_factory=dict)  # optional: skill -> beginner/intermediate/advanced
    profile_id: str | None = None

    def query_text(self, max_chars: int = 1500) -> str:
        """A compact text form of the profile for retrieval models: role, skills, then resume text."""
        parts = []
        if self.target_role:
            parts.append(f"Target role: {self.target_role}.")
        if self.title:
            parts.append(f"Title: {self.title}.")
        if self.skills:
            parts.append("Skills: " + ", ".join(self.skills) + ".")
        parts.append(self.text[:max_chars])
        return " ".join(p for p in parts if p)
