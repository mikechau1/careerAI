"""M2-04: build the fixed evaluation profiles in eval/profiles.json.

    python -m skillpath.eval.profiles

Two sources:
1. Real resumes (processed/resumes.parquet, CC0). Kept if the headline maps to a role
   family, the resume has at least 6 technical skills, and the headline isn't an executive
   title. At most PER_FAMILY per family.
2. Synthetic personas (below). The livecareer resume dataset has almost no data, ML,
   cloud or security people, which are the roles this project targets.

Every profile goes through parse_resume, so skills are extracted the same way the app will
extract them from an upload. The JSON is committed, so all models are always scored on
exactly the same profiles.
"""

import json
import re

import pandas as pd

from skillpath.config import PROCESSED_DIR, ROOT_DIR
from skillpath.profile import Profile
from skillpath.resume import parse_resume
from skillpath.skills.taxonomy import load_taxonomy

PROFILES_JSON = ROOT_DIR / "eval" / "profiles.json"
PER_FAMILY = 2
MIN_TECH_SKILLS = 6
EXECUTIVE = r"vice president|\bvp\b|director|chief|\bcio\b|\bcto\b"  # not the job seekers this app is for

PERSONAS = [
    ("persona-mle-junior", "AI/ML Engineer", """MACHINE LEARNING ENGINEER (NEW GRAD)
M.S. Computer Engineering. Built a RAG chatbot with LangChain, OpenAI API embeddings and a ChromaDB
vector database; fine-tuned Hugging Face transformers in PyTorch for text classification (NLP).
Deployed models as FastAPI services in Docker. Python, SQL, scikit-learn, pandas, NumPy, Git, Linux.
Coursework: deep learning, recommender systems, statistics."""),
    ("persona-mle-senior", "AI/ML Engineer", """SENIOR MACHINE LEARNING ENGINEER
8 years of experience shipping machine learning systems. Led MLOps platform on AWS (SageMaker, Lambda,
S3) with Kubernetes and Terraform; CI/CD with GitHub Actions. Built ranking and recommender systems,
feature engineering pipelines in Apache Spark, model monitoring. Python, TensorFlow, PyTorch, XGBoost,
SQL, Apache Airflow, Docker. Mentoring, system design. M.S. Computer Science."""),
    ("persona-ds", "Data Scientist", """DATA SCIENTIST
4 years of experience in statistics and predictive modeling for e-commerce. A/B testing and
experimentation, time series forecasting, customer segmentation. Python (pandas, NumPy, scikit-learn,
XGBoost), R, SQL, Tableau, Jupyter. Communicated insights to stakeholders. M.S. Statistics."""),
    ("persona-de", "Data Engineer", """DATA ENGINEER
5 years of experience building ETL pipelines. Apache Spark, Apache Airflow, dbt, Snowflake, Apache
Kafka, AWS (S3, Redshift), Python, SQL, data modeling, data warehousing. Docker, Git, CI/CD.
B.S. Computer Science."""),
    ("persona-da", "Data Analyst", """DATA ANALYST
3 years of experience in reporting and business intelligence. SQL, Microsoft Excel, Power BI, Tableau,
data visualization, dashboards, Python (pandas). Requirements gathering with stakeholders, data
quality checks. B.A. Economics."""),
    ("persona-devops", "Cloud/DevOps Engineer", """DEVOPS ENGINEER
6 years of experience. Kubernetes, Docker, Terraform, Ansible, Jenkins, GitHub Actions, CI/CD,
AWS and Microsoft Azure, Linux administration, Bash and Python scripting, Prometheus and Grafana
monitoring, site reliability engineering, infrastructure as code. B.S. Information Technology."""),
    ("persona-sec", "Security Engineer", """SECURITY ANALYST
4 years of experience in a security operations center. SIEM (Splunk), incident response, vulnerability
management, penetration testing, network security, firewalls, identity and access management,
NIST and ISO 27001 compliance, Python and PowerShell scripting, Linux, Windows Server.
CompTIA Security+, CISSP. B.S. Cybersecurity."""),
    ("persona-frontend", "Software Engineer", """FRONTEND SOFTWARE ENGINEER
3 years of experience building web applications. JavaScript, TypeScript, React, Redux, Next.js,
HTML, CSS, Tailwind CSS, REST APIs, GraphQL, Node.js, Jest unit testing, Git, Agile/Scrum,
accessibility. B.S. Computer Science."""),
]


def resume_profiles(tax) -> list[Profile]:
    resumes = pd.read_parquet(PROCESSED_DIR / "resumes.parquet")
    picked, per_family, seen_titles = [], {}, set()
    for r in resumes.itertuples():
        p = parse_resume(r.text, profile_id=f"resume-{r.resume_id}")
        n_tech = sum(tax.is_technical(s) for s in p.skills)
        if re.search(EXECUTIVE, p.title or "", re.I) or p.title in seen_titles:  # the dataset has near-duplicates
            continue
        if p.target_role and n_tech >= MIN_TECH_SKILLS and per_family.get(p.target_role, 0) < PER_FAMILY:
            per_family[p.target_role] = per_family.get(p.target_role, 0) + 1
            seen_titles.add(p.title)
            picked.append(p)
    return picked


def build() -> list[Profile]:
    tax = load_taxonomy()
    personas = [parse_resume(text, target_role=role, profile_id=pid) for pid, role, text in PERSONAS]
    return resume_profiles(tax) + personas


def to_record(p: Profile) -> dict:
    return {"profile_id": p.profile_id, "target_role": p.target_role, "title": p.title,
            "highest_degree": p.highest_degree, "years_experience": p.years_experience,
            "skills": p.skills, "text": p.text}


def load_profiles() -> list[Profile]:
    records = json.loads(PROFILES_JSON.read_text(encoding="utf-8"))
    return [Profile(**r) for r in records]


def main():
    profiles = build()
    PROFILES_JSON.parent.mkdir(parents=True, exist_ok=True)
    PROFILES_JSON.write_text(json.dumps([to_record(p) for p in profiles], indent=2), encoding="utf-8")
    print(f"wrote {len(profiles)} profiles to {PROFILES_JSON}")
    for p in profiles:
        print(f"  {p.profile_id:28s} {p.target_role:22s} {len(p.skills):3d} skills  {p.title[:40] if p.title else ''}")


if __name__ == "__main__":
    main()
