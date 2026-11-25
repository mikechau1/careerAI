# Data

Raw and processed data files are git-ignored. Each teammate downloads them locally into `data/raw/`.

| Folder | Contents |
|---|---|
| `raw/` | Downloaded datasets, unmodified |
| `processed/` | Cleaned outputs, e.g. `jobs.parquet` (M1-04) |
| `chroma/` | Vector store, built by the indexing script (M2-02) |
| `uploads/` | Resumes uploaded in the app, kept locally only |

## Downloading the data

1. **Sign in to Kaggle (one time).** You need a Kaggle account. With the venv active, run:

   ```bash
   kaggle auth login
   ```

   This opens your browser to sign in and caches the credentials locally.

   **Alternative:** generate a token at https://www.kaggle.com/settings/api (**Generate New Token**). Then either save it to `~/.kaggle/access_token` or set it as the `KAGGLE_API_TOKEN` environment variable. Never commit the token.
2. **Run the download script** from the repo root with the venv active:

   ```bash
   python scripts/download_data.py --list   # check file sizes first
   python scripts/download_data.py          # download everything into data/raw/
   ```

## Sources

| Key | Dataset | License | Use | Folder |
|---|---|---|---|---|
| `jobs` | [1.3M LinkedIn Jobs & Skills (2024)](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) | ODC-By | Job descriptions, plus skills already extracted per posting. Feeds retrieval, skill gaps and the skill association rules. | `raw/jobs/` |
| `courses` | [Coursera Courses Dataset 2021](https://www.kaggle.com/datasets/khusheekapoor/coursera-courses-dataset-2021) | Not stated on Kaggle | Course recommendations | `raw/courses/` |
| `resumes` | [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) | CC0 | Test profiles for evaluation (M2-04) | `raw/resumes/` |
| `onet` | [O*NET 30.2 Database](https://www.onetcenter.org/database.html) | CC BY 4.0 | Skill list (`Technology Skills.txt`) and occupations | `raw/onet/` |

### What's in each dataset

**jobs** (`raw/jobs/`, 6.2 GB uncompressed). All three files are keyed by `job_link`.

| File | Size | Contents |
|---|---|---|
| `linkedin_job_postings.csv` | 415 MB | Title, company, location, level, type, first-seen date |
| `job_skills.csv` | 673 MB | Comma-separated list of extracted skills per posting |
| `job_summary.csv` | 5.1 GB | Full description text |

Row counts: 1,348,454 postings; 1,296,381 with skills; 1,297,332 with a description. All postings were first seen between 2024-01-12 and 2024-01-17. That window is too short for trend forecasting, so **M6-02** (ARIMA) would need another data source. The postings cover every industry. A rough title regex matches about 114k as tech. M1-04 filters them to tech roles properly and writes a small `processed/jobs.parquet`. M1-04 filters them to tech roles and writes a small `processed/jobs.parquet`, so teammates can share that file instead of each downloading 6 GB. Read the raw CSVs in chunks (`pd.read_csv(..., chunksize=...)`).

**courses** (`raw/courses/Coursera.csv`): 3,522 courses, every one with a description and skills.
- Columns: `Course Name`, `University`, `Difficulty Level`, `Course Rating`, `Course URL`, `Course Description`, `Skills`.
- **Caveat:** `Skills` is space-separated, so multi-word skills run together (e.g. `Drama  Comedy  creative writing`). M1-07 should match skills against the skill list rather than split on spaces.

**resumes** (`raw/resumes/`): 2,484 resumes in 24 categories.
- `Resume/Resume.csv` has the columns `ID`, `Resume_str`, `Resume_html`, `Category`. The original PDFs are in `data/data/<CATEGORY>/`.
- The tech-relevant categories are INFORMATION-TECHNOLOGY (120) and ENGINEERING (118).

**onet** (`raw/onet/db_30_2_text/`): 41 tab-separated files.
- The ones we use are `Technology Skills.txt` (32,773 rows; 8,785 unique technologies, 171 flagged "Hot Technology"), `Occupation Data.txt` (1,016 occupations) and `Alternate Titles.txt`.

## Processed data (tech only)

Build these after downloading:

```bash
python -m skillpath.data.preprocess_jobs      # about 3 min, reads the 5 GB description file
python -m skillpath.data.preprocess_resumes   # a few seconds
python -m skillpath.skills.annotate           # about 3 min: canonical skills for jobs and courses (M1-07)
```

Once these are built, teammates only need the two small parquet files, not the 6 GB raw jobs data.

**`processed/jobs.parquet`**: 47,431 tech jobs, 115 MB.
- Columns: `job_id`, `title`, `role_family`, `company`, `location`, `country`, `level`, `job_type`, `first_seen`, `skills` (list), `description`, `job_link`.
- How it's built:
  1. Of 1.35M postings, keep the 61,240 whose title maps to a tech role family ([roles.py](../src/skillpath/data/roles.py)).
  2. Keep those with a description of at least 200 characters.
  3. Remove 11,001 reposts (same company, title and description).
- 99.9% have skills (median 23 per job). The median description is 3,684 characters.

| Role family | Jobs |
|---|---:|
| Software Engineer | 17,548 |
| IT/Systems | 12,830 |
| Cloud/DevOps Engineer | 3,685 |
| Security Engineer | 3,540 |
| QA/Test Engineer | 3,073 |
| Data Analyst | 2,492 |
| Data Engineer | 2,093 |
| AI/ML Engineer | 1,374 |
| Data Scientist | 796 |

**Canonical skills on jobs (M1-07).** `skillpath.skills.annotate` adds three columns:

| Column | Contents | Jobs with at least 3 skills | Median |
|---|---|---:|---:|
| `skills_listed` | The employer's listed skills, mapped to canonical names | 90.7% | 9 |
| `skills_text` | Canonical skills found in the title and description | 94.0% | 11 |
| `skills_canonical` | Both combined. **The recommenders use this one.** | 97.2% | 13 |

**`processed/courses.parquet`**: 3,424 Coursera courses after de-duplication.
- Columns: `course_id`, `title`, `provider`, `difficulty`, `rating`, `url`, `skills` (canonical), `is_tech`, `description`, `skills_raw`.
- 1,945 courses are tech (`is_tech`: at least one technical skill). All of them have skills; 56% have 3 or more, median 3. The other courses are humanities, business and so on.

**`processed/resumes.parquet`**: 157 tech resumes, out of 2,484.
- Columns: `resume_id`, `category`, `title`, `role_family`, `tech_terms` (list), `text`.
- A resume is kept if it's in INFORMATION-TECHNOLOGY and mentions at least 2 core tech skills, or if it's in any other category and mentions at least 6. This catches software people filed under ENGINEERING, CONSULTANT and so on.
- Breakdown: 90 IT, 20 Engineering, 17 Consultant, 30 other.

## Skill taxonomy (`skills.csv`, committed)

347 canonical skills (331 technical, 16 soft) in 18 categories. Every part of the system should go through this file to name skills, so that "k8s", "Kubernetes" and "Red Hat OpenShift" count as the same skill.

```python
from skillpath.skills.taxonomy import load_taxonomy
tax = load_taxonomy()
tax.normalize("ReactJS")        # "React"
tax.normalize_list(job_skills)  # canonical, deduplicated, unknowns dropped
tax.prose_aliases("R")          # aliases safe to search for in free text
```

**Columns:** `skill`, `category`, `aliases` (pipe-separated), `job_count` / `job_share` (number and share of tech jobs that list the skill), `onet_hot` (O*NET "Hot Technology").

**How it's built:**
- The source of truth is the hand-curated [seed.py](../src/skillpath/skills/seed.py).
- `python -m skillpath.skills.build` adds 257 O*NET technology names as aliases (e.g. "Oracle Java" → Java, "Border Gateway Protocol BGP" → Routing and Switching) and counts the jobs for each skill.
- 304 skills appear in at least 100 of the 47k tech jobs. The most common are Python (24% of jobs), SQL (20%), Java (16%) and AWS (15%).

**Coverage:** 41% of raw skill mentions in the jobs map to a skill. Most of the unmapped mentions aren't skills: degrees ("bachelor's degree"), benefits ("paid time off"), citizenship and generic words ("innovation"). The build prints the top unmapped strings so they can be reviewed.

**To add a skill or alias:** edit `seed.py`, rerun the build, and commit both files. The build fails if two skills claim the same alias. `EXACT_ONLY` in `seed.py` lists ambiguous forms ("R", "Go", "AD", "rest"). Those match exact skill-list entries only, never free text.

### Collaborative filtering check

None of the four datasets has user-level interactions such as ratings, enrollments or clicks per user. The courses dataset has only an aggregate rating per course. So collaborative filtering (**M6-04**) would need a separate interaction dataset. The core plan uses content-based and hybrid ranking instead.
