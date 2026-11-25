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

### Collaborative filtering check

None of the four datasets has user-level interactions such as ratings, enrollments or clicks per user. The courses dataset has only an aggregate rating per course. So collaborative filtering (**M6-04**) would need a separate interaction dataset. The core plan uses content-based and hybrid ranking instead.
