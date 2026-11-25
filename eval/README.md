# Evaluation set

Fixed test profiles and relevance labels, so that every model is scored on exactly the same inputs.

| File | Contents |
|---|---|
| `profiles.json` | 18 test profiles: 10 real tech resumes (Kaggle resume dataset, CC0) and 8 synthetic personas, covering all 9 role families. Built by `python -m skillpath.eval.profiles`. |
| `qrels.csv` | Relevance labels, `profile_id, job_id, grade (0-3), source`. Written by the harness for every job in the pooled results. |
| `qrels_manual.csv` | *(optional)* Hand labels, `profile_id, job_id, grade`. They override the heuristic labels. |
| `hybrid_weights.json` | Model 3 weights tuned on this set (M2-07). |

## Running it

```bash
python -m skillpath.eval.harness --models tfidf semantic hybrid hybrid-norole
```

For each profile, every model returns its top 50 jobs. The union of those lists is the **pool**. Each job in the pool gets a grade, and the harness reports P@K, R@K, F1@K and NDCG@K for K = 5 and 10.
- **No role filter** is passed to the models.
- **Recall** is measured against the relevant jobs in the pool (TREC-style pooling), not against all 47k jobs.

## How labels are made (`skillpath/eval/judge.py`)

```
grade = role relation + skill fit                      (0-3)
  role relation: 2 = job's role family is the target role, 1 = adjacent family, 0 = unrelated
  skill fit:     +1 if the profile has at least 30% of the job's employer-listed technical skills
relevant (binary metrics) = grade >= 2
```

- The role family comes from the job **title** (M1-04), and fit uses the employer's **listed** skills.
- Models 1 and 2 rank on the full job text, so these labels don't just reward what a model looks at.
- Model 3 uses skill overlap and the target role as features. Read its scores next to the `hybrid-norole` ablation.

**These are proxy labels, not human judgments.** Before the final report, someone should hand-check a sample:
1. Open `qrels.csv`.
2. Copy any rows you disagree with into `qrels_manual.csv`, with the corrected grade.
3. Rerun the harness. Manual labels take precedence.
