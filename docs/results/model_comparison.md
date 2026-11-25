# Model comparison (M2-07, preliminary)

Offline evaluation on the 18 fixed profiles in `eval/profiles.json` (10 real tech resumes + 8 personas, all 9 role families). Labels come from the heuristic judge in `skillpath/eval/judge.py`: role relation (title-based family) + skill fit on employer-listed skills. Recall is pooled across all models' top 50. No role filter is passed to the models.

Reproduce:

```bash
python -m skillpath.eval.harness --models tfidf semantic hybrid-text hybrid-norole hybrid
```

## Results

| Model | P@5 | P@10 | R@10 | F1@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| 1. TF-IDF + cosine | 0.889 | 0.900 | 0.068 | 0.127 | 0.749 | 0.737 |
| 2. Embeddings + ChromaDB | 0.967 | 0.972 | 0.074 | 0.138 | 0.810 | 0.793 |
| 3a. Hybrid, text signals only (TF-IDF 0.5 + semantic 1.0) | 0.944 | 0.950 | 0.072 | 0.134 | 0.817 | **0.820** |
| 3b. Hybrid without the role feature | 0.956 | 0.956 | 0.073 | 0.135 | 0.905 | 0.877 |
| 3c. Hybrid, all features (app default) | 1.000 | 1.000 | 0.076 | 0.142 | 0.990 | 0.991 |

NDCG@10 by target role:

| Role | TF-IDF | Semantic | Hybrid text | Hybrid no-role | Hybrid |
|---|---:|---:|---:|---:|---:|
| AI/ML Engineer | 0.959 | 0.981 | 0.981 | 0.953 | 1.000 |
| Cloud/DevOps Engineer | 0.686 | 0.780 | 0.935 | 0.773 | 1.000 |
| Data Analyst | 0.623 | 0.692 | 0.688 | 0.836 | 1.000 |
| Data Engineer | 0.656 | 0.785 | 0.702 | 0.804 | 1.000 |
| Data Scientist | 0.726 | 0.708 | 0.811 | 0.621 | 0.951 |
| IT/Systems | 0.540 | 0.590 | 0.477 | 0.905 | 0.948 |
| QA/Test Engineer | 0.576 | 0.623 | 0.792 | 0.901 | 1.000 |
| Security Engineer | 0.750 | 0.903 | 0.899 | 0.947 | 1.000 |
| Software Engineer | 0.985 | 0.954 | 1.000 | 0.978 | 1.000 |

## How to read these numbers

1. **Semantic beats TF-IDF:** NDCG@10 0.793 vs 0.737, and P@10 0.972 vs 0.900. Neither model shares a signal with the judge, so this comparison is clean.
2. **The text-only hybrid (3a) is the fair test of "hybrid > single model".** It shares no feature with the judge and improves NDCG@10 over semantic alone (0.820 vs 0.793), with slightly lower P@10. The gain is uneven by role: large for Cloud/DevOps and QA, negative for Data Engineer and IT/Systems.
3. **3b and 3c are inflated by construction.** Model 3's skill-overlap feature is close to the judge's skill-fit signal, and its role feature *is* the judge's role signal. Their scores show the features work as designed. They are **not** evidence of better recommendations. Don't present 3c's 0.99 as a headline result.
4. **Recall is low for every model** (≈0.07 at K = 10) because the pool holds about 100–140 relevant jobs per profile. It's only comparable *between* models.

## Before the final report (M4-01)

- **Hand-label a sample of the pool.** About 10 jobs per profile, copied into `eval/qrels_manual.csv`, which overrides the heuristic labels. Then re-run the harness and re-tune (`python -m skillpath.recsys.hybrid --tune`). Human labels break the circularity for 3b and 3c.
- **Expect instability.** 18 profiles is a small set, so per-role numbers rest on 1–2 profiles each.
