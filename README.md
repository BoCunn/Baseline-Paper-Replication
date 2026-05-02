# 20 Newsgroups Text Classification Experiment

A beginner-friendly Python script that trains a Multinomial Naive Bayes classifier on the 20 Newsgroups dataset, runs multiple experiments to measure variance, and compares results to a published baseline.

---

## Requirements

- Python 3.7+
- `scikit-learn`
- `numpy`
- `matplotlib`

Install dependencies with:

```bash
pip install scikit-learn numpy matplotlib
```

---

## Usage

```bash
python main.py
```

The dataset (~14 MB) is downloaded automatically on the first run and cached locally by sklearn.

**Estimated runtime:** 30–90 seconds on first run, 20–40 seconds after that.

---

## What It Does

1. **Loads** a 5-category subset of the 20 Newsgroups dataset (`rec.sport.hockey`, `sci.space`, `talk.politics.guns`, `comp.graphics`, `rec.autos`)
2. **Preprocesses** text using TF-IDF with English stopwords removed
3. **Trains** a Multinomial Naive Bayes classifier across 5 independent runs, each with a different random train/test split
4. **Reports** mean accuracy, standard deviation, and a per-run results table
5. **Plots** accuracy across runs and saves the chart as `accuracy_plot.png`
6. **Compares** results to a ~0.83 reference baseline from the literature
7. **Prints** a discussion of why results vary and how preprocessing choices affect reproducibility

---

## Output

- Console: statistics, results table, paper comparison, and discussion
- File: `accuracy_plot.png` — line plot of accuracy per run with mean ± 1 std bands