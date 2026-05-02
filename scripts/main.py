
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#data loading

# 5 categories to keep runtime manageable
categories = [
    'rec.sport.hockey',
    'sci.space',
    'talk.politics.guns',
    'comp.graphics',
    'rec.autos'
]

print("Loading the 20 Newsgroups dataset...")
print(f"Categories used: {categories}\n")

# Load all data (train + test combined to do splits)
newsgroups = fetch_20newsgroups(
    subset='all',
    categories=categories,
    shuffle=True,
    random_state=42,   # reproducible shuffle of the full dataset
    remove=('headers', 'footers', 'quotes')  # remove metadata to avoid leakage
)

texts = newsgroups.data      # list of raw text strings
labels = newsgroups.target   # integer class labels

print(f"Total documents loaded: {len(texts)}")
print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}\n")



# TF-IDF converts raw text into numerical feature vectors.
# - stop_words='english' removes common words like "the", "is", etc.
# - max_features limits vocabulary size to keep things fast
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

# Fit and transform all documents into a TF-IDF matrix
X = vectorizer.fit_transform(texts)
y = labels

print(f"TF-IDF matrix shape: {X.shape}  (documents x features)\n")



NUM_RUNS = 5          # number of independent train/test split experiments
TEST_SIZE = 0.2       # 20% of data held out for testing each run
accuracies = []       # store accuracy from each run

print("=" * 50)
print("Running experiments...")
print("=" * 50)

for run in range(NUM_RUNS):
    # Each run uses a different random seed, giving a different train/test split
    # This simulates how results can vary due to randomness in data splitting
    random_seed = run * 10   # seeds: 0, 10, 20, 30, 40

    # Split the already-vectorized data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=random_seed
    )

    # Train Multinomial Naive Bayes on the training set
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predict labels for the test set
    y_pred = model.predict(X_test)

    # Compute accuracy (fraction of correctly classified documents)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"  Run {run + 1} | seed={random_seed:>2} | "
          f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]} | "
          f"Accuracy: {acc:.4f}")



mean_acc = np.mean(accuracies)
std_acc  = np.std(accuracies)

print()
print("=" * 50)
print("SUMMARY STATISTICS")
print("=" * 50)
print(f"  Mean Accuracy : {mean_acc:.4f}")
print(f"  Std Deviation : {std_acc:.4f}")
print("=" * 50)



print()
print("RESULTS TABLE")
print("-" * 30)
print(f"{'Run':<8} {'Accuracy':>10}")
print("-" * 30)
for i, acc in enumerate(accuracies, start=1):
    print(f"{i:<8} {acc:>10.4f}")
print("-" * 30)
print(f"{'Mean':<8} {mean_acc:>10.4f}")
print(f"{'Std':<8} {std_acc:>10.4f}")
print("-" * 30)

# graphs

run_numbers = list(range(1, NUM_RUNS + 1))

plt.figure(figsize=(8, 5))
plt.plot(run_numbers, accuracies, marker='o', linewidth=2,
         color='steelblue', label='Run Accuracy')

# Draw horizontal lines for mean and ± 1 std deviation
plt.axhline(y=mean_acc, color='orange', linestyle='--',
            linewidth=1.5, label=f'Mean ({mean_acc:.4f})')
plt.axhline(y=mean_acc + std_acc, color='gray', linestyle=':',
            linewidth=1, label=f'Mean ± 1 Std')
plt.axhline(y=mean_acc - std_acc, color='gray', linestyle=':', linewidth=1)

plt.title('Naive Bayes Accuracy Across Runs\n(20 Newsgroups, 5 Categories)')
plt.xlabel('Run Number')
plt.ylabel('Accuracy')
plt.xticks(run_numbers)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_plot.png', dpi=150)
plt.show()

print("\n[Plot saved as accuracy_plot.png]")

# ---------------------------------------------------------------
# 8. PAPER REPRODUCTION ELEMENT
# ---------------------------------------------------------------

# We compare our results to a commonly cited baseline.
# Research on Naive Bayes text classification on 20 Newsgroups
# (e.g., Rennie et al., 2003; McCallum & Nigam, 1998) typically
# reports accuracy around 0.83 on the full 20-category dataset.
# Our subset of 5 categories is expected to be somewhat easier.

PAPER_ACCURACY = 0.83   # assumed reference accuracy from the literature

print()
print("=" * 50)
print("PAPER REPRODUCTION COMPARISON")
print("=" * 50)
print(f"  Reference ('paper') accuracy : {PAPER_ACCURACY:.4f}")
print(f"  Our mean accuracy            : {mean_acc:.4f}")
difference = mean_acc - PAPER_ACCURACY
direction  = "above" if difference >= 0 else "below"
print(f"  Difference                   : {abs(difference):.4f} {direction} reference")
print("=" * 50)

# ---------------------------------------------------------------
# 9. DISCUSSION
# ---------------------------------------------------------------

discussion = """
DISCUSSION
----------
1. Why do results vary between runs?
   Each run uses a different random seed to split the data into
   training and test sets. Because the model only sees the training
   portion, a slightly different split can lead to different
   distributions of topics in train vs. test, which changes how
   well the model generalizes. This is a core source of variance
   in empirical ML experiments.

2. Why might our results differ from the reference paper?
   Several factors can cause differences:
   - Dataset scope: We use only 5 of the 20 categories, making
     the task easier (fewer classes to confuse).
   - Preprocessing: We remove headers, footers, and quoted text
     to reduce data leakage. Papers may not do this.
   - Feature choices: Our TF-IDF uses max 10,000 features with
     English stopwords removed. Different settings affect accuracy.
   - Train/test split ratio: Papers often use the official
     train/test split; we randomly split each time.
   - Smoothing: sklearn's MultinomialNB uses additive (Laplace)
     smoothing by default (alpha=1.0); papers may tune this.

3. What can we conclude about reproducibility?
   Even a "simple" experiment like Naive Bayes text classification
   is sensitive to preprocessing decisions, dataset scope, and
   random splits. This is why reporting mean ± std across multiple
   runs (rather than a single number) is important for honest
   scientific communication.
"""

print(discussion)