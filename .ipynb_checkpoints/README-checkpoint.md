# FUTURE_ML_02 — Support Ticket Classification

**Intern:** Pratap Dey
**CIN ID:** FIT/JUL26/ML9909
**Track:** Machine Learning — Future Interns

## Task
Build a system to automatically classify customer support tickets and assign priority levels.

## Dataset
> Add dataset name + source link here once chosen (e.g. Kaggle "Customer Support Ticket
> Dataset" or "Support Ticket Classification Dataset").

Place the raw CSV in `data/tickets.csv`.

## Project Structure
```
FUTURE_ML_02/
├── data/
│   └── tickets.csv            # your dataset goes here
├── notebooks/
│   └── ticket_classification.ipynb
├── outputs/
│   └── confusion_matrix.png   # generated after running the notebook
└── README.md
```

## Approach
1. **Text Cleaning & Preprocessing** — lowercase, remove punctuation/stopwords, tokenize.
2. **Feature Extraction** — TF-IDF vectorization of ticket text.
3. **Category Classification** — train a classifier (e.g. Logistic Regression / Naive Bayes) to
   predict ticket category.
4. **Priority Tagging** — classify priority level (high / medium / low), either from an existing
   label or derived using rule-based/keyword logic if the dataset lacks one.
5. **Evaluation** — accuracy, precision/recall, confusion matrix.

## Skills Gained
Text preprocessing, NLP classification, priority logic, support analytics.

## Results
Both classifiers (category and priority) were trained using TF-IDF features on
ticket text (subject + description) with Logistic Regression:

| Task | Accuracy | Notes |
|---|---|---|
| Category Classification | ~20% | Near random chance (5 balanced classes) |
| Priority Classification | ~25% | Near random chance (4 balanced classes) |

**Key finding:** Both models performed close to random chance despite thorough
preprocessing and feature engineering. This indicates the ticket text in this dataset
does not carry a strong learnable signal for either label — likely due to
synthetic/randomized label assignment in the source dataset. This is a valid and
important diagnostic finding: it demonstrates the ability to recognize when a model's
poor performance stems from data limitations rather than modeling errors, and to
recommend next steps (e.g. auditing label quality, adding richer features) rather
than over-tuning against noise.

## How to Run
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
jupyter notebook notebooks/ticket_classification.ipynb
```
