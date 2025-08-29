# Project Overview

Online reviews play a crucial role in shaping public perception of local businesses. However, the presence of irrelevant, misleading, or low-quality reviews can distort the true reputation of a place. This project was developed for a hackathon challenge on assessing the quality and relevancy of Google location reviews.

We built a classification pipeline that labels reviews into four categories:

- **Advertisement** (e.g., promotions, links, coupons)
- **Irrelevant Content** (not related to the business)
- **Review without Visit** (user admits they never visited)
- **Clean Review** (genuine, relevant review)

Our approach combines:
- Zero-shot classification + rule-based overrides for initial labeling
- Feature engineering: TF-IDF textual features (unigrams + bigrams) and custom regex-based metadata features (URLs, promo keywords, “never been” phrases, etc.)
- Machine learning models: Logistic Regression (baseline + rich model), Random Forest baseline
- Evaluation: precision, recall, F1-score, confusion matrix

# Setup Instructions
**Clone the repository**
```bash
git clone https://github.com/Augeliua/TechJam_ReviewFiltering.git
cd TechJam_ReviewFiltering
```

**Create environment & install dependencies** 
```bash
conda create -n reviews python=3.11
conda activate reviews
pip install -r requirements.txt
```
Required libraries:
- pandas, numpy, scikit-learn, scipy
- nltk (stopwords, wordnet)
- transformers, torch
- matplotlib, wordcloud
- joblib

**Download NLTK resources**
```bash
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```

# How to Reproduce Results
**1. Prepare the dataset**
- Place cleaned_dataset.csv in the project root.
- Ensure it contains:
  - text (review content)
  - label (ground truth; if not available, policy_final is generated).

**2. Run the notebook**
- Open notebook.ipynb.
- Execute cells step by step:
  - Imports and downloads
  - Load dataset and preprocess
  - Zero-shot + rule-based overrides
  - Feature extraction (TF-IDF + regex)
  - Model training and evaluation

**3. Evaluate models**
- Baseline Logistic Regression (TF-IDF 3000) → Accuracy = 92.1%
- Rich Logistic Regression (TF-IDF 30k + regex features) → Accuracy = 93.7%
- Random Forest Baseline → Lower performance than logistic regression

**4. Make Predictions**
```bash
import joblib
model = joblib.load("artifacts/logreg_rich.joblib")
preds = model.predict([
    "Great food and friendly staff!", 
    "Visit www.promo.com for discounts"
])
print(preds)
```
#Team Member Contributions

