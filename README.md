# Project Overview

Online reviews play a crucial role in shaping public perception of local businesses. However, the presence of Irrelevant, promotional, or low-quality reviews may distort the true reputation of a place. This project was developed for the **TechJam Hackathon** challenge on assessing the quality and relevancy of Google location reviews.

We built a classification pipeline that labels reviews into four categories:

- **Advertisement** (promotions, links, coupons, URLs)
- **Irrelevant Content** (not related to the business or never visited business)
- **Rant** (negative opinions to express anger or contempt with little or no constructive feedback)
- **Feedback** (genuine, relevant review: may be positive/negative)

Our approach combines:
- Zero-shot classification + rule-based overrides for initial labeling
- Feature engineering: TF-IDF (uni/bi-grams) + regex/numeric signals (URLs, promo words, “never been” phrases, punctuation, all-caps ratio, etc.)
- Machine learning models: Logistic Regression + Random Forest 
- Evaluation: precision, recall and F1-score

# Setup Instructions
**Clone the repository**
```bash
git clone https://github.com/Augeliua/TechJam_ReviewFiltering.git
cd TechJam_ReviewFiltering
```

**Create environment & install dependencies** 
```bash
conda create -n reviews python=3.11 -y
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
  - label (harmonised into the 4 classes).

**2. Run the notebook**
- Open notebook.ipynb.
- Execute cells step by step:
  - Imports and downloads
  - Load dataset and preprocess
  - Zero-shot + rule-based overrides
  - Feature extraction (TF-IDF + regex)
  - Model training and evaluation

**3. Evaluate models**
- Baseline Logistic Regression (TF-IDF 3000) → Accuracy = 91.0%
- Logistic Regression (TF-IDF 30k + regex features) → Accuracy = 95.4%
- Random Forest Baseline → Lower performance than logistic regression however we still chose it as its a more robust model.

**4. Make Predictions**
```bash
import joblib
vec_art = joblib.load("baseline_tfidf3k_logreg.joblib")
vectorizer_base, logreg_base = vec_art["vectorizer_base"], vec_art["logreg_base"]

X = vectorizer_base.transform(["Nice ambience!", "Use code SAVE10 at www.shop.com"])
print(logreg_base.predict(X))

```
#Team Member Contributions
| Name                                      | Contributions |
|-------------------------------------------|---------------|
| Fong Zhi Lum                              | Website + frontend + backend            |
| Fong Xin Yi                               | Website + frontend + backend              |
| Dela Cruz Dionieleth Angelina Carlos\     | Data Cleaning + EDA + Notebook Modelling              |
| Chua Ziying                               | Data Cleaning + EDA + Notebook Modelling              |
| Veda Ho Yong Qian                         | Data Cleaning + EDA + Notebook Modelling              |

