# Project Overview

Online reviews play a crucial role in shaping public perception of local businesses. However, the presence of irrelevant, misleading, or low-quality reviews can distort the true reputation of a place.

We built a classification pipeline that labels reviews into four categories:

- Advertisement (e.g., promotions, links, coupons)
- Irrelevant Content (not related to the business)
- Review without Visit (user admits they never visited)
- Clean Review (genuine, relevant review)

Our approach combines:
- Zero-shot classification + rule-based overrides for initial labeling
- Feature engineering: TF-IDF textual features (unigrams + bigrams) and custom regex-based metadata features (URLs, promo keywords, “never been” phrases, etc.)
- Machine learning models: Logistic Regression (baseline + rich model), Random Forest baseline
- Evaluation: precision, recall, F1-score, confusion matrix
