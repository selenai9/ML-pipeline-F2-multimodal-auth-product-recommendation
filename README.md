# ML Pipeline F2: Multimodal Authentication and Product Recommendation

Formative 2 group assignment. The system verifies a user with their face and voice before showing a predicted product recommendation.

## Flow

1. Face recognition checks the user. If it fails, access is denied.
2. The product recommendation model predicts the product the customer would buy.
3. Voice verification confirms the transaction. If it fails, access is denied.
4. The predicted product is displayed.

## Repository structure

```
data/       raw spreadsheets and merged_dataset.csv
models/     saved product model and its input columns
scripts/    product_model.py
formative2_multimodal.ipynb   data merge, EDA, feature engineering and product model
```

Face and voice parts (images, audio, feature files, models and the command-line app) will be added by the other team members.

## Data merge

- `customer_social_profiles.xlsx` and `customer_transactions.xlsx` are joined on a shared `customer_id`.
- Assumption: the number in the social ID (for example `A178`) is the same customer as the number in the transaction ID (`178`).
- Cleaning: 5 duplicate social rows removed, 10 missing ratings filled with the median, social profiles reduced to one row per customer, and 33 customers without a social profile filled with median scores and "Unknown", with a `has_social_profile` column added.
- Validation: row count, duplicate transaction IDs and a spot check on one customer.

## Features

Date features (month, day of week, weekend), one-hot encoded platform and sentiment, plus the original number columns.

## Product model

Shallow Random Forest. Results on the 30 test rows:

| Metric | Score |
|---|---|
| Accuracy | 0.200 |
| F1-score | 0.174 |
| Loss | 1.675 |

Five-fold cross-validation accuracy was about 0.25 for the forests, against 0.20 for random guessing. The relationships between the features and the product category are weak and the dataset is small (150 rows), so performance is close to chance.

## How to run

Install the packages:

```
pip install pandas openpyxl scikit-learn matplotlib joblib
```

Train and save the product model from the repo's main folder:

```
python scripts/product_model.py
```

## Team

- Selena ISIMBI: data merge, EDA, feature engineering, product recommendation model
  
-Joshua Agonzibwa RUSOKE: image data and facial recognition model

- Celine SHOGA: audio data and voiceprint model
