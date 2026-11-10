# ML Pipeline F2: Multimodal Authentication and Product Recommendation

Formative 2 group assignment. The system verifies a user with their face and voice before showing a predicted product recommendation.

## Flow

1. Face recognition checks the user. If it fails, access is denied.
2. The product recommendation model predicts the product the customer would buy.
3. Voice verification confirms the transaction. If it fails, access is denied.
4. The predicted product is displayed.

## Repository structure

```
audio/      voice clips, named <member>_<phrase> (e.g. selena_yes_approve.mp4)
data/       raw spreadsheets, merged_dataset.csv and audio_features.csv
models/     saved product model and its input columns, voice_model.pkl
scripts/    product_model.py, audio_pipeline.py, voice_model.py
formative2_multimodal.ipynb   data merge, EDA, feature engineering, product model, audio and voice model
```

The face part (image features, face model and the command-line app) will be added by the other team members.

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

## Audio and voiceprint model

- Each member records "Yes, approve" and "Confirm transaction". Clips are converted to mono 16 kHz, trimmed of silence and scaled to the same peak volume.
- Augmentations (5 per clip): pitch up and down 1.5 semitones, 20% faster and slower, background noise at 20 dB.
- Features (34 per row): mean and standard deviation of 13 MFCCs, spectral roll-off, RMS energy, spectral centroid and zero-crossing rate, saved to `data/audio_features.csv`.
- The model is tested leave-one-phrase-out: it is trained on one phrase and tested on the other, so it never hears the test clip.
- A voice is approved only if the model is at least 60% sure, the voice is close to that member's own recordings, and it matches the person the face model recognised.

## How to run

Install the packages:

```
pip install pandas openpyxl scikit-learn matplotlib joblib librosa soundfile imageio-ffmpeg
```

Train and save the product model from the repo's main folder:

```
python scripts/product_model.py
```

Extract audio features, then train and save the voice model:

```
python scripts/audio_pipeline.py
python scripts/voice_model.py
```

Check one voice clip (optionally against the member the face model recognised):

```
python scripts/voice_model.py --verify audio/selena_yes_approve.mp4 --member selena
```

## Team

- Selena ISIMBI: data merge, EDA, feature engineering, product recommendation model
  
- Joshua Agonzibwa RUSOKE: audio data, voiceprint model, unauthorized voice demo, report and video

- Celine SHOGA: image data, facial recognition model and command-line app
