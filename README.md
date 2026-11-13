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
images/     face photos, named <member>_<expression> (e.g. selena_neutral.jpeg)
data/       raw spreadsheets, merged_dataset.csv, audio_features.csv and image_features.csv
models/     product model, face_model.pkl and voice_model.pkl
scripts/    product_model.py, audio_pipeline.py, voice_model.py, image_pipeline.py,
            face_model.py, app.py, convert_heic.py, make_image_features.py, make_face_model.py
formative2_multimodal.ipynb   data merge, EDA, feature engineering, product model, audio and voice model
```

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

## Image and face recognition model

- Each member provides 3 photos: neutral, smiling and surprised (`images/<member>_<expression>`). Celine's iPhone photos were converted from HEIC to JPG with `scripts/convert_heic.py`.
- Photos are cropped to the face (OpenCV face detector, with a centre crop if no face is found). Augmentations (6 per photo): rotated left and right by 15 degrees, flipped, grayscale, brighter and darker.
- Features (948 per row): HOG edge features plus colour histograms, saved to `data/image_features.csv`.
- The model is tested leave-one-expression-out: it is trained on two expressions and tested on the third. Random Forest scored accuracy 0.976 and F1 0.976 (loss 0.302); Logistic Regression scored 0.952 and 0.952 (loss 0.148). Random Forest was saved to `models/face_model.pkl`.
- A face is approved only if the model is at least 70% sure (`CONFIDENCE_THRESHOLD = 0.70` in `scripts/face_model.py`) and the face is close to that member's own photos.
- **Why 70%:** at 60%, a stranger's photo (`images/unauthorized_person.jpg`) was only just accepted (confidence 0.60). We raised the cutoff to 70%, and the stranger is now denied, while Selena's own photo still scores 0.99. A stricter cutoff means a real member may sometimes be rejected. We accepted this because it is safer to wrongly reject a member than to let a stranger in. We tested with only one stranger photo, so 70% is a demonstration setting and not a proven security level.
- The model is currently trained on Selena's and Celine's photos only. With so few people and photos, the scores should be read carefully, and the voice check acts as a second layer.

## Command-line app

`scripts/app.py` runs the full flow: face check, product prediction, voice check. Each team member is linked to one customer from the merged dataset (Selena is customer 107, Celine is 113, Joshua is 132). The product model runs after the face check, but the predicted product is shown only if the voice also matches the same person.

Successful transaction:

```
python scripts/app.py --face images/selena_neutral.jpeg --voice audio/selena_yes_approve.mp4
```

Unauthorized attempt (a stranger's photo, denied at the face step):

```
python scripts/app.py --face images/unauthorized_person.jpg --voice audio/selena_yes_approve.mp4
```

Unauthorized attempt (a stranger's voice):

```
python scripts/app.py --face images/selena_neutral.jpeg --voice audio/unauthorized_yes_approve.m4a
```

If you leave out `--face` or `--voice`, the app asks you to type the file path.

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

Install the extra packages for the face part, then extract the image features and train the face model:

```
pip install scikit-image pillow pillow-heif opencv-python==4.10.0.84
python scripts/convert_heic.py
python scripts/make_image_features.py
python scripts/make_face_model.py
```

`convert_heic.py` is only needed for HEIC photos.

## Team

- Selena ISIMBI: data merge, EDA, feature engineering, product recommendation model, command-line app
  
- Joshua Agonzibwa RUSOKE: audio data, voiceprint model, unauthorized voice demo, report and video

- Celine SHOGA: image data, facial recognition model