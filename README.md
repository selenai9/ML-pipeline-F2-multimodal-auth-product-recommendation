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
formative2_multimodal.ipynb   data merge, EDA, feature engineering, product model, image and face model, audio and voice model
requirements.txt              Python packages needed to run everything
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

- Results with all three members (36 rows): Random Forest scored accuracy 0.806, F1 0.810 and loss 0.620. Logistic Regression scored 0.722, 0.720 and 0.485. Random Forest was saved to `models/voice_model.pkl`.

- Every original recording was recognised correctly. All 7 mistakes were augmented copies of Celine's and Selena's clips that the model labelled as Joshua, mostly the background-noise copies. Joshua's recordings have a steady background hum, so the model has partly learned that "noisy" means Joshua. Recording his phrases again in a quiet room should improve the score.

- Verification tests: all 13 decisions were correct. Each member's own clips are approved (confidence 0.92 to 0.97), the same clips are denied when paired with another member's face, and the stranger's clip (`audio/unauthorized_yes_approve.m4a`) is denied at confidence 0.38.

## Image and face recognition model

- Each member provides 3 photos: neutral, smiling and surprised (`images/<member>_<expression>`). Celine's iPhone photos were converted from HEIC to JPG with `scripts/convert_heic.py`.

- Photos are cropped to the face (OpenCV face detector, with a centre crop if no face is found). Augmentations (6 per photo): rotated left and right by 15 degrees, flipped, grayscale, brighter and darker.

- Features (948 per row): HOG edge features plus colour histograms, saved to `data/image_features.csv`.

- The model is tested leave-one-expression-out: it is trained on two expressions and tested on the third. With all three members (63 rows), Random Forest scored accuracy 0.968, F1 0.968 and loss 0.419. Logistic Regression scored 0.968, 0.969 and 0.122. Only 2 rows were wrong, both augmented copies (a rotated Selena photo and a flipped Celine photo).

- **Why Random Forest:** the two models score almost the same, but Logistic Regression is overconfident on faces it has never seen. It accepted the stranger's photo as Celine with 0.85 confidence. Random Forest gives the stranger only 0.47, so the confidence check rejects them. `scripts/face_model.py` therefore always saves the Random Forest to `models/face_model.pkl`.

- A face is approved only if the model is at least 70% sure (`CONFIDENCE_THRESHOLD = 0.70` in `scripts/face_model.py`) and the face is close to that member's own photos.

- **Why 70%:** with two members, a stranger's photo (`images/unauthorized_person.jpg`) was only just accepted at a 60% cutoff (confidence 0.60), so we raised it to 70%. With three members, every team photo scores 0.97 to 1.00 and the stranger scores 0.47, so all 10 test decisions are correct. A stricter cutoff means a real member may sometimes be rejected. We accepted this because it is safer to wrongly reject a member than to let a stranger in. We tested with only one stranger photo, so 70% is a demonstration setting and not a proven security level.

- **Use OpenCV 4.x.** OpenCV 5 no longer includes the face detector file, so the code quietly falls back to a centre crop. The features then come out different and the face model no longer matches. `requirements.txt` pins the version.

## Command-line app

`scripts/app.py` runs the full flow: face check, product prediction, voice check. Each team member is linked to one customer from the merged dataset (Selena is customer 107, Celine is 113, Joshua is 132). The product model runs after the face check, but the predicted product is shown only if the voice also matches the same person.

Successful transaction:

```
python scripts/app.py --face images/selena_neutral.jpeg --voice audio/selena_yes_approve.mp4
python scripts/app.py --face images/joshua_neutral.jpg --voice audio/joshua_yes_approve.mp4
```

Wrong voice for the face (Joshua's photo with Selena's voice, denied at the voice step):

```
python scripts/app.py --face images/joshua_neutral.jpg --voice audio/selena_yes_approve.mp4
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
pip install -r requirements.txt
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

Extract the image features, then train and save the face model:

```
python scripts/convert_heic.py
python scripts/make_image_features.py
python scripts/make_face_model.py
```

`convert_heic.py` is only needed for HEIC photos.

## Team Members

- Selena Isimbi: data merge, EDA, feature engineering, product recommendation model, command-line app
  
- Joshua Rusoke Agonzibwa: audio data, voiceprint model, unauthorized voice demo, report and video

- Celine Shoga: image data, facial recognition model
