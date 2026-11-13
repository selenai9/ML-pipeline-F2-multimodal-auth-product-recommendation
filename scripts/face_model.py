"""
Facial recognition model (Formative 2, Part 2).

Trains a classifier on data/image_features.csv, evaluates it, saves it to
models/face_model.pkl, and provides verify_face() for the command-line app.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from image_pipeline import (ROOT, FEATURES_PATH, load_image, extract_features,
                            feature_names, parse_name)

MODEL_PATH = ROOT / "models" / "face_model.pkl"
CONFIDENCE_THRESHOLD = 0.70   # the model must be at least 60% sure
DISTANCE_MARGIN = 1.15        # how far past the largest "normal" distance is still accepted
META_COLS = ["member", "sample_id", "augmentation"]


def load_features():
    data = pd.read_csv(FEATURES_PATH).copy()
    cols = [c for c in data.columns if c not in META_COLS]
    data = data.assign(expression=data["sample_id"].map(lambda s: parse_name(s)[1]))
    return data, cols


def candidate_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42)),
    }


def _distance(model, rows, centroid):
    """Typical per-feature gap between a face and a member's average face,
    measured after scaling. Small = looks like that member."""
    scaled = model.named_steps["standardscaler"].transform(rows)
    return np.sqrt(((scaled - centroid) ** 2).mean(axis=1))


def _centroid(model, rows):
    return model.named_steps["standardscaler"].transform(rows).mean(axis=0)


def evaluate(data, cols):
    """Leave-one-expression-out: train on two expressions, test on the third.

    The test photo was never seen in training (augmented copies of it are
    also held out), so a good score means the model recognises the person,
    not the photo. Also collects the distance of each held-out ORIGINAL photo
    to its member's average face. Real photos arrive upright and unedited, so
    only originals (not rotated or darkened copies) are used to set the
    stranger-rejection threshold.
    """
    expressions = sorted(data["expression"].unique())
    results, held_out_distance = {}, {}
    for name, model in candidate_models().items():
        y_true, y_pred, proba_all = [], [], []
        for held in expressions:
            train, test = data[data["expression"] != held], data[data["expression"] == held]
            model.fit(train[cols].values, train["member"])
            proba = model.predict_proba(test[cols].values)
            y_true += list(test["member"])
            y_pred += list(model.classes_[proba.argmax(axis=1)])
            proba_all.append((test["member"].values, proba, list(model.classes_)))
            if name == "Logistic Regression":
                for m in model.classes_:
                    c = _centroid(model, train.loc[train["member"] == m, cols].values)
                    own = test[(test["member"] == m) & (test["augmentation"] == "original")]
                    d = _distance(model, own[cols].values, c)
                    held_out_distance.setdefault(m, []).extend(d)
        classes = sorted(data["member"].unique())
        probs = np.zeros((len(y_true), len(classes)))
        i = 0
        for _, p, cl in proba_all:
            for row in p:
                for k, c in enumerate(cl):
                    probs[i, classes.index(c)] = row[k]
                i += 1
        results[name] = {"accuracy": accuracy_score(y_true, y_pred),
                         "f1": f1_score(y_true, y_pred, average="macro"),
                         "loss": log_loss(y_true, np.clip(probs, 1e-6, 1), labels=classes),
                         "y_true": y_true, "y_pred": y_pred}
    return results, held_out_distance


def train_and_save():
    data, cols = load_features()
    results, held_out = evaluate(data, cols)
    best = max(results, key=lambda n: results[n]["f1"])
    model = candidate_models()[best]
    model.fit(data[cols].values, data["member"])
    centroids = {m: _centroid(model, data.loc[data["member"] == m, cols].values)
                 for m in model.classes_}
    limits = {m: DISTANCE_MARGIN * max(held_out[m]) for m in model.classes_}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "name": best, "columns": cols,
                 "centroids": centroids, "distance_limit": limits}, MODEL_PATH)
    return results


def verify_face(image_path):
    """Step 1 of the system: does this photo show a team member?

    Approved only if (1) the model is at least 60% sure who it is, and
    (2) the face is not much further from that member's usual faces than the
    member's own held-out photos were. Check 2 catches strangers, because a
    classifier always picks *someone* from the team.
    """
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    row = extract_features(load_image(image_path)).reshape(1, -1)
    proba = model.predict_proba(row)[0]
    member = model.classes_[proba.argmax()]
    conf = float(proba.max())
    dist = float(_distance(model, row, bundle["centroids"][member])[0])
    limit = bundle["distance_limit"][member]
    if conf < CONFIDENCE_THRESHOLD:
        reason = f"low confidence ({conf:.2f} < {CONFIDENCE_THRESHOLD})"
    elif dist > limit:
        reason = f"face too different from {member}'s photos (distance {dist:.2f} > {limit:.2f})"
    else:
        reason = "face recognised"
    return {"approved": reason == "face recognised",
            "predicted_member": member if reason == "face recognised" else None,
            "best_guess": member, "confidence": conf, "distance": dist, "reason": reason}
