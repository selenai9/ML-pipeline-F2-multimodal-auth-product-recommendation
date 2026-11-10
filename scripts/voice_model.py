"""Voiceprint Verification Model
Reads data/audio_features.csv, compares two models, prints their scores and
saves the best one to models/voice_model.pkl. Run from anywhere:
    python scripts/voice_model.py                  # train, evaluate, save
    python scripts/voice_model.py --verify FILE    # test one clip
    python scripts/voice_model.py --verify FILE --member selena

The command-line app imports verify_voice() from this file.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_pipeline import FEATURES_PATH, ROOT, features_for_file  # noqa: E402

MODEL_PATH = ROOT / "models" / "voice_model.pkl"
ID_COLUMNS = ["member", "sample_id", "augmentation"]

# A voice is approved only if the model is at least this sure about the speaker
CONFIDENCE_THRESHOLD = 0.6
# ... and the clip is not much further from that member's usual voice than
# their own recordings are (1.5 x the 95th-percentile distance of their own rows)
DISTANCE_MARGIN = 1.5


def candidate_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "Random Forest": make_pipeline(
            StandardScaler(), RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=2, random_state=42)),
    }


def load_features(path=FEATURES_PATH):
    data = pd.read_csv(path)
    # phrase = the sample id without the member name, e.g. "yes_approve"
    data["phrase"] = [s[len(m) + 1:] for s, m in zip(data["sample_id"], data["member"])]
    feature_cols = [c for c in data.columns if c not in ID_COLUMNS + ["phrase"]]
    return data, feature_cols


def evaluate(model, data, feature_cols):
    """Leave-one-phrase-out test: train on every version of one phrase and test
    on the other phrase. The model never sees the test clip (not even an
    augmented copy of it), so the score shows whether it recognises the
    speaker rather than the recording."""
    y_true, y_pred, proba_rows, classes = [], [], [], sorted(data["member"].unique())
    for phrase in sorted(data["phrase"].unique()):
        train = data[data["phrase"] != phrase]
        test = data[data["phrase"] == phrase]
        if train["member"].nunique() < len(classes):
            continue  # a member has no other phrase to train on
        model.fit(train[feature_cols], train["member"])
        proba = pd.DataFrame(model.predict_proba(test[feature_cols]),
                             columns=model.classes_).reindex(columns=classes, fill_value=0)
        y_true += list(test["member"])
        y_pred += list(model.predict(test[feature_cols]))
        proba_rows.append(proba.to_numpy())
    proba_all = np.vstack(proba_rows)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "loss": log_loss(y_true, proba_all, labels=classes),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def distance_limits(model, data, feature_cols):
    """Each member's centre point (in scaled feature space) and how far their
    own rows usually are from it. Used to reject voices that do not sound
    like anyone in the team, even if the classifier still picks someone."""
    scaler = model.named_steps["standardscaler"]
    scaled = scaler.transform(data[feature_cols])
    centroids, limits = {}, {}
    for member in sorted(data["member"].unique()):
        rows = scaled[(data["member"] == member).to_numpy()]
        centre = rows.mean(axis=0)
        dists = np.linalg.norm(rows - centre, axis=1)
        centroids[member] = centre
        limits[member] = float(np.percentile(dists, 95) * DISTANCE_MARGIN)
    return centroids, limits


def train_and_save():
    data, feature_cols = load_features()
    members = sorted(data["member"].unique())
    print(f"Loaded {len(data)} rows, {len(feature_cols)} features, members: {members}")
    if len(members) < 2:
        raise SystemExit("Need clips from at least 2 members to train a voice model. "
                         "Add the other members' clips to audio/ and rerun audio_pipeline.py.")

    print("\nLeave-one-phrase-out evaluation:")
    results = {}
    for name, model in candidate_models().items():
        r = evaluate(model, data, feature_cols)
        results[name] = r
        print(f"  {name:20s} accuracy {r['accuracy']:.3f}   F1 {r['f1']:.3f}   loss {r['loss']:.3f}")

    # best F1 wins; lower loss breaks a tie
    best_name = min(results, key=lambda n: (-results[n]["f1"], results[n]["loss"]))
    print(f"\nChosen model: {best_name}")

    final = candidate_models()[best_name]
    final.fit(data[feature_cols], data["member"])
    centroids, limits = distance_limits(final, data, feature_cols)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final,
        "model_name": best_name,
        "feature_columns": feature_cols,
        "members": members,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "centroids": centroids,
        "distance_limits": limits,
        "scores": {k: {m: v[m] for m in ("accuracy", "f1", "loss")} for k, v in results.items()},
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH.relative_to(ROOT)}")
    return results


def verify_voice(audio_path, expected_member=None, model_path=MODEL_PATH):
    """Check whether a clip is an approved team voice.
    If expected_member is given (the person the face model recognised), the
    voice must also belong to that same person.
    Returns a dict with approved (True/False), the predicted member, the
    confidence, and the reason for the decision."""
    bundle = joblib.load(model_path)
    model, cols = bundle["model"], bundle["feature_columns"]

    x = pd.DataFrame([features_for_file(audio_path)])[cols]
    proba = model.predict_proba(x)[0]
    best = int(np.argmax(proba))
    predicted, confidence = model.classes_[best], float(proba[best])

    scaled = model.named_steps["standardscaler"].transform(x)[0]
    distance = float(np.linalg.norm(scaled - bundle["centroids"][predicted]))
    limit = bundle["distance_limits"][predicted]

    if confidence < bundle["confidence_threshold"]:
        approved, reason = False, f"not sure who is speaking (confidence {confidence:.2f})"
    elif distance > limit:
        approved, reason = False, f"voice is too different from {predicted}'s recordings"
    elif expected_member is not None and predicted != expected_member.lower():
        approved, reason = False, f"voice belongs to {predicted}, but the face was {expected_member}"
    else:
        approved, reason = True, f"voice matches {predicted}"

    return {
        "approved": approved,
        "predicted_member": predicted,
        "confidence": confidence,
        "distance": distance,
        "distance_limit": limit,
        "reason": reason,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", metavar="FILE", help="audio clip to check instead of training")
    parser.add_argument("--member", help="member the face model recognised (optional)")
    args = parser.parse_args()

    if args.verify:
        r = verify_voice(args.verify, args.member)
        status = "APPROVED" if r["approved"] else "DENIED"
        print(f"{status}: {r['reason']}")
        print(f"  predicted {r['predicted_member']}, confidence {r['confidence']:.2f}, "
              f"distance {r['distance']:.2f} (limit {r['distance_limit']:.2f})")
    else:
        train_and_save()


if __name__ == "__main__":
    main()
