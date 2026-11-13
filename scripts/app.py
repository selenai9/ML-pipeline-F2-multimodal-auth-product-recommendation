"""Command-line app: face -> product prediction -> voice -> result.

Run from the repo's main folder:
    python scripts/app.py
    python scripts/app.py --face images/selena_neutral.jpeg --voice audio/selena_yes_approve.mp4
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from face_model import verify_face
from voice_model import verify_voice

ROOT = Path(__file__).resolve().parent.parent

# each team member stands for one customer in the merged dataset
MEMBER_CUSTOMER = {"selena": 107, "celine": 113, "joshua": 132}


def ask(prompt, given):
    """Use the path given on the command line, or ask the user to type one."""
    return given or input(prompt).strip().strip('"')


def predict_product(customer_id):
    """Predict the product from the customer's most recent purchase."""
    data = pd.read_csv(ROOT / "data" / "merged_dataset.csv")
    rows = data[data["customer_id"] == customer_id].sort_values("purchase_date")
    latest = rows.iloc[[-1]]
    model = joblib.load(ROOT / "models" / "product_model.pkl")
    columns = joblib.load(ROOT / "models" / "product_model_columns.pkl")
    proba = model.predict_proba(latest[columns])[0]
    best = proba.argmax()
    return model.classes_[best], float(proba[best])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--face", help="path to the face image")
    parser.add_argument("--voice", help="path to the voice clip")
    args = parser.parse_args()

    print("=== Product Recommendation System ===")

    # Step 1: face check
    face = verify_face(ask("Face image path: ", args.face))
    if not face["approved"]:
        print(f"ACCESS DENIED (face): {face['reason']}")
        return
    member = str(face["predicted_member"])
    print(f"Face recognised: {member} (confidence {face['confidence']:.2f})")

    # Step 2: the product model runs, but the result stays hidden until the voice check
    customer_id = MEMBER_CUSTOMER.get(member)
    if customer_id is None:
        print(f"ACCESS DENIED: no customer account is linked to {member}")
        return
    product, confidence = predict_product(customer_id)
    print(f"A prediction is ready for customer {customer_id}. Confirm with your voice to see it.")

    # Step 3: voice check
    voice = verify_voice(ask("Voice clip path (e.g. 'Yes, approve'): ", args.voice),
                         expected_member=member)
    if not voice["approved"]:
        print(f"ACCESS DENIED (voice): {voice['reason']}")
        return
    print(f"Voice verified: {voice['reason']}")
    print(f"ACCESS GRANTED. Predicted product for customer {customer_id}: "
          f"{product} ({confidence:.0%} sure)")


if __name__ == "__main__":
    main()