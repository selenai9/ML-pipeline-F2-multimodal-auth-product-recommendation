from face_model import train_and_save, MODEL_PATH

results = train_and_save()

for name, r in results.items():
    print(f"{name:20s} accuracy {r['accuracy']:.3f}   F1 {r['f1']:.3f}   loss {r['loss']:.3f}")

print("Face model saved to", MODEL_PATH)