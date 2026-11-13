from image_pipeline import build_feature_table, FEATURES_PATH

table = build_feature_table()
table.to_csv(FEATURES_PATH, index=False)

print("Saved to", FEATURES_PATH)
print("Rows and columns:", table.shape)
print(table.groupby(["member", "augmentation"]).size().unstack())