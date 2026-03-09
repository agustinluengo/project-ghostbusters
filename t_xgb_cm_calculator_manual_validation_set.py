import numpy as np
import pandas as pd
import joblib

# LOAD DATA
validation_df = pd.read_csv("/cp_asin_validation_set.csv")

# RENAME
validation_df = validation_df.rename(columns={
    'p_rev.1':'avg_prev',
    'pcogs.1':'avg_pcogs',
    'cp.1':'avg_cp',
    'display_ads_amt.1':'avg_display_ads_amt',
    'vfcc.1':'avg_vfcc',
    'sales_discount.1':'avg_sales_discount',
    'net_ppm.1':'avg_net_ppm',
    'units_shipped.1':'avg_units_shipped'
})

# FEATURE ENGINEERING
validation_df["cppu"] = validation_df["cp"] / validation_df["p_rev"]
validation_df["NetPPM (%)"] = validation_df["net_ppm"] / validation_df["p_rev"]
validation_df["asp"] = validation_df["p_rev"] / validation_df["units_shipped"]
validation_df["acu"] = validation_df["pcogs"] / validation_df["units_shipped"]
percentile_cols = ["p_rev","units_shipped","net_ppm"]

for col in percentile_cols:

    n_unique = validation_df[col].nunique()
    q = min(5, n_unique)

    labels = list(range(1, q+1))

    validation_df[f"{col}_pct_cat"] = pd.qcut(
        validation_df[col],
        q=q,
        labels=labels,
        duplicates="drop"
    )

# DROP UNUSED
# #validation_df = validation_df.drop(columns=['warehouse_id','avg_sales_discount','sales_discount'])

# PERCENTILE FEATURES
percentile_cols = ["p_rev","units_shipped","net_ppm"]

for col in percentile_cols:

    n_unique = validation_df[col].nunique()
    q = min(5, n_unique)

    labels = list(range(1, q+1))

    validation_df[f"{col}_pct_cat"] = pd.qcut(
        validation_df[col],
        q=q,
        labels=labels,
        duplicates="drop"
    )

# LOAD TRAINED OBJECTS
preprocessor = joblib.load("/preprocessor.pkl")
model = joblib.load("/xgb_best_model.pkl")

# TRANSFORM FEATURES
X_new = preprocessor.transform(validation_df)

# -----------------------------
# Column Names
# -----------------------------

ordinal_cols = [
    "asin",
    "gl_product_group",
    "product_category"
]

ordinal_names = ordinal_cols

remainder_cols = validation_df.columns.difference(ordinal_cols)

feature_names = list(ordinal_names) + list(remainder_cols)

df_encoded = pd.DataFrame(X_new, columns=feature_names, index=validation_df.index)

df_encoded = df_encoded.drop(columns=['cp','avg_cp','cppu'])

# PREDICT
predictions = model.predict(df_encoded)

validation_df["prediction"] = predictions

# SAVE OUTPUT
validation_df.to_csv("/predictions.csv", index=False)