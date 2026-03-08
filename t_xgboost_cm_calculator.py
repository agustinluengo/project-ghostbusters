import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load data
# -----------------------------
df = pd.read_parquet("./9be6127a-e740-4cd0-8534-4c147d8e6a3b_short.parquet")
df.rename(columns={'p_rev.1':'avg_prev',
                   'pcogs.1':'avg_pcogs', 
                   'cp.1':'avg_cp', 
                   'display_ads_amt.1':'avg_display_ads_amt', 
                   'vfcc.1':'avg_vfcc',
                   'sales_discount.1':'avg_sales_discount',
                   'net_ppm.1':'avg_net_ppm',
                   'units_shipped.1':'avg_units_shipped'}, inplace=True)

df["NetPPM (%)"] = df["net_ppm"] / df["p_rev"]
df["asp"] = df["p_rev"] / df["units_shipped"]
df["acu"] = df["pcogs"] / df["units_shipped"]


df = df.dropna()

df["item_size"] = df["item_size"].replace({
    "Small":1,
    "Medium":2,
    "Large":3,
    "HeavyBulky":4
})

percentile_cols = ["p_rev","units_shipped","net_ppm"]

for col in percentile_cols:

    n_unique = df[col].nunique()
    q = min(5, n_unique)

    labels = [1,2,3,5,6][:q] #["very_low","low","medium","high","very_high"]

    df[f"{col}_pct_cat"] = pd.qcut(
        df[col],
        q=q,
        labels=labels,
        duplicates="drop"
    )


ordinal_cols = [
    "asin",
    "gl_product_group",
    "product_category"
]

onehot_cols = [
    "warehouse_id",
]


# -----------------------------
# Column Transformer
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ordinal_cols),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot_cols)
    ],
    remainder="passthrough"   # keep all other columns automatically
)

# -----------------------------
# Fit + Transform
# -----------------------------

X_encoded = preprocessor.fit_transform(df)

# -----------------------------
# Column Names
# -----------------------------

ordinal_names = ordinal_cols

onehot_names = preprocessor.named_transformers_["onehot"].get_feature_names_out(onehot_cols)

remainder_cols = df.columns.difference(ordinal_cols + onehot_cols)

feature_names = list(ordinal_names) + list(onehot_names) + list(remainder_cols)

df_encoded = pd.DataFrame(X_encoded, columns=feature_names, index=df.index)

print(df_encoded.shape)


# -----------------------------
# Train Test Split
# -----------------------------

df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan)
df_encoded = df_encoded.dropna()


X = df_encoded.drop(columns=["cp"])
y = df_encoded["cp"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape, X_test.shape)


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# initialize model
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# train model
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)


importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_*100
}).sort_values("importance", ascending=False)

print(importance.head(20))


import matplotlib.pyplot as plt

importance.head(15).plot(
    x="feature",
    y="importance",
    kind="barh",
    figsize=(8,6)
)

plt.title("XGBoost Feature Importance")
plt.show()