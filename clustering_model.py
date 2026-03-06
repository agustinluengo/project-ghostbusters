import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# -----------------------------
# Load data
# -----------------------------
df = pd.read_parquet("./9be6127a-e740-4cd0-8534-4c147d8e6a3b_short.parquet")

# -----------------------------
# Cleaning
# -----------------------------
df = df.dropna()

df["item_size"] = df["item_size"].replace({
    "Small":1,
    "Medium":2,
    "Large":3,
    "HeavyBulky":4
})

# -----------------------------
# Feature Engineering
# -----------------------------
df["CM"] = df["cp"] / df["p_rev"]
df["NetPPM (%)"] = df["net_ppm"] / df["p_rev"]
df["asp"] = df["p_rev"] / df["units_shipped"]
df["acu"] = df["pcogs"] / df["units_shipped"]
df["cppu"] = df["cp"] / df["units_shipped"]

# -----------------------------
# Percentile Categories
# -----------------------------
percentile_cols = ["p_rev","units_shipped","net_ppm","cp"]

for col in percentile_cols:

    n_unique = df[col].nunique()
    q = min(5, n_unique)

    labels = ["very_low","low","medium","high","very_high"][:q]

    df[f"{col}_pct_cat"] = pd.qcut(
        df[col],
        q=q,
        labels=labels,
        duplicates="drop"
    )

# -----------------------------
# Encoding columns
# -----------------------------

ordinal_cols = [
    "asin",
    "gl_product_group",
    "product_category"
]

onehot_cols = [
    "warehouse_id",
    "p_rev_pct_cat",
    "units_shipped_pct_cat",
    "net_ppm_pct_cat",
    "cp_pct_cat"
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

from sklearn.model_selection import train_test_split
import numpy as np
df_encoded = df_encoded.replace([np.inf, -np.inf], np.nan)
df_encoded = df_encoded.dropna()


X = df_encoded.drop(columns=["cp","CM"])
y = df_encoded["cp"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape, X_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Unsupervised Clustering (KMeans)
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init="auto"
)

clusters = kmeans.fit_predict(X_train)

X_train["cluster"] = clusters

# Evaluate Clustering

from sklearn.metrics import silhouette_score

score = silhouette_score(X_train.drop(columns="cluster"), clusters)

print("Silhouette Score:", score)
#Silhouette Score: 0.8820038162565282

X_train.var().sort_values(ascending=False).head(10)

# acu                  1.721624e+09
# asp                  1.660056e+09
# cppu                 5.286432e+07
# display_ads_amt.1    4.608529e+07
# asin                 1.260083e+07
# item_size            4.139076e+06
# cp.1                 1.013124e+06
# display_ads_amt      9.325484e+05
# product_category     2.636220e+04
# net_ppm              2.012461e+03



cluster_features = [
    "p_rev",
    "units_shipped",
    "pcogs",
    "display_ads_amt",
    "sales_discount",
    "item_size"
]

X_cluster = X_train[cluster_features]

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_scaled)

score = silhouette_score(X_scaled, clusters)

print("Silhouette Score:", score)
#Silhouette Score: 0.39455426010373335

# Visualize Clusters (TSNE)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, perplexity=30, random_state=42)

X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8,6))

plt.scatter(
    X_tsne[:,0],
    X_tsne[:,1],
    c=clusters,
    cmap="viridis",
    alpha=0.7
)

plt.title("Cluster Visualization (t-SNE)")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")

plt.show()