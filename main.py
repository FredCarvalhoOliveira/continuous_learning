import pandas as pd
from river import linear_model, preprocessing, metrics, compose
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


ratings = pd.read_csv(
    "/Users/nimda/Documents/personal_repos/continuous_learning/data/ratings.csv"
)
ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})

movies = pd.read_csv(
    "/Users/nimda/Documents/personal_repos/continuous_learning/data/movies.csv"
)[["movieId", "genres"]]
movies = movies.rename(columns={"movieId": "item_id"})
ratings = ratings.merge(movies, on="item_id", how="left")
ratings["genres"] = ratings["genres"].fillna("")

print("Shape:", ratings.shape)
print(ratings.head())

ratings["implicit"] = (ratings["rating"] >= 4).astype(
    int
)  # convert to implicit feedback dataset

print(ratings.head())

model = compose.Pipeline(
    ("onehot", preprocessing.OneHotEncoder()),
    (
        "scale + logreg",
        preprocessing.StandardScaler() | linear_model.LogisticRegression(),
    ),
)

metric = metrics.ROCAUC()

UPDATE_EVERY = 1000
y_true_history = []
y_score_history = []

plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
(roc_line,) = ax.plot([], [], color="C0", label="ROC")
ax.plot([0, 1], [0, 1], "k--", label="Chance")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Live ROC")
ax.legend(loc="lower right")
auc_text = ax.text(0.62, 0.03, "AUC: N/A", transform=ax.transAxes)
fig.canvas.draw()
plt.show(block=False)

step_index = 0
for _, row in tqdm(ratings.iterrows(), total=len(ratings), desc="Training"):
    x = {"user_id": str(row.user_id), "item_id": str(row.item_id)}  # features
    genres_str = row.genres if isinstance(row.genres, str) else ""
    if genres_str and genres_str != "(no genres listed)":
        for g in genres_str.split("|"):
            if g:
                x[f"genre_{g}"] = 1
    y = row.implicit  # label

    proba = model.predict_proba_one(x).get(1, 0.5)  # probability for class 1
    model.learn_one(x, y)
    metric.update(y, proba)

    y_true_history.append(int(y))
    y_score_history.append(float(proba))

    if step_index % UPDATE_EVERY == 0 and step_index > 0:
        labels_present = set(y_true_history)
        if 0 in labels_present and 1 in labels_present:
            fpr, tpr, _ = roc_curve(y_true_history, y_score_history, pos_label=1)
            sk_auc = auc(fpr, tpr)
            roc_line.set_data(fpr, tpr)
            auc_text.set_text(f"AUC: {sk_auc:.4f} | online: {metric.get():.4f}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

    step_index += 1

print("Final ROC AUC:", metric.get())

if 0 in set(y_true_history) and 1 in set(y_true_history):
    fpr, tpr, _ = roc_curve(y_true_history, y_score_history, pos_label=1)
    sk_auc = auc(fpr, tpr)
    roc_line.set_data(fpr, tpr)
    auc_text.set_text(f"AUC: {sk_auc:.4f} | online: {metric.get():.4f}")
    fig.canvas.draw()
plt.ioff()
plt.show()
