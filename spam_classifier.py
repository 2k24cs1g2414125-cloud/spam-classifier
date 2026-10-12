import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load dataset (SMS Spam Collection from Kaggle)
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Convert text to TF-IDF features
vec = TfidfVectorizer(stop_words="english")
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test_vec)))

# Predict on a new message
def predict(msg):
    return "spam" if model.predict(vec.transform([msg]))[0] == 1 else "not spam"

print(predict("You won a free prize, click now"))
