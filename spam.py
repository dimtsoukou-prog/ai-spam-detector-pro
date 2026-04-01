import pandas as pd
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

print("download from kaggle...")
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")

csv_path = os.path.join(path, "spam.csv")

df = pd.read_csv(csv_path, encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(f"📊 Συνολικές γραμμές δεδομένων: {len(df)}")

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 4. 
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

accuracy = model.score(vectorizer.transform(X_test), y_test)
print(f"Training done! The accuracy is: {accuracy * 100:.2f}%")