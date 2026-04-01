#  AI SMS Spam Detector (98.39% Accuracy)

A professional Machine Learning microservice that classifies SMS messages as **Ham** (Legit) or **Spam** using a Naive Bayes classifier.

##  Methodology
- **Dataset:** UCI SMS Spam Collection (via Kaggle).
- **Preprocessing:** Text Vectorization using `CountVectorizer`.
- **Model:** `MultinomialNB` (Multinomial Naive Bayes), ideal for text classification.
- **Performance:** Achieved **98.39% accuracy** on the test set.

##  Tech Stack
- **AI/ML:** Scikit-Learn, Pandas, Joblib.
- **API:** FastAPI, Uvicorn.
- **Deployment:** Render / GitHub Actions.

##  How to use
1. Clone the repo.
2. Run `pip install -r requirements.txt`.
3. Run `uvicorn main:app --reload`.
4. Send a POST request to `/predict` with a JSON body:
   ```json
   { "text": "WINNER! Claim your prize now!" }
