
from flask import Flask, render_template, request
import joblib

# Create Flask application
app = Flask(__name__)

# Load trained ML model and TF-IDF vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    message = ""

    if request.method == "POST":

        message = request.form.get("message", "").strip()

        if message:

            # Convert message into TF-IDF features
            message_vector = vectorizer.transform([message])

            # Predict using trained model
            result = model.predict(message_vector)

            if result[0] == 1:
                prediction = "SPAM"
            else:
                prediction = "HAM"

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)
