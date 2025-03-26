from  flask import Flask,render_template, request
import joblib

app = Flask(__name__)

vectorise = joblib.load('tfidf.pkl')
modal = joblib.load('spam_98.pkl')

@app.route('/', methods=['GET','POST'])
def index():
    result = None
    if request.method == 'POST':
        email = request.form.get('email')

        email_vector = vectorise.transform([email])
        prediction = modal.predict(email_vector)

        result = "Email is Spam" if prediction[0] == 1 else "Email is Ham" 

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)