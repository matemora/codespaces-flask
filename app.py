from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('text_classifier.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.json['text']
    category = model.predict([text])[0]
    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(debug=True)
