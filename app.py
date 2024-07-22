from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

def most_frequent_string(string_array):
  """
  Finds the most frequent string in a given array of strings.

  Args:
      string_array: An array of strings.

  Returns:
      The most frequent string in the array. If multiple strings have the same 
      highest frequency, returns one of them.
  """
  # Initialize a dictionary to store string counts
  string_counts = {}
  # Loop through the array and count occurrences
  for string in string_array:
    string_counts[string] = string_counts.get(string, 0) + 1  # Use get(string, 0) for default value of 0
  # Find the string with the maximum count
  most_frequent = max(string_counts, key=string_counts.get)
  return most_frequent

@app.route('/')
def home():
    return render_template('index.html')

model_gbm = joblib.load('prod_model_gbm.pkl')
model_lr = joblib.load('prod_model_lr.pkl')
model_nb = joblib.load('prod_model_nb.pkl')
model_svm = joblib.load('prod_model_svm.pkl')
@app.route('/classify', methods=['POST'])
def classify():
    text = request.json['text']
    category_gbm = model_gbm.predict([text])[0]
    category_lr = model_lr.predict([text])[0]
    category_nb = model_nb.predict([text])[0]
    category_svm = model_svm.predict([text])[0]

    return jsonify({'category': most_frequent_string([category_gbm, category_lr, category_nb, category_svm])})

if __name__ == '__main__':
    app.run(debug=True)
