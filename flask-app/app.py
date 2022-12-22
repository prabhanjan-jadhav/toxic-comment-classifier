from flask import Flask, redirect, url_for, render_template, request
from simpler_utils import get_predictions
# WSGI application
app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/home/')
def home():
    return "Welcome to the Toxic Comment Classification Website"

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    comment = ''
    if request.method=='POST':
        comment = request.form['comment']
        try :
            preds = get_predictions(comment=comment)
            return preds
        except:
            return "Error during prediction"

if __name__=='__main__':
    app.run(debug=True)

