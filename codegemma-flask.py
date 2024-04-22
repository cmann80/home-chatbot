from flask import Flask, render_template, request
import os
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run():
    code = request.form['code']
    output = subprocess.run(['python', '-c', code], capture_output=True, text=True)
    return render_template('index.html', output=output.stdout)

if __name__ == '__main__':
    app.run(debug=True)