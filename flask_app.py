
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, render_template, send_file, make_response, url_for, Response, redirect, request
import sys
import os

app = Flask(__name__)

#decorator for homepage
@app.route('/' )
def index():
    return render_template('index.html', PageTitle = "Landing page")

#These functions will run when POST method is used.
@app.route('/', methods = ["POST"] )
def generate_lineup():
    os.system('python3 ~/baseball_shift/shift_scheduling_sat.py > ~/mysite/templates/output.txt')
    os.system('~/mysite/templates/convert.sh ~/mysite/templates/output.txt')
    return render_template('output.html', PageTitle = "Execution result")

if __name__ == '__main__':
    app.run(main)
