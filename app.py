from flask import Flask,render_template,request,redirect,flash,session,url_for
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from datetime import datetime

app = Flask(__name__)


@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')
    # if request.method == 'POST':
    #     if 'user' in session:
    #         pass
    #     else:
    #         pass
    # else:
    #     if 'user' in session:
    #         pass
    #     else:
    #         return render_template('home.html')


@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('admin_dashboard'))
    else:
        return render_template('login.html')
    
@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        return render_template('admin_dashboard.html')
    else:
        return render_template('register.html')
    
@app.route('/admin_dashboard',methods=['GET','POST'])
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/add_photos',methods=['GET','POST'])
def add_photos():
    return render_template('add_photos.html')

@app.route('/train',methods=['GET','POST'])
def train():
    return render_template('train.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)