from flask import Flask,render_template,request,redirect,flash,session,url_for
#from flask_sqlalchemy import SQLAlchemy
#from flask_mail import Mail
from datetime import datetime
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import cv2
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import datetime
import csv
import pandas as pd


with open('password.txt') as f:
    db_password = f.read()

app = Flask(__name__)

app.secret_key = 'your secret key'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = db_password
app.config['MYSQL_DB'] = 'attendance_system'

mysql = MySQL(app)

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

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
    msg=''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        username = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        value=request.form.get('teacher')
        if value=="1":
            cursor.execute('SELECT * FROM teachers WHERE temail = % s AND password = % s', (username, password, ))
            account = cursor.fetchone()
            if account:
                session['loggedin'] = True
                session['tid'] = account['tid']
                session['tname'] = account['tname']
                msg = 'Logged in successfully !'
                return redirect(url_for('admin_dashboard',msg=msg))
            else:
                msg = 'Incorrect username / password !'
        else:
            cursor.execute('SELECT * FROM students WHERE semail = % s AND student_id = % s', (username, password, ))
            account = cursor.fetchone()
            if account:
                session['loggedin'] = True
                session['student_id'] = account['student_id']
                session['fname'] = account['fname']
                msg = 'Logged in successfully !'
                return redirect(url_for('train',msg=msg))
            else:
                msg = 'Incorrect username / password !'
    return render_template('login.html')
    
@app.route('/register',methods=['GET','POST'])
def register():
    msg=''
    if request.method == 'POST' and 'tname' in request.form and 'password' in request.form and 'temail' in request.form :
        tname = request.form['tname']
        password = request.form['password']
        email = request.form['temail']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM teachers WHERE tname = % s', (tname, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', tname):
            msg = 'Username must contain only characters and numbers !'
        else:
            cursor.execute('INSERT INTO teachers VALUES (NULL, % s, % s, % s)', (tname, email, password, ))
            mysql.connection.commit()
            msg = 'You have successfully registered !'
            flash(msg)
            return redirect(url_for('login'))
        return redirect(url_for('register'))
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    else:
        return render_template('register.html')
    
@app.route('/admin_dashboard',methods=['GET','POST'])
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/student_home',methods=['GET','POST'])
def student_home():
    return render_template('student_home.html')

@app.route('/register_student',methods=['GET','POST'])
def register_student():
    msg=''
    if request.method == 'POST' and 'sid' in request.form and 'name' in request.form and 'dob' in request.form and 's_email' in request.form :
        sid=request.form['sid']
        name = request.form['name']
        dob = request.form['dob']
        s_email = request.form['s_email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM students WHERE name = % s', (name, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', s_email):
            msg = 'Invalid email address !'
        elif not re.match(r'[A-Za-z0-9]+', name):
            msg = 'Username must contain only characters and numbers !'
        elif not name or not dob or not s_email or not sid:
            msg = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO students VALUES (% s, % s, % s, % s)', (sid ,name, dob, s_email, ))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
            return render_template('admin_dashboard.html')
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    else:
        print(msg)
        return render_template('register_student.html',msg=msg)
    

@app.route('/add_photos',methods=['GET','POST'])
def add_photos():
    if request.method == 'POST' and 'name' in request.form and 'sid' in request.form:
        name = request.form['name']
        face_id=request.form['sid']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM students WHERE name = % s', (name, ))
        account = cursor.fetchone()
        if account:
            session['name'] = account['name']
            vid_cam = cv2.VideoCapture(0)
            face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
            count = 0
            assure_path_exists("data/1")
            while (True):
                # Capture video frame
                _, image_frame = vid_cam.read()

    # Convert frame to grayscale
                gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
                faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loops for each faces
                for (x, y, w, h) in faces:
        # Crop the image frame into rectangle
                    cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Increment sample face image
                    count += 1

        # Save the captured image into the datasets folder
                    cv2.imwrite("data/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        # Display the video frame, with bounded rectangle on the person's face
                    cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

    # If image taken reach 100, stop taking video
                elif count >= 50:
                    print("Successfully Captured")
                    break

# Stop video
            vid_cam.release()

# Close all started windows
            cv2.destroyAllWindows()
            return render_template('admin_dashboard.html')
        else:
            msg = 'Incorrect name'
    return render_template('add_photos.html')

@app.route('/create_dataset',methods=['GET','POST'])
def create_dataset():
    return render_template('train.html')

@app.route('/train',methods=['GET','POST'])
def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector= cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml");
    def getImagesAndLabels(path):
    #get the path of all the files in the folder
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #create empth face list
        faceSamples=[]
    #create empty ID list
        Ids=[]
    #now looping through all the image paths and loading the Ids and the images
        for imagePath in imagePaths:
        #loading the image and converting it to gray scale
            pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
            imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
            Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
            faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
            for (x,y,w,h) in faces:
                faceSamples.append(imageNp[y:y+h,x:x+w])
                Ids.append(Id)
        return faceSamples,Ids

    faces,Ids = getImagesAndLabels('data')
    s = recognizer.train(faces, np.array(Ids))
    print("Successfully trained")
    recognizer.write('model1.yml')

    return render_template('train.html')

@app.route('/mark_attendance_details',methods=['GET','POST'])
def mark_attendance_details():
    return render_template('mark_attendance_details.html')

@app.route('/mark_your_attendance',methods=['GET','POST'])
def mark_your_attendance():
    class_names = {1:"Shreya",2:"Shreyatwo", 3:"Muskan",4:"Kumkum"}#,2:"Muskan",3:"Balaji",4:"Tejashree"} #name of people

    # load the model from disk
    filename = "model1.yml"
    
    f = pd.read_csv('attendance.csv')
    # Initialize LPBH model object and load the model
    lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
    lbph_face_classifier.read(filename)
    cap=cv2.VideoCapture(0)
    cap.set(3,640) # set Width
    cap.set(4,480) # set Height'''
    font = cv2.FONT_HERSHEY_SIMPLEX
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
    
    while True:
        ret, img = cap.read()
        
        if ret == False:
            print('Camera is not on')
            break
        else:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)
        #Scaling factor 1.3
        # Minimum naber 5
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cropped_image = img[y:y+h, x:x+w] 
                IMG_SIZE=450
                color_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            #cnnModel(color_img)

                predictions = lbph_face_classifier.predict(color_img)
                print(predictions)
                value=int(predictions[0])
                name= class_names[predictions[0]]
                cv2.putText(img, str(class_names[predictions[0]]), (x+5,y-5), font, 1, (255,0,255), 2)
                cv2.putText(img, str(round(predictions[1],2))+"%", (x+5,y+h-5), font, 1, (255,255,0), 1)
                #name=predictions[0]
            #print()
                if class_names[predictions[0]] in class_names.values():
                    val=str(class_names[predictions[0]])
                    print(val)
                    f.loc[value-1, 'Name'] = name
                    f.loc[value-1, 'Attendance'] = "Present"
                    print("yes")
                    cap.release()
                    cv2.destroyAllWindows()
                    f.to_csv('attendance.csv', index=False)
                    return render_template('home.html')
            cv2.imshow('video',img)        
                    #cv2.waitKey(100) 
                    
            #cv2.imshow('video',img)
            #return render_template('home.html')
                    #current_time = now.strftime("%H-%M-%S")
                #lnwriter.writerow([val,current_time])
            
    return render_template('home.html')

@app.route('/logout',methods=['GET','POST'])
def logout():
    session.pop('loggedin', None)
    session.pop('tid', None)
    session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, port=5001)