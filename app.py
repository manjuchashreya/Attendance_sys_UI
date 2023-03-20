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
from werkzeug.utils import secure_filename
import json
import ast


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
SEND_VARIABLE = {}

class_names = {1903064:"Shreya Manjucha" , 1903032: "Muskan Gupta", 1903091:"Rishabh Raghuvanshi", 1903123:"Unnati Sarothi",1903129:"Tejashree Tambe", 1903139:"Balaji Wadawadagi"}#,2:"Muskan",3:"Balaji",4:"Tejashree"} #name of people

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
current_time = now.strftime("%H:%M:%S")

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

def allowed_file(filename):     
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            if password == 'Tsec2023':
                cursor.execute('SELECT * FROM teachers WHERE temail = % s AND password = % s', (username, password, ))
                account = cursor.fetchone()
                if account:
                    session['loggedin'] = True
                    session['tid'] = account['tid']
                    session['tname'] = account['tname']
                    session['isTeacher'] = True
                    session['isAdmin'] = True
                    return redirect(url_for('admin_dashboard'))
                else:
                    msg = 'Incorrect Email or password !'
                    flash(msg)
                    return redirect(url_for('login'))
            else:
                cursor.execute('SELECT * FROM teachers WHERE temail = % s AND password = % s', (username, password, ))
                account = cursor.fetchone()
                if account:
                    session['loggedin'] = True
                    session['tid'] = account['tid']
                    session['tname'] = account['tname']
                    session['isTeacher'] = True
                    session['isAdmin'] = False
                    return redirect(url_for('home'))
                else:
                    msg = 'Incorrect Email or password !'
                    flash(msg)
                    return redirect(url_for('login'))
        else:
            cursor.execute('SELECT * FROM students WHERE semail = % s AND student_id = % s', (username, password, ))
            account = cursor.fetchone()
            if account:
                session['loggedin'] = True
                session['student_id'] = account['student_id']
                session['fname'] = account['fname']
                session['isTeacher'] = False
                return redirect(url_for('home'))
            else:
                msg = 'Incorrect Email or password !'
                flash(msg)
                return redirect(url_for('login'))
    else:
        flash(msg)
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
    if 'loggedin' in session:
        return render_template('admin_dashboard.html')
    else:
        return redirect(url_for('login'))

@app.route('/register_student',methods=['GET','POST'])
def register_student():
    msg=''
    if request.method == 'POST' and 'student_id' in request.form and 'fname' in request.form and 'pno' in request.form and 'semail' in request.form and 'optsub1' in request.form :
        sid=request.form['student_id']
        name = request.form['fname']
        pno = request.form['pno']
        semail = request.form['semail']
        val=request.form.get('optsub1')
        if val=="1":
            val="BDA"
        elif val=="2":
            val="SM"
        print(val)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM students WHERE fname = % s', (name, ))
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', semail):
            msg = 'Invalid email address !'
        elif not name or not pno or not semail or not sid:
            msg = 'Please fill out the form !'
        else:
            print("shreya")
            cursor.execute('INSERT INTO students VALUES (% s, % s, % s, % s, % s)', (sid ,name, pno, semail, val, ))
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
    if 'loggedin' in session:
        msg=''
        if request.method == 'POST' and 'name' in request.form and 'sid' in request.form:
            fname = request.form['name']
            face_id=request.form['sid']
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM students WHERE fname = % s', (fname, ))
            account = cursor.fetchone()
            if account:
                session['fname'] = account['fname']
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
                        msg='Successfully Captured'
                        flash(msg)
                        break

                # Stop video
                vid_cam.release()

                # Close all started windows
                cv2.destroyAllWindows()
                msg = "Photos Captured Successfully!!"
                print(msg)
                flash(msg)
                return redirect(url_for('admin_dashboard'))
            else:
                msg = 'Incorrect name'
                flash(msg)
                return redirect(url_for('add_photos'))
        return render_template('add_photos.html')
    else:
        return redirect(url_for('login'))


@app.route('/create_dataset',methods=['GET','POST'])
def create_dataset():
    return render_template('train.html')


@app.route('/train',methods=['GET','POST'])
def train():
    msg = ''
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
    msg = "Successfully trained!!"
    print(msg)
    flash(msg)
    recognizer.write('model1.yml')
    return redirect(url_for('admin_dashboard'))

@app.route('/mark_attendance_details',methods=['GET','POST'])
def mark_attendance_details():
    return render_template('mark_attendance_details.html')

@app.route('/teacher_dashboard',methods=['GET','POST'])
def teacher_dashboard():
    return render_template('teacher_dashboard.html')

@app.route('/student_dashboard',methods=['GET','POST'])
def student_dashboard():
    return render_template('student_dashboard.html')

@app.route('/live',methods=['GET','POST'])
def live():
    #class_names = {1:"Shreya",2:"Shreyatwo", 3:"Muskan",4:"Kumkum", 1903064:"shreya3" , 1903032: "Muskan Gupta"}#,2:"Muskan",3:"Balaji",4:"Tejashree"} #name of people
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    print(current_date)
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
    sub_name="SM"
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
                #print(predictions)
                #value=int(predictions[0])
                #name= class_names[predictions[0]]
                cv2.putText(img, str(class_names[predictions[0]]), (x+5,y-5), font, 1, (255,0,255), 2)
                cv2.putText(img, str(round(predictions[1],2))+"%", (x+5,y+h-5), font, 1, (255,255,0), 1)
                sid=predictions[0]
                #print()
                if class_names[predictions[0]] in class_names.values():
                    val=str(class_names[predictions[0]])
                    print(val)
        
                    #f.loc[value-1, 'Name'] = name
                    #f.loc[value-1, 'Attendance'] = "Present"
                    print("yes")
                
                    #f.to_csv('attendance.csv', index=False)
            cv2.imshow('video',img) 
            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
            
                # If image taken reach 100, stop taking video
    cap.release()
    cv2.destroyAllWindows()
    return render_template('home.html')
                    
            #cv2.imshow('video',img)
            #return render_template('home.html')
                    #current_time = now.strftime("%H-%M-%S")
                #lnwriter.writerow([val,current_time])


@app.route('/upload',methods=['GET','POST'])
def upload():
    # class_names = {1:"Shreya",2:"Shreyatwo", 3:"Muskan",4:"Kumkum", 1903064:"shreya3"}#,2:"Muskan",3:"Balaji",4:"Tejashree"} #name of people
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    print(current_date)
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
   # sub_name="SM"
    if request.method == 'POST':
        # check if the post request has the file part
        if 'studentImg' not in request.files:
            flash('No file part')
            print('No file part')
            return redirect(url_for('mark_attendance_details'))
        Uploaded_file = request.files['studentImg']
        # if user does not select file, browser also
        # submit a empty part without filename
        if Uploaded_file.filename == '':
            flash('No selected file')
            print('No selected file')
            return redirect(url_for('mark_attendance_details'))
        if Uploaded_file and allowed_file(Uploaded_file.filename):
            Uploaded_file.save(os.path.join('./upload_Img', secure_filename(Uploaded_file.filename)))
            print('Image Uploaded successfully!!')
    
            # Read the image
            cap = cv2.imread(f"upload_Img/{Uploaded_file.filename}")

            # Convert it to GrayScale Image
            gray = cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)

            # Using HaarCascasde detect multiple faces
            faces = face_classifier.detectMultiScale(gray,1.3,5) #Scaling factor = 1.3 , minNeighbors = 5

        if request.method == 'POST':
            sub_name=request.form.get('subjopt')
            print(sub_name)  
            for (x,y,w,h) in faces:
                cv2.rectangle(cap,(x,y),(x+w,y+h),(255,0,0),2)
                cropped_image = cap[y:y+h, x:x+w] 
                color_img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            
                # Using LPBH model recognise the detected faces
                predictions = lbph_face_classifier.predict(color_img)
                print(predictions)
                sid=predictions[0]
                name= class_names[predictions[0]]
                cv2.putText(cap, class_names[predictions[0]], (x+5,y-5), font, 1, (255,0,255), 2)
                cv2.putText(cap, str(round(predictions[1],2))+"%", (x+5,y+h-5), font, 1, (255,255,0), 1)
                if class_names[predictions[0]] in class_names.values():
                    val=str(class_names[predictions[0]])
                    print(val)
                    print("yes")

                    
                    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                    cursor.execute('SELECT * FROM students WHERE student_id = % s', (sid, ))
                    account = cursor.fetchone()
                    if account:
                        cursor.execute('INSERT INTO attendance VALUES (NULL, % s, % s, % s,% s, % s, % s)', (current_date,current_time, sid, val,sub_name,"Present"))
                        mysql.connection.commit()

                        #current_time = now.strftime("%H-%M-%S")
                        #lnwriter.writerow([val,current_date])

            # Shows the result and save it as per name given in Results folder 
            cv2.imshow('Recognised Faces',cap)

            
            parent= "Results"
            directory = "LPBH_model_results"
            path=os.path.join(parent,directory)
            os.makedirs(path,exist_ok=True)
            #file_name = input('Enter name to save the result:')
            file_name_path=f"Results/{directory}/{sid}.jpg"
            cv2.imwrite(f'{file_name_path}',cap)

            # Destroy all windows and return to menu
            # cap.release()
            cv2.destroyAllWindows()
            return render_template('home.html')
                    
            #cv2.imshow('video',img)
            #return render_template('home.html')
                    #current_time = now.strftime("%H-%M-%S")
                #lnwriter.writerow([val,current_time])

@app.route('/fetch_Attendance', methods=['GET','POST'])
def fetch_Attendance():
    if request.method == 'POST':
        UID = request.form.get('UID') # [u'Item 1'] []
        Blockchain = request.form.get('Blockchain') # [u'Item 2'] []
        SM = request.form.get('SM') # [u'Item 3'] []
        BDA = request.form.get('BDA')
        startDate = request.form.get('start')
        endDate = request.form.get('end')
        print(UID,Blockchain,SM,BDA,startDate,endDate)
        sd = startDate.split(" ")
        sd1 = sd[0].split('/')
        sd2 = f'{sd1[2]}-{sd1[0]}-{sd1[1]}'
        ed = endDate.split(" ")
        ed1 = ed[0].split('/')
        ed2 = f'{ed1[2]}-{ed1[0]}-{ed1[1]}'
        print(sd2,ed2)
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(f'SELECT * FROM attendance WHERE subject_name = "{UID}" OR subject_name = "{Blockchain}" OR subject_name = "{SM}" OR subject_name = "{BDA}" AND date BETWEEN "{sd2}" AND "{ed2}" ')
        attendanceFetch = cursor.fetchall()
        csv_filename =  f'{sd2}TO{ed2}'
        if SM != None:
            csv_filename = csv_filename + f'_{SM}'
        if Blockchain != None:
            csv_filename = csv_filename + f'_{Blockchain}'
        if BDA != None:
            csv_filename = csv_filename + f'_{BDA}'
        if UID != None:
            csv_filename = csv_filename + f'_{UID}'
        return render_template('teacher_dashboard.html',attendanceFetch = attendanceFetch, csv_filename = csv_filename)
    return render_template('teacher_dashboard.html')

  
@app.route('/excel',methods=['GET','POST'])
def excel():
    # a=[{'attendance_id': 1, 'date': '2023-03-15', 'time': '00:59:55', 'student_id': 1903032, 'student_fname': 'Muskan Gupta', 'subject_name': 'SM', 'attendance': 'Present'}]
    if request.method == 'POST':
        file_name = request.form['csv_filename']
        print(f'Name: {file_name}')
        attendance_fetch = request.form['UI_attend']
        res=ast.literal_eval(attendance_fetch)
        field_names=['attendance_id','date','time','student_id','student_fname','subject_name', 'attendance']
        with open(f'{file_name}.csv','w') as csvfile:
           writer  = csv.DictWriter(csvfile, fieldnames=field_names)
           writer.writeheader()
           writer.writerows(res)

        print("Excel sheet downloaded successfully!!")
        return redirect(url_for('teacher_dashboard'))

    return redirect(url_for('teacher_dashboard'))

@app.route('/fetch_Attendance_student', methods=['GET','POST'])
def fetch_Attendance_student():
    if 'loggedin' in session and session['isTeacher']==False:
        if request.method == 'POST':
            UID = request.form.get('UID') # [u'Item 1'] []
            Blockchain = request.form.get('Blockchain') # [u'Item 2'] []
            SM = request.form.get('SM') # [u'Item 3'] []
            BDA = request.form.get('BDA')
            startDate = request.form.get('start')
            endDate = request.form.get('end')
            # print(UID,Blockchain,SM,BDA,startDate,endDate)
            sd = startDate.split(" ")
            sd1 = sd[0].split('/')
            sd2 = f'{sd1[2]}-{sd1[0]}-{sd1[1]}'
            ed = endDate.split(" ")
            ed1 = ed[0].split('/')
            ed2 = f'{ed1[2]}-{ed1[0]}-{ed1[1]}'
            # print(sd2,ed2)
            login_student=session['fname']
            print(login_student)
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            # cursor.execute(f'SELECT * FROM attendance WHERE subject_name = "{UID}" OR subject_name = "{Blockchain}" OR subject_name = "{SM}" OR subject_name = "{BDA}" AND date BETWEEN "{sd2}" AND "{ed2}" AND ')
            cursor.execute(f'SELECT a.date,a.time,s.fname,a.subject_name FROM attendance a,students s WHERE a.student_id=s.student_id AND (a.subject_name = "{UID}" OR a.subject_name = "{Blockchain}" OR a.subject_name = "{SM}" OR a.subject_name = "{BDA}") AND (a.date BETWEEN "{sd2}" AND "{ed2}") AND s.fname="{login_student}"')
            attendanceFetch = cursor.fetchall()
            print(attendanceFetch)
            csv_filename =  f'{sd2}TO{ed2}'
            if SM != None:
                csv_filename = csv_filename + f'_{SM}'
            if Blockchain != None:
                csv_filename = csv_filename + f'_{Blockchain}'
            if BDA != None:
                csv_filename = csv_filename + f'_{BDA}'
            if UID != None:
                csv_filename = csv_filename + f'_{UID}'
            return render_template('student_dashboard.html',attendanceFetch = attendanceFetch, csv_filename = csv_filename)
        return render_template('student_dashboard.html')
    else:
        return redirect(url_for('login'))

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
    if 'loggedin' in session:
        session.pop('loggedin', None)
        if session['isTeacher']:
            if session['isAdmin']:
                session.pop('isAdmin', None)
            session.pop('tid', None)
            session.pop('tname', None)
            session.pop('isTeacher', None)
        else:
            session.pop('student_id', None)
            session.pop('fname', None)
            session.pop('isTeacher', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, port=5001)