from flask import *
import mysql.connector
import cv2
import torch
import os
import numpy as np
from torchvision import transforms
from facenet_pytorch import MTCNN,InceptionResnetV1
import time
import pickle
from PIL import Image
from random import *
from email_otp import *
import base64
from Crypto.Cipher import AES
from Crypto.Util import Counter
from Crypto import Random

BS = 16
pad = lambda s: bytes(s + (BS - len(s) % BS) * chr(BS - len(s) % BS), 'utf-8')
unpad = lambda s : s[0:-ord(s[-1:])]

class AESCipher:

    def __init__( self, key ):
        self.key = bytes(key, 'utf-8')

    def encrypt( self, raw ):
        raw = pad(raw)
        iv = Random.new().read( AES.block_size )
        cipher = AES.new(self.key, AES.MODE_CBC, iv )
        return base64.b64encode( iv + cipher.encrypt( raw ) )

    def decrypt( self, enc ):
        enc = base64.b64decode(enc)
        iv = enc[:16]
        cipher = AES.new(self.key, AES.MODE_CBC, iv )
        return unpad(cipher.decrypt( enc[16:] )).decode('utf8')

cipher = AESCipher('mysecretpassword')


app = Flask(__name__)
app.secret_key = 'EmailAuthenticationByShivamYadav2021'
mydb = mysql.connector.connect(host="localhost",user="root",password="",database="vote")
mycursor = mydb.cursor()


@app.route("/")
def homepage():
    return render_template('index.html')
@app.route("/AdminLogin")
def AdminLogin():
    return render_template('block.html')
@app.route("/logi")
def logi():
    return render_template('login.html')
@app.route("/AtmLogin")
def AtmlLogin():
    return  render_template('atmlogin.html')
@app.route("/adlog")
def adlog():
    return  render_template("admin.html")

@app.route('/val',methods=['GET', 'POST'])
def val():
    if request.method == 'POST':
        if request.form.get('username') == 'abc' and request.form.get('password') == '123':
            return render_template('admin.html')
        else:
            return render_template('login.html', msg='Invalid Username or Password')

    else:
        return render_template('login.html')

@app.route("/Admin", methods=['GET', 'POST'])
def Admin():
    #global email
    global encrypted
    if request.method == 'POST':
        card = request.form['name']
        un = request.form['ename']
        pnum=request.form['pnum']
        add=request.form['add']
        email = request.form['email']
        encrypted = cipher.encrypt(pnum)
        encrypted1 = cipher.encrypt(add)
        sql = "INSERT INTO regtb (`card`, `un`, `pnum`, `add`, `email`) VALUES (%s, %s, %s, %s, %s)"
        val = (card, un,encrypted,encrypted1, email)
        mycursor.execute(sql, val)
        mydb.commit()


        image_size = 600
        frame_rate = 64
        vid_len = 20
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu")

        # Save all face images of a person as a pickle file
        def save_face_images(frames, boxes):
            transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor()
            ])
            for f in range(len(frames)):
                img = np.asarray(frames[f])
                box = boxes[f]
                if len(box.shape) == 3:
                    # Go into loop only when there is atleast 1 face in image
                    # Loop for num of boxes in each image
                    for b in range(box.shape[1]):
                        start = (np.clip(int(box[0][b][0]) - 15, 0, 480), np.clip(int(box[0][b][1]) - 50, 0, 640))
                        end = (np.clip(int(box[0][b][2]) + 15, 0, 480), np.clip(int(box[0][b][3]) + 20, 0, 640))
                        crop_pic = img[start[1]:end[1], start[0]:end[0]]
                    img_crop = Image.fromarray(crop_pic)
                    img_crop = transform(img_crop)
                    img_crop = torch.unsqueeze(img_crop, 0)
                    save_tensor = model(img_crop)
                    return save_tensor

        v_cap = cv2.VideoCapture(0)
        v_cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size)
        v_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size)
        count = 1
        prev = 0
        try:
            os.mkdir(card)
        except FileExistsError:
            pass

        mtcnn = MTCNN(image_size=image_size, keep_all=True, device=device, post_process=True)
        model = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
        start = time.time()
        frames = []
        boxes = []
        print(
            'Try to keep your face at the centre of the screen and turn ur face slowly in order to capture diff angles of your face')
        time.sleep(3)
        print('A window will pop up in abt 3 seconds')
        time.sleep(3)
        save_tensor = None

        # 20 sec loop to input truth face images
        while True:
            time_elapsed = time.time() - prev
            curr = time.time()
            if curr - start >= vid_len:
                break
            ret, frame = v_cap.read()
            cv2.imshow('Recording and saving Images', frame)
            if time_elapsed > 1. / frame_rate:  # Collect frames every 1/frame_rate of a second
                prev = time.time()
                frame_ = Image.fromarray(frame)
                frames.append(frame_)
                batch_boxes, prob, landmark = mtcnn.detect(frames, landmarks=True)
                frames_duplicate = frames.copy()
                boxes.append(batch_boxes)
                boxes_duplicate = boxes.copy()
                # show imgs with bbxs
                if save_tensor == None:
                    save_tensor = save_face_images(frames_duplicate, boxes_duplicate)
                else:
                    temp = save_face_images(frames_duplicate, boxes_duplicate)
                    if temp is not None:
                        save_tensor = torch.cat([temp, save_tensor], dim=0)
                        print(save_tensor.shape)
                count += 1
                frames = []
                boxes = []
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Open file for pickling
        face_file = open(card + '/' + un, 'ab')
        pickle.dump(save_tensor, face_file)
        face_file.close()
        v_cap.release()
        cv2.destroyAllWindows()



        return render_template('facecam.html')

@app.route('/face', methods=['GET', 'POST'])
def face():
    global data1
    if request.method == 'POST':
        card_number = request.form.get('card')

        #sql = "SELECT * FROM `facetb` WHERE `card` = %s AND `email` = %s"

        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ])


        frame_rate = 16
        prev = 0
        batch_size = 32
        image_size = 600
        threshold = 0.85
        device = device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu")
        bbx_color = (0, 255, 255)

        current_person = None

        def detect_imgs(img):
            global current_person
            person_ = None
            img = transform(img)
            img = torch.unsqueeze(img, 0)
            img = model(img)
            minimum = torch.tensor(99)
            for face_, name in zip(faces, face_names):
                temp = torch.min(torch.norm((face_ - img), dim=1))
                if temp < minimum and temp < threshold:
                    minimum = temp
                    person_ = name
                    current_person = name
            return person_, minimum.item()

        def show_images(frames, boxes, color):
            temp = None
            for f in range(len(frames)):
                img = np.asarray(frames[f])
                box = boxes[f]
                if len(box.shape) == 3:
                    # Go into loop only when there is atleast 1 face in image
                    # Loop for num of boxes in each image
                    for b in range(box.shape[1]):
                        start = (np.clip(int(box[0][b][0]) - 15, 0, 600), np.clip(int(box[0][b][1]) - 20, 0, 600))
                        end = (np.clip(int(box[0][b][2]) + 15, 0, 600), np.clip(int(box[0][b][3]) + 15, 0, 600))
                        img = cv2.rectangle(img, start, end, color, 2)
                        crop_pic = img[start[1]:end[1], start[0]:end[0]]
                        crop_pic = Image.fromarray(crop_pic)
                        person, diff = detect_imgs(crop_pic)
                        if person is not None:
                            cv2.putText(img, person + ': ' + '{:.2f}'.format(diff), (start[0], start[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                            temp = 1
                        else:
                            cv2.putText(img, 'Unknown' + ': ' + '{0}'.format(diff), (start[0], start[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                            temp = 0
                cv2.imshow('Detection', img)
                if temp == 1:
                    return 1
                else:
                    return 0

        # Init MTCNN object
        mtcnn = MTCNN(image_size=image_size, keep_all=True, device=device, post_process=True)
        model = InceptionResnetV1(pretrained='vggface2', classify=False).eval()
        # Real time data from webcam
        frames = []
        boxes = []

        # Load stored face data related to respective card number
        faces = []
        face_names = []
        face_file = None
        try:
            for person in os.listdir(card_number):
                face_file = open(card_number + '/' + person, 'rb')
                if face_file is not None:
                    face = pickle.load(face_file)
                    faces.append(face)
                    face_names.append(str(person))
        except FileNotFoundError:
            print('Face data doesnt exist for this card.')
            exit()

        # Infinite Face Detection Loop
        v_cap = cv2.VideoCapture(0)
        v_cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size)
        v_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size)
        flag = False
        face_results = []
        start = time.time()
        while (True):
            time_elapsed = time.time() - prev
            break_time = time.time() - start
            if break_time > 10:
                break
            ret, frame = v_cap.read()
            if time_elapsed > 1. / frame_rate:  # Collect frames every 1/frame_rate of a second
                prev = time.time()
                frame_ = Image.fromarray(frame)
                frames.append(frame_)
                batch_boxes, prob, landmark = mtcnn.detect(frames, landmarks=True)
                frames_duplicate = frames.copy()
                boxes.append(batch_boxes)
                boxes_duplicate = boxes.copy()
                # show imgs with bbxs
                face_results.append(show_images(frames_duplicate, boxes_duplicate, bbx_color))
                frames = []
                boxes = []
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        v_cap.release()
        cv2.destroyAllWindows()
        accuracy = (sum(face_results) / len(face_results)) * 100
        print('Percentage match ' + '{:.2f}'.format(accuracy))
        if accuracy > 50:
            print('Authorization Successful')
            sql = "SELECT email FROM `regtb` WHERE `card` =%s "
            val = (card_number,)
            mycursor.execute(sql, val)
            account = mycursor.fetchone()
            acc = account[0]


            current_otp = sendEmailVerificationRequest(
                receiver=acc)  # this function sends otp to the receiver and also returns the same otp for our session storage
            session['current_otp'] = current_otp
            return render_template('verify.html')



        else:
            print('Authorization Unsuccessful')
            return render_template('Unauthorization.html')
            quit()


@app.route('/validate', methods=["POST"])
def validate():
    # Actual OTP which was sent to the receiver
    current_user_otp = session['current_otp']
    print("Current User OTP", current_user_otp)

    # OTP Entered by the User
    user_otp = request.form['otp']
    print("User OTP : ", user_otp)

    if int(current_user_otp) == int(user_otp):
        return render_template('voting.html')
    else:
        return "<h3> Oops! Email Verification Failure, OTP does not match. </h3>"

@app.route('/add',methods=['POST','GET'])
def add():
    if request.method == 'POST':
        cname = request.form.get('name')
        sql = 'SELECT * FROM `counts` WHERE `name` = %s'
        val = (cname,)
        mycursor.execute(sql, val)
        result = mycursor.fetchall()
        if result:
            for row in result:
                num = int(row[2])
                print(num)
                sql1 = 'UPDATE `counts` SET `count` = %s WHERE `name` = %s'
                val1 = (num+1, cname)
                mycursor.execute(sql1, val1)
                mydb.commit()
                print(num + 1)
                return render_template('success.html',msg = 'Vote Added Successfully')
        else:
            return 'No Data'

@app.route('/count')
def count():
    sql = "SELECT * FROM `counts`"
    mycursor.execute(sql)
    result = mycursor.fetchall()
    return render_template('count.html',data=result)
@app.route('/view')
def view():
    sql = "SELECT * FROM regtb"
    mycursor.execute(sql)
    result = mycursor.fetchall()
    return render_template('view.html',data=result)



if __name__ == '__main__':
    app.run(debug=True)