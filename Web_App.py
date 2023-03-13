import streamlit as st
import pandas as pd
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input as ResNet50_preprocess_input
import pyttsx3
import hashlib
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64

#Database Connections
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()

#Loaded model is ResNet50
model = tf.keras.models.load_model('model.h5')

#Class Names
map_dict = {0: 'strong negative correlation',
            1: 'strong positive correlation',
            2: 'weak negative correlation',
            3: 'weak positive correlation',
            4: 'non correlation'
            }

#Hash Functions
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(name TEXT, username TEXT NOT NULL PRIMARY KEY, email TEXT, password TEXT, user_type TEXT)')

def create_resulttable():
	c.execute('CREATE TABLE IF NOT EXISTS resultstable(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, result TEXT, FOREIGN KEY(username) REFERENCES userstable(username))')

def create_imagetable():
	c.execute('CREATE TABLE IF NOT EXISTS imagestable(image TEXT, result_id INTEGER, FOREIGN KEY(result_id) REFERENCES resultstable(id))')

def get_result_id():
	c.execute('SELECT MAX(id) FROM resultstable')
	data = c.fetchall()
	return data

def add_userdata(name,username,email,password):
	user="user"
	c.execute('INSERT INTO userstable(name,username,email,password,user_type) VALUES (?,?,?,?,?)',(name,username,email,password,user))
	conn.commit()

def add_resultdata(username,result):
	user="user"
	c.execute('INSERT INTO resultstable(username,result) VALUES (?,?)',(username,result))
	conn.commit()

def add_imagedata(image,result_id):
	user="user"
	c.execute('INSERT INTO imagestable(image, result_id) VALUES (?,?)',(image, result_id))
	conn.commit()

def add_admin():
	password = make_hashes("123")
	name="Admin"
	admin="admin"
	email = "admin@scatter.com"
	c.execute('INSERT INTO userstable(name,username,email,password,user_type) VALUES (?,?,?,?,?)',(name,name,email,password,admin))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def control_admin(username,password):
	c.execute('SELECT user_type FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

def view_all_results():
	c.execute('SELECT * FROM resultstable')
	data = c.fetchall()
	return data

def view_all_images():
	c.execute('SELECT username, image, result FROM imagestable INNER JOIN resultstable ON imagestable.result_id = resultstable.id')
	data = c.fetchall()
	return data

def view_user_images(username,password):
	c.execute('SELECT image, result FROM imagestable INNER JOIN resultstable ON imagestable.result_id = resultstable.id INNER JOIN userstable ON resultstable.username = userstable.username WHERE userstable.username = ? AND userstable.password = ?',(username,password))
	data = c.fetchall()
	return data


#Prediction function
def predict(username):
	create_resulttable()
	create_imagetable()
	uploaded_file = st.file_uploader("Choose an image file", type="png")
	if uploaded_file is not None:
		file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
		opencv_image = cv2.imdecode(file_bytes, 1)
		opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
		resized = cv2.resize(opencv_image,(160,160))
		st.image(opencv_image, channels="RGB")
		resized = ResNet50_preprocess_input(resized)
		img_reshape = resized[np.newaxis,...] 
		prediction = model.predict(img_reshape).argmax()
		st.subheader("This scatter plot image has {}.".format(map_dict [prediction]))
		add_resultdata(username, map_dict [prediction])
		result_id = get_result_id()
		img_str = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
		add_imagedata(img_str,int(result_id[0][0]))

		Audio_Button = st.button("Audio") 
		if Audio_Button:
			engine = pyttsx3.init()
			newVoiceRate = 145
			engine.setProperty('rate',newVoiceRate)
			engine.say("This scatter plot image has {}.".format(map_dict [prediction]))
			engine.runAndWait()


def main():
	count = 0
	st.title("Scatter Plot Classifier")
	menu = ["Home","Login","SignUp"]		
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Welcome to Scatter Plot Classifier!")
		st.subheader("If you have an account you can Login or you can create a new account.")

	elif choice == "Login":
		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			create_usertable()
			
			if view_all_users()==[]:
				add_admin()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:

				st.success("Logged in as {}".format(username))
				if(control_admin(username,hashed_pswd)[0]==('admin',)): #When user is admin, the menu changes according to it.
					task = st.selectbox("Task",["Home","Profiles","Results","Prediction", "Images"])
					if task == "Home":
						st.subheader("Welcome {}!".format(username))
						
					elif task == "Profiles": #To view all users
						st.subheader("User Profiles")
						user_result = view_all_users()
						clean_db = pd.DataFrame(user_result,columns=["Name","Username","Email","Password", "User Type"])
						st.dataframe(clean_db)
					
					elif task == "Results": #To view all results
						st.subheader("Results")
						user_result = view_all_results()
						clean_db = pd.DataFrame(user_result,columns=["Id","Username", "Result"])
						st.dataframe(clean_db)

					elif task == "Prediction": #To predict the uploaded picture's class.
						st.subheader("Prediction")
						predict(username)

					elif task == "Images": #To view all the images with the usernames and labels.
						st.subheader("Previous Images")
						for k in view_all_images():
							st.subheader("User: {}".format(k[0]))
							st.subheader("Result: {}".format(k[2]))
							i = base64.b64decode(k[1])
							i = io.BytesIO(i)
							i = mpimg.imread(i, format='JPG')
							st.image(i, channels="RGB")
							st.subheader("------------------------------------------------------------------------")
							
				elif(control_admin(username,hashed_pswd)[0]==('user',)): #When user is user, the menu changes according to it.
					task = st.selectbox("Task",["Home","Prediction", "Images"])
					if task == "Home":
						st.subheader("Welcome!")

					elif task == "Prediction": #To predict the uploaded picture's class.
						st.subheader("Prediction")
						predict(username)

					elif task == "Images": #To view all the images that current user uploaded.
						st.subheader("Images")
						for k in view_user_images(username,hashed_pswd):
							st.subheader("Result: {}".format(k[1]))
							i = base64.b64decode(k[0])
							i = io.BytesIO(i)
							i = mpimg.imread(i, format='JPG')
							st.image(i, channels="RGB")
							st.subheader("------------------------------------------------------------------------")
							
				else:
					st.subheader(control_admin(username,hashed_pswd)[0])
			else:
				st.warning("Incorrect Username/Password")

	#Sign Up function
	elif choice == "SignUp": 
		st.subheader("Create a New Account")
		new_name = st.text_input("Name")
		new_email = st.text_input("E-mail")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')
		
		if st.button("Signup"):
			create_usertable()
			add_userdata(new_name,new_user,new_email,make_hashes(new_password))
			st.success("You have successfully created a valid account")
			st.info("Go to Login Menu to login")


if __name__ == '__main__':
	main()