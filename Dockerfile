#Using the base image with Python 3.10
 FROM python:3.10
 
 #Set our working directory as app
 WORKDIR /app 
 COPY requirements.txt /app
 #Installing Python packages through requirements.txt file
 RUN pip install -r requirements.txt
 

 # Copy the model's directory and server.py files
 COPY . .
 ADD ./models ./models

 #Exposing port 5000 from the container
 EXPOSE 5000

 ENV FLASK_APP=server.py
 #Starting the Python application
 CMD ["flask", "run", "--host=0.0.0.0", "--port=5000" ]