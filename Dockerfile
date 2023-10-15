#Using the base image with Python 3.10
 FROM python:3.10
 
 #Set our working directory as app
 WORKDIR /app 
 COPY requirements.txt requirements.txt
 #Installing Python packages through requirements.txt file
 RUN pip install -r requirements.txt
 
 # Copy the model's directory and server.py files
 ADD ./models ./models
 ADD server.py server.py
 
 #Exposing port 5000 from the container
 EXPOSE 5000
 #Starting the Python application
 CMD ["flask", "run", "--host=0.0.0.0", "--port=5000" ,"server:app"]