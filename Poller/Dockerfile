# Use an official Python runtime as a parent image
FROM python

# Set the working directory to /app
WORKDIR /app

# Copy the project files into the container
COPY ElasticSeachHandle /app/Poller/ElasticSeachHandle
COPY RecommenderSystem /app/Poller/RecommenderSystem
COPY poller.py /app/Poller/poller.py
COPY poller_dev.py /app/Poller/poller_dev.py
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN pip install gunicorn

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the Flask app with Gunicorn and 4 worker processes
CMD ["gunicorn", "-b", "0.0.0.0:80", "-w", "4", "Poller.poller:Rec"]