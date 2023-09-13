# myapp.conf

# Bind the application to the specified host and port
bind = "0.0.0.0:8000"

# Number of worker processes to spawn
# You can adjust this based on your server's capabilities
workers = 4

# Number of threads per worker process (optional)
# You can adjust this based on your application's needs
threads = 2

# Set the application module or script
# Replace 'your_app_module' with the actual name of your application module or script
# For example, if you're using a Flask app, you might set this to "your_app:app"
# For a Django app, you might set this to "your_project.wsgi:application"
module = "Poller.poller:app"

# The following settings are optional and can be adjusted as needed

# Daemonize the Gunicorn process (run in the background)
# daemon = true

# Log file locations
# errorlog = "/path/to/error.log"
# accesslog = "/path/to/access.log"