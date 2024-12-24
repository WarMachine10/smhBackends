# Use Python 3.12 as the base image
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app/backend

# Copy the requirements.txt file to the container
COPY Requirements.txt /app/backend/

# Install Python dependencies using pip
RUN pip install --no-cache-dir -r Requirements.txt

# Install Gunicorn
RUN pip install gunicorn

# Copy the entire application code into the container
COPY . /app/backend/

# Expose port 8000 for Gunicorn
EXPOSE 8000

# Create the start script to launch Gunicorn and apply migrations at runtime
RUN echo '#!/bin/bash\n\
python manage.py migrate\n\
gunicorn --bind 0.0.0.0:8000 SKM:application' > /app/backend/start.sh

# Make the start script executable
RUN chmod +x /app/backend/start.sh

# Run the start script when the container starts
CMD ["/app/backend/start.sh"]
