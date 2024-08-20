# Use Python 3.9 as the base image
FROM python:3.12

# Install Nginx
RUN apt-get update && apt-get install -y nginx

# Set the working directory within the container
WORKDIR /app/backend

# Copy the requirements.txt file to the container
COPY Requirements.txt /app/backend/

# Install Python dependencies using pip
RUN pip install -r Requirements.txt

# Install Gunicorn
RUN pip install gunicorn

# Copy the entire application code to the container
COPY . /app/backend/

# Copy Nginx configuration
COPY nginx.conf /etc/nginx/sites-available/default

# Expose port 80 for Nginx
EXPOSE 80

# Apply migrations to set up the database (SQLite in this case)
RUN python manage.py migrate

# Create a start script
RUN echo '#!/bin/bash\n\
nginx\n\
gunicorn --bind 0.0.0.0:8000 SKM:application' > /app/backend/start.sh

RUN chmod +x /app/backend/start.sh

# Run the start script
CMD ["/app/backend/start.sh"]