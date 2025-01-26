FROM python:3.12-slim

# Install PostgreSQL development libraries required by psycopg2
RUN apt-get update && apt-get install -y libpq-dev build-essential


# Set the working directory within the container
WORKDIR /api-flask

# Copy the necessary files and directories into the container
COPY . /api-flask/

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Expose port 5000 for the Flask application
EXPOSE 5000

HEALTHCHECK CMD curl --fail http://localhost:5000/ || exit 1

CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000", "-w", "4"]
