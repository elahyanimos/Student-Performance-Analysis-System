# Use an official Python image as the base
FROM python:3.11.0

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files from the current directory into the container
COPY . .

# Expose port 5000 for Flask application (adjust if needed)
EXPOSE 5000

# Command to run the application
CMD ["python", "main.py"]

