# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code into the container
COPY . .

# Expose the port your app will run on
# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Run main.py using uvicorn when the container launches
# We bind to 0.0.0.0 to make it accessible
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]