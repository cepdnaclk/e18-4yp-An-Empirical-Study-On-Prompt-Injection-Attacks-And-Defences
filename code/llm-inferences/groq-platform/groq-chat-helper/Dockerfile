# Use the official Python base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy data folder
COPY data/ .

# Copy the dataset creating script
COPY dataset_from_templates.py .

# Generate the prompt dataset
RUN dataset_from_templates.py

# Copy the Python script
COPY groq_chat_helper.py .

# Set the entry point
CMD ["python", "groq_chat_helper.py"]