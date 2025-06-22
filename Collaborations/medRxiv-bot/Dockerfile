# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Set environment variables (optional, if not using a .env file in production)
# ENV AZURE_LLAMA_KEY=<your_key>
# ENV AZURE_LLAMA_ENDPOINT=<your_endpoint>
# ENV SEECHAT_API_KEY=<your_api_key>
# ENV SEMANTIC_SCHOLAR_API_KEY=<your_api_key>

# Command to run the application
CMD ["python", "medRxiv_Bot.py"]