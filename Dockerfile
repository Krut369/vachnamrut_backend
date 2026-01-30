# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependencies first (for better caching)
COPY requirements.txt .

# 4. Install dependencies
# We add --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY . .

# 6. Expose the port (Render usually uses 10000)
EXPOSE 10000

# 7. The command to run the application
# We use host 0.0.0.0 so it is accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]