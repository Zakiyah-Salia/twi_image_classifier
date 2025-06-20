# ✅ Base image with Python 3.12 (official)
FROM python:3.12-slim

# ✅ Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ✅ Set working directory
WORKDIR /app

# ✅ Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ✅ Copy project files into container
COPY . .

# ✅ Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ✅ Expose the port Flask will run on
EXPOSE 10000

# ✅ Start the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
