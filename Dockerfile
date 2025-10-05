# Use official Python 3.9 image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy your project files
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install exact dependencies required by the model
RUN pip install mlflow==3.1.4 \
    matplotlib==3.9.4 \
    numpy==2.0.2 \
    scikit-learn==1.6.1 \
    scipy==1.13.1 \
    pandas==2.3.3 \
    dash==2.10.0  # optional, for your Dash app

# Expose port for Dash app
EXPOSE 8050

# Set environment variable for Python
ENV PYTHONUNBUFFERED=1

# Run your Dash app
CMD ["python", "app.py"]
