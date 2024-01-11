# Base Image
FROM python:3.8

# Defining the workspace that we're going to be
# working in
WORKDIR /src/novel-crawlers

# Updating the dependencies
RUN pip install -U \
    pip

# Install Dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy files
COPY . .

# Marking the Docker container
ARG GIT_HASH
ENV GIT_HASH=$(GIT_HASH:-dev)

RUN mkdir -p /data/files && chown -R 65534:65534 /data

VOLUME ["/data/files"]

# Exposing the port
EXPOSE 10000

# Defining the entrypoint
ENTRYPOINT ["python", "./src/main.py"]
