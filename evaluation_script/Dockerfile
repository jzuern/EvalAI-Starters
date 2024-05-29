FROM python:3.7.5

ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
  apt-get install -q -y default-jre default-jdk vim && \
  rm -rf /var/lib/apt/lists/*

RUN mkdir /code
RUN mkdir /data
WORKDIR /code

ADD requirements.txt /code/
RUN pip install -r requirements.txt

ADD . /code

# Add the wayve_scene_reconstruction_benchmark.zip to the data folder (for local debugging of Docker container)
# ADD wayve_scene_reconstruction_benchmark.zip /data/


CMD ["./docker/wait-for-it.sh", "django:8000", "--", "python", "-m", "scripts.workers.submission_worker"]
