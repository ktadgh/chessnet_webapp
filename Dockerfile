#syntax=docker/dockerfile:1
FROM docker
COPY --from=docker/buildx-bin /buildx /usr/libexec/docker/cli-plugins/docker-buildx
RUN docker buildx version

FROM python:3.10-slim


COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

# COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html


COPY . /app
ENV FLASK_APP="application.py"
EXPOSE 5000
#CMD ["python","application.py"]
CMD [ "flask", "run","--host","0.0.0.0","--port","5000"]