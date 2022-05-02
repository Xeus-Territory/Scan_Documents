# It not enough to run image with dockerfile, so we need to add some commands into dockerfile
# Hope see some of the commands can addsome to run completely image

FROM python:3

WORKDIR /dev

COPY . .

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirement.txt

ENTRYPOINT ["python"]