FROM ubuntu:20.04

MAINTAINER cocoshe
WORKDIR /ts2
COPY . .
RUN apt-get update && apt-get install -y python3-pip && pip3 install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
EXPOSE 5001
CMD ["python3", "app.py"]