FROM python:3.7

ADD . /projs/Bopt

ENV PYTHONPATH=/projs/Bopt \
	TERM=xterm 

WORKDIR /projs/Bopt

RUN pip install -r requirements.txt

CMD python experiments/minst.py
