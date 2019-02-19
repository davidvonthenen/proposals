FROM python:3
COPY server.py /
RUN mkdir /web
COPY name.txt /web
CMD [ "python", "-u", "server.py" ]
