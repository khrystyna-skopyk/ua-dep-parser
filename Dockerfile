FROM python:3.7-buster
COPY ./ /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python api/data_loader.py

EXPOSE 5000

CMD ['python', 'app.py']