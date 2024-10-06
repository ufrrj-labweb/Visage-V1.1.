FROM python:3.7
EXPOSE 80
WORKDIR /opt
COPY requirements.txt ./
RUN pip install -r ./requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80", "--workers","4"]