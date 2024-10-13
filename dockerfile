FROM jupyter/datascience-notebook
EXPOSE 80
WORKDIR /Users/mhctds/sensoriamentoSocialDashboard
COPY ./requirements.txt .
#RUN python -m pip install --no-cache -r ./requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80", "--workers","4"]