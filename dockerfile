FROM python:3.9
#FROM python:3.12.7

# Exponha a porta padrão do Jupyter Notebook
EXPOSE 8888

# Defina o diretório de trabalho no contêiner
WORKDIR /Users/mhctds/sensoriamentoSocialDashboard

# Atualize o repositório e instale libyaml-dev
#USER root
#RUN apt-get update && apt-get install -y libyaml-dev

# Copie todos os arquivos do diretório atual para o diretório de trabalho no contêiner
#COPY requirements.txt .
# Instale as dependências do requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

# Comando para iniciar o Jupyter Notebook
CMD ["start-notebook.sh"]
