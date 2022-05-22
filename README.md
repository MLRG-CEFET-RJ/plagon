# Plagon
Plagiarism identifier


# Documento de instalação e configuração do framework
## Executar os comandos exatamente como estão, passo-a-passo no terminal, dentro da pasta raiz do projeto:

### Criar banco de dados sqlite3 com os artigos:
	- make build
	- make dev
	- apt update
	- apt install sqlite3
	- pip install nltk
	- python
	- import nltk
	- nltk.download()
	- d
	- punkt
	- q
	- exit()
	- python ./src/pancorpus2db_train.py --srcdir ./pancorpus/train --destfile ./plag_train.db

### Criar modelo skipthoughts:
	- pip install skipthoughts
	- pip install tensorflow==1.13.2
	- python src/create_st_model.py --source train --dest ./
