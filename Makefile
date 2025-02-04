install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

generate-requirements:
    pipreqs . --force

build:
	docker build -t bert-fine-tune .

run:
	docker run -p 8888:8888 -v $(PWD):/app -it bert-fine-tune

stop:
	docker stop $(docker ps -q --filter ancestor=bert-fine-tune)