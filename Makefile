install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt
	pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
	pip install jupyter ipykernel
	python -m ipykernel install --user --name=bert-fine-tune --display-name "bert-fine-tune"

generate-requirements:
    pipreqs . --force

build:
	docker build -t bert-fine-tune .

push:
	docker login
	docker tag bert-fine-tune baijiuyang/bert-fine-tune
	docker push baijiuyang/bert-fine-tune

run:
	docker run --gpus all -p 8888:8888 -v $(pwd):/app -it bert-fine-tune

run-on-windows:
	docker run --gpus all -p 8888:8888 -v "%cd%":/app -it bert-fine-tune

run-prebuilt:
	docker run --gpus all -p 8888:8888 -v $(pwd):/app -it baijiuyang/bert-fine-tune

run-prebuilt-on-windows:
	docker run --gpus all -p 8888:8888 -v "%cd%":/app -it baijiuyang/bert-fine-tune

stop:
	docker stop $(docker ps -q --filter ancestor=bert-fine-tune)