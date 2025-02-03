install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

generate-requirements:
    pipreqs . --force

build:
	docker build -t $(DOCKER_IMAGE_NAME) .