# bert-fine-tune-classification
Please see task demonstration in `notebook.ipynb`.
# Ways to run the notebook:
- **Option 1**: Run the pre-built docker container
    - cd to repo root
    - run `make run-prebuilt` (or `make run-prebuilt-on-windows`)
- **Option 2**: Build and run docker container
    - cd to repo root
    - run `make build`
    - run `make run` (or `make run-on-windows`)
- **Option 3**: Run without docker
    - cd to repo root
    - run `make`
    - open `notebook.ipynb`
    - choose the kernel `bert-fine-tune`
