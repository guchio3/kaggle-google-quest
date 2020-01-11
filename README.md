# kaggle-google-quest
### env settings and its basic usage
 1. build kaggle gpu image in your local env (because it's based on kaggle gpu image, which does not exist some-hub officially)
     - `git clone git@github.com:Kaggle/docker-python.git; cd docker-python; ./build --gpu` 
 1. clone this repo
     - `cd; git clone git@github.com:guchio3/kaggle-google-quest.git; cd kaggle-google-quest`
 1. run commands
     - shell           : `docker-compose run shell`
         - ex. debug using pudb
     - python commands : `docker-compose run python {something.py}`
         - train : ``
         - predict : ``
     - notebooks       : `docker-compose run --service-ports jn`


### install packages into kaggle notebook
 1. pip download -d ${SOME_DIR_NAME} ${PACKAGE_NAME} (on docker env)
 1. kaggle d init; vi dataset-metadata.json; kaggle d create (or version)
 1. (on kaggle notebook) !pip install --no-deps /kaggle/input/${YOUR_DATASET_NAME}/*
