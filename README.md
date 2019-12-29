# kaggle-google-quest
### env settings and its basic usage
 1. build kaggle gpu image in your local env (because my env based on kaggle gpu image, which does not exist some online-hub)
     - `git clone git@github.com:Kaggle/docker-python.git; cd docker-python; ./build --gpu` 
 1. clone this repo
     - `cd; git clone git@github.com:guchio3/kaggle-google-quest.git; cd kaggle-google-quest`
 1. run commands
     - python commands : `docker-compose run python {something.py}`
     - notebooks       : `docker-compose run --service-ports jn`
