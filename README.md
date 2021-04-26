# la (Learning Algorithm)
the bayesian optimization algorithms and experiments

## for running, clone the repo to your local machine
build docker image:
docker build -t bopt:v1 . 

run the docker container:
docker run -it bopt:v1 

run the container with argument, ie:
docker run -it bopt:v1 python experiments/minst.py --opt RMSprop

The experiments/minst.py is the entry point of the app, you don't need to change. 

The output of the app is:
1. The loss over the training step
2. The evalulation of the best model
The output are in tensorboard, such as:




