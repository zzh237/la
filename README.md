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

Adam:
<p float="left">
<img src="https://github.com/zzh237/la/blob/main/docs/result_exp_1.jpg" width="500" height="350">
<img src="https://github.com/zzh237/la/blob/main/docs/result_exp_2.jpg" width="500" height="350">  
</p>

SGD:
<p float="left">
<img src="https://github.com/zzh237/la/blob/main/docs/result_SGD_exp_1.jpg" width="500" height="350">
<img src="https://github.com/zzh237/la/blob/main/docs/result_SGD_exp_2.jpg" width="500" height="350">  
</p>



