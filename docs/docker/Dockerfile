### Sting server Dockerfile example
### Use Sting Server as a flexible virtual machine

#Build from a Ubuntu 18.04 image with cuda installed
FROM nvidia/cuda:11.2.0-devel-ubuntu18.04

MAINTAINER MAGICS LAB <magics@cs.northwestern.edu>
#Install wanted softwares
RUN apt-get update  && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        git \
        vim \
    openssh-server \
    software-properties-common \
    python3 python3-dev python3-pip
#You can install libraries that you want here
#RUN pip3 install torch torchvision

#Since Ubuntu default has no password for root, and prohibit root ssh login
#Set a password for root, and allow root ssh permission
RUN echo 'root:Docker' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

#You can set global enviroment variables

#ENV KEY1=KEY2 \
#    KEY2=VALUE2


#Copy your code or script in
#COPY yourCode.py /path/yourCcode.py
COPY startup.sh /startup.sh

#Script that you want this container to run when started
ENTRYPOINT ["bash", "/startup.sh"]


