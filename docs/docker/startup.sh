#Reload ssh configurations when started
service ssh restart

#This line simply makes container keep running
tail -f /dev/null

docker start -i 18175765b9c4

docker exec -it 18175765b9c4 bash

docker cp 18175765b9c4:/projs/Bopt/result/RMSprop/minst.py/ /Users/zz/downloads/zzprojects/


docker run -it bopt:v1 python experiments/minst.py --opt RMSprop
