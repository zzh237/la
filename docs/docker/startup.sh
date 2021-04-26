#Reload ssh configurations when started
service ssh restart

#This line simply makes container keep running
tail -f /dev/null
