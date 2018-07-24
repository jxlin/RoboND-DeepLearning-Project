
# Use this one liner to enable port forwarding for your remote notebooks
# taken from here : https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh

## TO ENABLE PORT FORWARDING 
# ssh -N -f -L localhost:PORT_LOCAL:localhost:PORT_REMOTE 'remote_name'@'remote_ip'
ssh -N -f -L localhost:8888:localhost:8889 lab@192.168.0.167

## TO KILL THE PROCESS
# Find it -> ps aux | grep localhost:PORT_REMOTE
ps aux | grep localhost:8889
# kill it manually
kill -15 PROCESS_ID
