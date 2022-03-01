
# Tutorials of Using MPI on Amazon EC2

This is the tutorial of building a distributed computing system based on MPI and Amazon EC2.

## Instructions on configurations of Amazon EC2 computing instance
Here are instructions to install MPI and mpi4py packages and configure the ssh service to enable communications between instances. You can also directly use public AMI image on Amazon EC2, which is ready to use.

### An ready-to-use public Amazon EC2 AMI
Make sure to select area US East (Ohio) us-east-2. This AMI is anonymous and committed as it is.

```
AMI ID:  ami-05a7b2deb8470f517

owner:  753214619702
```

### Steps to configure an Amazon EC2 instance

- Install MPI
```
sudo apt install openmpi-bin openmpi-dev openmpi-common openmpi-doc libopenmpi-dev
```

- Install mpi4py
```
pip3 install mpi4py
```

- Configure ssh to enable passwordless login between instances
Please see this link as a reference : https://www.tecmint.com/ssh-passwordless-login-using-ssh-keygen-in-5-easy-steps/

- Configure ssh to enable login without manually entering 'yes': Please see this link as a reference: https://unix.stackexchange.com/questions/33271/how-to-avoid-ssh-asking-permission

After you configured the instance, you can commit the configurations and make an AMI for later usage.

### Edit security groups when you create instances
Edit the security groups to allow all traffic for your instance, so that ssh can work.




## Shell scripts for coordinating distributed computing systems

### Create instances and extract IP address information
Create instances on the Amazon EC2 website to build a distributed computing system, which consists of a master node and multiple worker nodes. Copy the Amazon instances information into the file `amazon_instances_info`, for example:
```
–	i-0f7f74cf0420a75a3	c5n.large	52.15.165.68	2021/08/08 09:11 GMT-7
–	i-003c00df228882bfd	c5n.large	18.218.185.168	2021/08/08 09:11 GMT-7
–	i-062f6166ac671a077	c5n.large	52.15.88.70	2021/08/08 09:11 GMT-7
```
Here are three instances in total and let the first instance to be the master node and the rest two instances are worker nodes. We need the IP address information for communications. In the example above, 52.15.165.68 is the IP address of the master node. We run the following script to get IP addresses of all nodes and store it into the file `nodeIPaddress`
```
./get_ip_address.sh
```
or
```
awk '{ print $4 }' amazon_instances_info > nodeIPaddress
```
which extracts strings of the 4th column of the file `amazon_instances_info` and store them in the file `nodeIPaddress`.



### Login master node
Run the scipt to login the master node
```
./login_master.sh 1
```
The index after the command indicates i-th node. In this case, '1' stands for the first node or master node. The content of the `login_master.sh`:
```
#!/usr/bin/ksh
ARRAY=()
while read LINE
do
    ARRAY+=("$LINE")
done < nodeIPaddress
ssh  -i ~/AmazonEC2/.ssh/linux_key_pari.pem ubuntu@${ARRAY[$1]}
```
in which `~/AmazonEC2/.ssh/linux_key_pari.pem` is the key pair permission generated and downloaded when you create an instance on Amazon EC2. In this case, the name of my key-pair is `linux_key_pari.pem` and placed in the directory `~/AmazonEC2/.ssh/`. You need to change it correspondingly based on your cases. You may also need to install `ksh` libraries using `apt-get install ksh` to run the script.

### Run the MPI program in the master node
The next step is to run the MPI program in the master node. Before running the MPI program, for example, `train_darl1n.py`. You need it to put the `train_darl1n.py` in the same directory of all nodes.  (You can use the shell script `transferFile.sh` to upload files from local host to Amazon EC2, which will be explained later.). Then you can run the script
```
./run_spread.sh
```
or other similar scripts. The file `nodeIPaddress` should be in the same directory with `run_spread.sh` in the master node.

### Helper scripts

- `./transfer.sh`: transfer files from the local host to the node listed in the file `nodeIPaddress`, for example,
`./transfer.sh /home/smile/aaai_darl1n/setup.py /home/ubuntu/aaai_darl1n/` transfers file `/home/smile/aaai_darl1n/setup.py` on the local host to the Amazon EC2 instance directory `/home/ubuntu/aaai_darl1n/`.


- `./download_file.sh`: download files from the Amazon EC2 instance to the local host.

- `./ExecuteCommandAllNodes.sh`: execute a command in all nodes
```
i=1
filename='nodeIPaddress'
while IFS= read -r line
do
echo $line
echo "Execution Done!"
ssh -i  ~/AmazonEC2/.ssh/linux_key_pari.pem -n ubuntu@$line 'killall python3'
((i=i+1))
done < $filename
```
which executes command `killall python3` in all nodes.
