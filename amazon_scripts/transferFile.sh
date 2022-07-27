#!/bin/sh
# pass the ssh public key of host to ec2 instances


filename='nodeIPaddress'
while read line; do
echo $line
#scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem $1 ubuntu@$line:~/IROS22_DARL1N/mpe_local/multiagent/scenarios/
#scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem $1 ubuntu@$line:~/IROS22_DARL1N/amazon_scripts/
#scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem $1 ubuntu@$line:~/IROS22_DARL1N/maddpg_o/maddpg_local/micro/
scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem $1 ubuntu@$line:~/IROS22_DARL1N/maddpg_o/experiments/
#scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem $1 ubuntu@$line:~/IROS22_DARL1N/train_simple_spread/
echo "Transfer $1 to $line Done !"
done < $filename
