#!/bin/ksh
# pass the ssh public key of host to ec2 instances
i=1
filename='nodeIPaddress'
while IFS= read -r line
do
echo $line
echo "Execution Done!"
ssh -i  ~/AmazonEC2/.ssh/linux_key_pari.pem -n ubuntu@$line 'killall python3'
((i=i+1))
done < $filename
