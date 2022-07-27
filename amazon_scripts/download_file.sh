#!/bin/sh
# Download 12agents_1.zip from Amazon EC2 instance with IP address 18.116.203.251 to local directory ~/aaai_darl1n/result/grassland/darl1n/
scp -i ~/AmazonEC2/.ssh/linux_key_pari.pem  ubuntu@13.58.36.60:/home/ubuntu/IROS22_DARL1N/result/simple_spread/darl1n.zip  ~/iros22_darl1n/result/simple_spread/
