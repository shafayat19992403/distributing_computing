#!/bin/bash
sudo yum update -y
sudo yum install -y https://s3.amazonaws.com/ec2-downloads-windows/SSMAgent/latest/linux_amd64/amazon-ssm-agent.rpm
sudo yum install -y amazon-ssm-agent
sudo systemctl start amazon-ssm-agent
cd /home/ec2-user
sudo yum install -y git
git clone https://github.com/Hasebul/distributed_computing.git
cd distributed_computing/
git checkout Data_Poison
sudo yum install python3 python3-pip -y
pip install -r Client/requirements.txt

python3 Client/client.py --server_address --poison_rate --threshold_loss --threshold_accuracy 