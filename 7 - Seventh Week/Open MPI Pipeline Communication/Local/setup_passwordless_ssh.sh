#!/bin/bash

# Check if both remote_username and remote_IP_address are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <remote_username> <remote_IP_address>"
    exit 1
fi

# Variables
remote_user="$1"
remote_ip="$2"

# Update package list and install required packages
sudo apt-get update
sudo apt-get install -y openssh-server

# Generate SSH key pair if it doesn't already exist
if [ ! -f id_rsa ]; then
    ssh-keygen -t rsa -b 4096 -N "" -f id_rsa
fi

# Copy the public key to the remote VM
ssh-copy-id -i id_rsa.pub $remote_user@$remote_ip

# Create or update the ~/.ssh/config file
config_file="${HOME}/.ssh/config"
touch $config_file
current_dir=$(pwd)
{
  echo "Host $remote_ip"
  echo "  User $remote_user"
  echo "  IdentityFile ${current_dir}/id_rsa"
} >> $config_file

# Test passwordless SSH connection
ssh $remote_ip 'echo "Passwordless SSH connection successful"'
