#!/bin/bash

# Install Open MPI packages
sudo apt-get update
sudo apt-get install -y openmpi-bin libopenmpi-dev

# Create the openmpi_ssh file
sudo bash -c 'cat > /usr/local/bin/openmpi_ssh << EOL
#!/bin/sh
echo "Running openmpi_ssh with arguments: \$@"
/usr/bin/ssh -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null "\$@"
EOL'

# Make the openmpi_ssh file executable
sudo chmod +x /usr/local/bin/openmpi_ssh

# Set the OMPI_MCA_plm_rsh_agent environment variable
echo "export OMPI_MCA_plm_rsh_agent=/usr/local/bin/openmpi_ssh" >> ~/.bashrc
source ~/.bashrc
