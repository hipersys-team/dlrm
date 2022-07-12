# DLRM for Personalization and Recommandation systems

This submodule contains the DLRM model used to generate TopoOpt's testbed DLRM result. It was adopted from Meta's implementaiton of DLRM with model parallelism. Please check the original README file at [here](DLRM_README.md) for a more detailed description. 

## Instructions
The scripts are designed to run distributed model parallel training of DLRM on MIT's 12 node cluster testbed. Please check [this document](https://docs.google.com/document/d/190nelkTXo7fEQNWRe4rnMglzAvV1jj-ZyShMcAGZH08/edit?usp=sharing) for how to setup RDMA forwarding and using the hacked version of NCCL. 

The scripts uses the [pytorch version](dlrm_s_pytorch.py) of this repository. 

`run_a100_fattree.sh` will run DLRM training on the Mellanox ConnectX5 NIC, which are connected to a single switch. 

`run_a100_topoopt.sh` will run DLRM training on the HPE nics, which are connected to the patch panel. Please be sure the forwarding is setup properly before running this test. Check the document above on how to setup RDMA fowarding. 

To execute the program, adjust the parameters in the scripts and run the script on ALL of the workers. The script will automatically pick the master and log the training output on the master machine. 



