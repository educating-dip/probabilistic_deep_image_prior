#!/bin/bash

if [[ $- != *i* ]]
then
    echo "Please run in interactive mode, i.e. bash -i ...; aborting."
    exit 1
fi

print_usage () {
    echo "usage:"
    echo -e "\tbash -i $0 -n environment_name [--cudatoolkit cudatoolkit_version]"
    echo -e "\tbash -i $0 -p path_to_env [--cudatoolkit cudatoolkit_version]"
}

if [[ $# -lt 2 ]]; then
    print_usage
    exit 1
fi
if [[ "$1" != "-n" ]] && [[ "$1" != "-p" ]]; then
    print_usage
    exit 1
fi
if [[ $# -ge 3 ]] && [[ "$3" != "--cudatoolkit" ]]; then
    print_usage
    exit 1
fi

flag=$1
env_path=$2
cudatoolkit_version=${4-10.2}

# exit when any command fails
set -e

# create and activate conda env
conda create $flag $env_path
conda activate $env_path

# install torch as suggested on https://pytorch.org/get-started/locally/
conda install pytorch torchvision cudatoolkit=$cudatoolkit_version -c pytorch-nightly

# install other conda packages
conda install tensorboard tensorboardx scikit-image imageio opt_einsum tqdm

# install astra package (should install version >= 2.0.0)
conda install astra-toolbox -c astra-toolbox

# install pip packages
pip install hydra-core tensorly https://github.com/odlgroup/odl/archive/master.zip bios

echo -e "created env ${env_path}; to activate it, use:\n\tconda activate ${env_path}"
