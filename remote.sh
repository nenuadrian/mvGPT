#!/usr/bin/env bash
# Gets a rented machine with NVIDIA GPUs ready to train: the code at HEAD, the BPE merges learned here,
# the C++ tokenizer built, and all of TinyStories, tokenized.
#   ./remote.sh HOST    (an ssh host, from ~/.ssh/config)
set -euo pipefail
host=$1

git archive HEAD | ssh "$host" 'mkdir -p mvGPT/data && tar -x -C mvGPT'
# The same merges, so the rented machine's tokens are the ones the laptop's models were trained on.
scp -q data/merges.txt "$host:mvGPT/data/"
ssh "$host" 'set -e
cd mvGPT
command -v g++ > /dev/null || { apt-get -qq update && apt-get -qq install -y g++ > /dev/null; }
# The image has the system Python, which pip leaves alone unless told to: fine on a machine we throw away.
pip install -q --root-user-action=ignore --break-system-packages numpy tensorboard pybind11
python setup.py -q build_ext --inplace
python data.py --full'
