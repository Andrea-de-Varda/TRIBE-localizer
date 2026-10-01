#!/bin/bash
# One-time: create the `tribe` conda env on Engaging (run inside an interactive compute allocation).
set -euo pipefail
source /etc/profile.d/modules.sh
module purge; module load deprecated-modules; module load anaconda3/2022.05-x86_64
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda create -y -n tribe python=3.11
conda activate tribe
pip install "tribev2 @ git+https://github.com/facebookresearch/tribev2.git@af58661791a351a448a489042a28f6c37e1c14b7"
pip install -e "/orcd/data/evelina9/001/USERS/devar_ag/TRIBE-localizer[test]"
python -c "import tribev2, tribeloc; print('ok')"
