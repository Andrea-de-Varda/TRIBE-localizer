# Sourced by the sbatch scripts. Environment and paths for Engaging.
source /etc/profile.d/modules.sh
module purge
module load deprecated-modules
module load anaconda3/2022.05-x86_64
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate tribe

# Hugging Face cache. Llama-3.2-3B is gated: log in once,
# with this HF_HOME exported, `hf auth login` (or export HF_TOKEN in ~/.bashrc). The token is never stored in this repository.
export HF_HOME=/orcd/data/evelina9/001/USERS/devar_ag/.hf_cache_new
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

PROJECT_DIR=/orcd/data/evelina9/001/USERS/devar_ag/TRIBE-localizer
cd "$PROJECT_DIR"
mkdir -p logs
