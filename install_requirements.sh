which pip
source /root/anaconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda init
source ~/.bashrc
conda activate env_perso
conda install -c conda-forge cudatoolkit cudnn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export CUDA_DIR=/usr/local/cuda-11.3
pip install -r requirements.txt
pip freeze
