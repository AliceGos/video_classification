source /root/anaconda3/etc/profile.d/conda.sh
conda init
conda update -n base -c defaults conda
conda create -n env_perso python=3.7
conda activate env_perso
conda update -n base -c defaults conda
source install_requirements.sh
echo "PYTHONPATH=${PYTHONPATH}:${SCRIPTS_DIR}" > .env
echo "export PATH=/root/anaconda3/envs/aecluster/bin:$PATH" >> ~/.bashrc
