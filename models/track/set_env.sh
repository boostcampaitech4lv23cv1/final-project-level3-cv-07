conda create -n track python=3.7 -y
source activate track
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
python setup.py develop