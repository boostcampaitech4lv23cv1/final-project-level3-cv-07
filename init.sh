apt-get update
apt-get -y install libgl1-mesa-glx
apt-get install -y libsm6 libxext6 libxrender-dev
apt-get install g++ -y

# Mongo DB
apt-get install mongodb -y
service mongodb start

pwd = pwd

# Backend
cd $pwd\backend/
bash set_env.sh

# Frontend
cd $pwd\frontend/
bash set_env.sh

# Cartoonize
cd $pwd\models/cartoonize/
bash set_env.sh

# Tracker
cd $pwd\models/track/
bash set_env.sh
