apt-get update
apt-get -y install libgl1-mesa-glx
apt-get install -y libsm6 libxext6 libxrender-dev
apt-get install g++ -y
apt-get install ffmpeg

# Mongo DB
apt-get install mongodb -y
service mongodb start

base_dir=$(pwd)

# Backend
cd $base_dir/backend/
bash set_env.sh

# Frontend
cd $base_dir/frontend/
bash set_env.sh

# Cartoonize
cd $base_dir/models/cartoonize/
bash set_env.sh

# Tracker
cd $base_dir/models/track/
bash set_env.sh