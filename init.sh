apt-get update
apt-get -y install libgl1-mesa-glx
apt-get install g++

# Mongo DB
apt-get install mongodb

# Backend
cd backend/
bash set_env.sh

# Frontend
cd ..
cd frontend/
bash set_env.sh

# Cartoonize
cd ..
cd models/cartoonize/
bash set_env.sh

# Tracker
cd ../../
cd models/track/
bash set_env.sh
