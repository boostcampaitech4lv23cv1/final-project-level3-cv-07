source deactivate
source activate backend
python backend/app.py &

source activate frontend
streamlit run frontend/app.py --server.port 30001 --server.fileWatcherType none &

sleep 2

source deactivate
source activate backend
python backend/app.py &

source deactivate
source activate cartoonize
python models/cartoonize/app.py &

source deactivate
source activate track
python models/track/app.py