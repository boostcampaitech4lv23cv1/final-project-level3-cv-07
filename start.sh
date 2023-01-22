source activate frontend
streamlit run frontend/app.py --server.port 30001 --server.fileWatcherType none &

source deactivate
source activate backend
python backend/app.py