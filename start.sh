source activate backend
python backend/app.py &

source deactivate
source activate frontend
streamlit run frontend/app.py --server.port 30001 &