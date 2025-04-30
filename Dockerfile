FROM python:3.10-slim
WORKDIR /app

# 1. install Python libs
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. copy code + notebook + model artefacts
COPY streamlit_app.py .
COPY notebooks/ notebooks/
COPY mlruns/ mlruns/

# 3. open both ports
EXPOSE 8501 8888

# 4. default locations & simple Jupyter password
ENV MLFLOW_TRACKING_URI="file:///app/mlruns" \
    JUPYTER_TOKEN="letmein" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 5. start Streamlit AND Jupyter
CMD bash -c "\
  streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 & \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=$JUPYTER_TOKEN \
"
