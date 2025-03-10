#################################################
Create a Python environment (Python version 3.11)

#################################################
Install these libraries in the environment cmd:
pip install fastapi
pip install uvicorn
pip install numpy
pip install pillow

#################################################
open a terminal:

uvicorn main:app --reload --port 8080

open a second terminal:

streamlit run .\streamlit_app.py 

