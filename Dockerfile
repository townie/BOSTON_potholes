FROM jupyter/datascience-notebook

# TODO fix this

# COPY requirements.txt requirements.txt
# RUN pip install -e requirements.txt

RUN pip install pandas-profiling
