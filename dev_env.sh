docker build -t potholes .

docker run -it --rm -v $PWD:/home/jovyan/work -p 8888:8888 potholes
