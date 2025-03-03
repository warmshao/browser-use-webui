docker build -t agent .
docker run -p 7788:7788 -p 6080:6080 -p 5901:5901 agent