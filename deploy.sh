#!/bin/bash

docker build -t mlops-api .

docker run -d -p 8000:8000 mlops-api
