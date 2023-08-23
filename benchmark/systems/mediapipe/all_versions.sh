#!/bin/bash

# Array of versions
#versions=('0.7.10' '0.8.0' '0.8.1' '0.8.2' '0.8.3.1' '0.8.4.2' '0.8.5' '0.8.6.2' '0.8.7.1' '0.8.7.2' '0.8.7.3' '0.8.8' '0.8.8.1' '0.8.10.1' '0.8.11' '0.9.0' '0.9.0.1' '0.9.1.0' '0.9.2.1' '0.9.3.0' '0.8.9' '0.8.9.1' '0.8.10' '0.10.0' '0.10.1' '0.10.2' '0.10.3')
#versions=('0.7.10' '0.8.11' '0.9.3.0' '0.10.3')
versions=('0.10.3')


for version in "${versions[@]}"
do
  echo "Processing version: $version"
  pip uninstall -y mediapipe

  pip install mediapipe==$version

  python main.py
done
