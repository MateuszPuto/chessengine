#!/bin/bash
#Use as standalone script in new folder

USR=$(whoami)

mkdir chessengine
cd chessengine
git clone https://github.com/MateuszPuto/chessengine.git .

cd ..
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b -p $HOME/miniconda
export PATH=$PATH:/home/$USR/miniconda/bin

conda init bash
conda activate env
conda install python=3.7.9 -y

cd chessengine
sudo dnf install python3-devel -y
pip3 install -r requirements.txt
sudo dnf install httpd-devel -y
sudo systemctl start httpd.service 
pip3 install mod_wsgi

mod_wsgi-express start-server wsgi.py

