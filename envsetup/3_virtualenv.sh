echo 'Setting up new Python virtual environment in ~/tensorflow-py3ve'

#
# VirtualEnv Setup
#
sudo apt-get install -y python3-pip python3-dev python-virtualenv
virtualenv --system-site-packages -p python3 ~/tensorflow-py3ve
source ~/tensorflow-py3ve/bin/activate
easy_install -U pip
pip3 install --upgrade 'tensorflow-gpu==1.5'
deactivate