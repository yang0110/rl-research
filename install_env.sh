# conda create -n rl_env python=3.10 -y
# conda activate rl_env
pip install minigrid
pip install dm-control
pip install torch==2.6.0
pip install "gymnasium[box2d]"
pip install "gymnasium[mujoco]"
pip install "gymnasium[atari]"
pip install "gymnasium[accept-rom-license]"
pip install ale-py
pip install shimmy 
pip install opencv-python
pip install matplotlib
pip install tensorboard
pip install pandas pyarrow torchvision networkx seaborn scikit-learn
pip install scikit-image
pip install termcolor
export MUJOCO_GL=egl