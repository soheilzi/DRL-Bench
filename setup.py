# setup.py
from distutils.core import setup

setup(
    name='drl',
    version='1.0',
    description='A robust and flexible deep reinforcement learning framework using jax',
    author='Soheil Zibakhsh Shabgahi',
    packages=['drl'],
    package_dir={'drl': 'drl'},
    package_data={'drl': ['*.py']},
    requires=['jax', 'jaxlib', 'numpy', 'gym', 'matplotlib', 'tqdm', 'optax', 'flax', 'dm_env', 'dm_control', 'dm_tree', 'dm_env_rpc', 'dm_env_rpc_viz', 'dm_env_rpc_viz_client', 'dm_env_rpc_vi', 'mujoco', 'tensorboard', 'torch']
)