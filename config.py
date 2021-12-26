import sys

# Communication config
COMM = 'TCP'

# Network configration
SERVER_ADDR = '192.168.101.120'
SERVER_PORT = 1883

K = 4  # Number of devices
G = 4  # Number of groups

# Unique clients order
HOST2IP = {'mars116XU': '192.168.101.116', 'mars117XU': '192.168.101.217', 'mars118XU': '192.168.101.218',
           'mars119XU': '192.168.101.219'}
CLIENTS_CONFIG = {'192.168.101.116': 0, '192.168.101.217': 1, '192.168.101.218': 2, '192.168.101.219': 3}
CLIENTS_LIST = ['192.168.101.116', '192.168.101.217', '192.168.101.218', '192.168.101.219']

# Dataset configration
dataset_name = 'CIFAR10'
home = sys.path[0].split('FedAdapt')[0] + 'FedAdapt'
dataset_path = "../datasets/CIFAR10/"
N = 50000  # data length

# train communication time
comm_time = 0.0

# Model Configration
model_cfg = {
    # (Type, in_channels, out_channels, kernel_size, out_size(c_out*h*w), flops(c_out*h*w*k*k*c_in))
    'VGG8': [('C', 3, 32, 3, 32 * 32 * 32, 32 * 32 * 32 * 3 * 3 * 3),
             ('C', 32, 32, 3, 32 * 32 * 32, 32 * 32 * 32 * 3 * 3 * 32),
             ('M', 32, 32, 2, 32 * 16 * 16, 0),
             ('C', 32, 64, 3, 64 * 16 * 16, 64 * 16 * 16 * 3 * 3 * 32),
             ('C', 64, 64, 3, 64 * 16 * 16, 64 * 16 * 16 * 3 * 3 * 64),
             ('M', 64, 64, 2, 64 * 8 * 8, 0),
             ('C', 64, 128, 3, 128 * 8 * 8, 128 * 8 * 8 * 3 * 3 * 64),
             ('C', 128, 128, 3, 128 * 8 * 8, 128 * 8 * 8 * 3 * 3 * 128),
             ('D', 8 * 8 * 128, 128, 1, 128, 128 * 8 * 8 * 128),
             ('D', 128, 10, 1, 10, 128 * 10)]
}
model_name = 'VGG8'
model_size = 5.35
model_flops = 158.73
total_flops = 39683328

split_layer = [9, 9, 9, 9]
model_len = 10

# FL training configration
R = 5  # FL rounds
LR = 0.01  # Learning rate
B = 100  # Batch size

# RL training configration
max_episodes = 100  # max training episodes
max_timesteps = 100  # max timesteps in one episode
exploration_times = 20  # exploration times without std decay
n_latent_var = 64  # number of variables in hidden layer
action_std = 0.5  # constant std for action distribution (Multivariate Normal)
update_timestep = 10  # update policy every n timesteps
K_epochs = 50  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
rl_gamma = 0.9  # discount factor
rl_b = 100  # Batchsize
rl_lr = 0.0003  # parameters for Adam optimizer
rl_betas = (0.9, 0.999)
iteration = {'192.168.0.14': 5, '192.168.0.15': 5, '192.168.0.25': 50, '192.168.0.36': 5,
             '192.168.0.29': 5}  # infer times for each device

random = True
random_seed = 0
