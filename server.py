import flwr as fl
from servers.v2_letsfed_fl_server import V2LetsServer

import yaml

if __name__ == '__main__':
    # --- Initialize FL Server ---
    with open('environment.yaml') as file:
        env = yaml.load(file, Loader=yaml.FullLoader)

    fl.server.start_server(
        server_address = "[::]:8888",
        strategy = V2LetsServer(env.get('N_CLIENTS')),
        config=fl.server.ServerConfig(num_rounds = env.get('N_ROUNDS')),
    )    