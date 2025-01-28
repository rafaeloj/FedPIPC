import argparse
import flwr as fl
from trainers.v2_letsfed_fl_client import V2LetsFLClient

import yaml

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()

    argparse.add_argument('--cid', type=int, required=True)

    # import environment.yaml
    with open('environment.yaml') as file:
        env = yaml.load(file, Loader=yaml.FullLoader)

    args = argparse.parse_args()
    client = V2LetsFLClient(cid=args.cid, model_type=env.get("MODEL_TYPE"), dataset_name=env.get("DATASET"), n_partitions=env.get('N_CLIENTS'))
    fl.client.start_client(
        server_address="localhost:8888",
        client=client
    )