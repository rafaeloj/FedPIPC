import flwr as fl
from flwr.common import FitRes, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_proxy import ClientProxy
import numpy as np
from functools import reduce
import typing as T
import pandas as pd
from .selections import random_from as random, poc, deev, r_robin_from as r_robin
import os
import pickle
import yaml
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class V2LetsServer(fl.server.strategy.FedAvg):
    def __init__(self, n_clients: int = 2):
        self.losses: T.List[T.Tuple[int, float]] = []
        self.likelihood_ratios: T.List[T.Tuple[int, float]] = []
        self.n_clients = n_clients
        self.how_many_time_selected = np.zeros(n_clients)
        self.how_many_time_non_participating = np.zeros(n_clients)
        self.how_many_time_participating = np.zeros(n_clients)
        self.last_parameters = None
        self.participant_set: T.Set[int] = set()
        self.non_participant_set: T.Set[int] = set()
        self.participant_weights: T.List[T.Tuple[int, fl.common.NDArrays]] = [(i, None) for i in range(n_clients)]
        self.client_weights = [None for _ in range(n_clients)]
        self.client_selections = pickle.load(open('selected_cids.pkl', 'rb'))
        
        
        # --- FedAVGM
        self.server_momentum = 0.8
        self.server_learning_rate = 0.3
        # --- FedAVGM
        
        # --- FEdYogi
        self.eta: float = 1e-2
        self.eta_l: float = 0.0316
        self.beta_1: float = 0.9
        self.beta_2: float = 0.99
        self.tau: float = 1e-3
        self.m_t = None
        self.v_t = None
        # --- FEdYogi
        
        super().__init__(
            fraction_fit = 1,
            min_available_clients = n_clients,
            min_fit_clients = n_clients,
            min_evaluate_clients = n_clients
        )

    def selection_method(self, server_round: int, method: str, losses: T.List[T.Tuple[int, float]], clients: T.Set[int], perc=0.2, decay=0.005) -> T.List[str]:
        if method == 'random':
            selected_clients = random(list(clients), perc)
        if method == 'poc':
            selected_clients = poc(losses, perc)
        if method == 'deev':
            selected_clients = deev(server_round, losses, self.avg_loss, decay=decay)
        if method == 'r_robin':
            selected_clients, self.how_many_time_selected = r_robin(list(clients), self.how_many_time_selected, perc)
        return selected_clients

    def client_selection(self, server_round: int, sel_invitation: str, sel_exploration: str, awarnesse: bool = True) -> T.Tuple[T.List[str], T.List[T.Tuple[str, T.Tuple[int, fl.common.NDArrays]]]]:
        if server_round == 1:
            self.participant_set = set(range(self.n_clients))
            return [str(cid) for cid in range(self.n_clients)], [] # A U B 
        
        client = self.participant_set | self.non_participant_set
        return self.selection_method(server_round, sel_invitation, self.losses, client), []
        
    def configure_fit(self, server_round: int, parameters, client_manager):
        self.rnd = server_round
        # --- Initialize FL Server ---
        with open('environment.yaml') as file:
            env = yaml.load(file, Loader=yaml.FullLoader)

        SEL_INVITATION = env.get('SEL_INVITATION')
        # SEL_EXPLORATION = env.get('SEL_EXPLORATION')
        FIT_METHOD = env.get('FIT_METHOD')
        self.AGG_METHOD = env.get('AGG_METHOD')
        AWARNESSE = env.get('AWARNESSE')
        
        self.sel = f'{FIT_METHOD}+{SEL_INVITATION}+{self.AGG_METHOD}'
        invitation, exploration = [str(cid) for cid in self.client_selections[server_round]], []
        
        self.selected_clients = invitation + [cid for cid, _ in exploration] # Garante que os clientes não participantes sejam selecionados corretamente

        print(f"Selected clients: {self.selected_clients}") ## Importante para acompanhar o que está acontecendo
        
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if server_round == 1:
            self.flwr_cid_to_nid = {
                client.cid: cid
                for cid, client in enumerate(clients)
            }
            self.last_parameters = parameters_to_ndarrays(parameters)

        config = {
            'rnd': server_round,
            'sel': self.sel,
        }

        config['selected_by_server'] = ','.join(self.selected_clients)

        p_fit_ins = fl.common.FitIns(parameters, config)
        np_fit_ins = {
            selected_cid: fl.common.FitIns(ndarrays_to_parameters(selection_model[1]), config)
            for selected_cid, selection_model in exploration
        } if AWARNESSE else p_fit_ins # Garante que o modelo dos melhores clientes participantes sejam enviados para os clientes não participantes.
        response = []
        if server_round == 1:
            for cid in range(self.n_clients):
                self.client_weights[cid] = parameters_to_ndarrays(parameters)
        # Coloca os modelos para os clientes certos.
        for idx_cid in range(len(clients)):
            cid = self.flwr_cid_to_nid[clients[idx_cid].cid]
            if str(cid) in self.selected_clients and cid in self.non_participant_set:
                try:
                    response.append((clients[idx_cid], np_fit_ins[cid]))
                except:
                    response.append((clients[idx_cid], np_fit_ins))
            else:
                response.append((clients[idx_cid], p_fit_ins))
        return response

    def aggregate(self, results):
        return aggregate(results)

    def new_aggregate(self, results, server_round: int, similarity_threshold=0.8):
        total_participations = sum(self.how_many_time_participating)
        participation_weights = [
            (cp / total_participations) for cp in self.how_many_time_participating
        ]

        aggregated_weights = [
            np.zeros_like(layer) for layer in results[0][0]
        ]

        total_examples = sum(num_examples for _, num_examples in results)

        for (parameters, num_examples), participation_weight in zip(results, participation_weights):
            for i, layer in enumerate(parameters):
                aggregated_weights[i] += layer * (num_examples / total_examples) * (1-participation_weight)

        # Mecanismo de memória para estabilidade
        if self.last_parameters is not None:
            delta_w = [
                new_l - old_y
                for new_l, old_y in zip(aggregated_weights, self.last_parameters)
            ]

            mean_delta = np.mean([np.linalg.norm(delta) for delta in delta_w]) + 1e-8
            std_delta = np.std([np.linalg.norm(delta) for delta in delta_w]) + 1e-8

            lambda_smooth = max(0.1, min(0.9, 1.0 / (1.0 + std_delta / mean_delta)))

            # 0.3 - cifar10
            # Aplicar suavização para atualizar os pesos
            aggregated_weights = [
                old_y + lambda_smooth * dw
                for old_y, dw in zip(self.last_parameters, delta_w)
            ]

        else:
            # Inicializar parâmetros se for a primeira rodada
            self.last_parameters = aggregated_weights.copy()

        self.last_parameters = aggregated_weights.copy()

        return aggregated_weights

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]):
        self.participant_weights = [
            (fit_resp.metrics['cid'], parameters_to_ndarrays(fit_resp.parameters)) for _, fit_resp in results if fit_resp.metrics['participate']
        ] # Atualizações dos participantes

        self.non_participant_weights = [
            (fit_resp.metrics['cid'], parameters_to_ndarrays(fit_resp.parameters)) for _, fit_resp in results if not fit_resp.metrics['participate']
        ] # Atualizações dos não participantes

        selected_clients = [ fit_resp for _, fit_resp in results if str(fit_resp.metrics['cid']) in self.selected_clients]

        participant_resp = [
            fit_resp for fit_resp in selected_clients if fit_resp.metrics['participate']
        ]

        non_participant_resp = [
            fit_resp for fit_resp in selected_clients if not fit_resp.metrics['participate']
        ]

        cid_p = {fit_resp.metrics['cid'] for fit_resp in participant_resp}
        cid_np = {fit_resp.metrics['cid'] for fit_resp in non_participant_resp}
        
        self.cid_p = cid_p
        self.cid_np = cid_np
        self._vira_casaca = self.non_participant_set & cid_p
        self.participant_set = (self.participant_set | cid_p) - cid_np # (P + SP) - SNP
        self._last_np = self.non_participant_set.copy()
        self.non_participant_set = (self.non_participant_set | cid_np) - cid_p # (NP + SNP) - SP

        for fit_resp in non_participant_resp:
            cid = fit_resp.metrics['cid']
            self.how_many_time_non_participating[int(cid)] += 1

        for fit_resp in participant_resp:
            cid = fit_resp.metrics['cid']
            self.how_many_time_participating[int(cid)] += 1
        
        parameters = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for fit_res in participant_resp
        ]

        if len(parameters) < 1: ## Não tinha clientes interessados suficiente para participação então não atualiza os pesos e escolhe outros clientes.
            return ndarrays_to_parameters(self.last_parameters), {}
        
        if 'avg' == self.AGG_METHOD:
            self.last_parameters = self.aggregate(parameters)
        elif 'maxfl' == self.AGG_METHOD:
            self.maxfl_aggregate_fit(server_round, participant_resp)
        elif 'avgm' == self.AGG_METHOD:
            self.fedavgm_aggregate_fit(server_round, parameters)
        elif 'yogi' == self.AGG_METHOD:
            self.fed_yogi(self.aggregate(parameters))
        else:
            self.new_aggregate(parameters, server_round)

        return ndarrays_to_parameters(self.last_parameters), {} 

    def maxfl_aggregate_fit(self, server_round: int, participant_resp):
        results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics['maxfl_w'])
            for fit_res in participant_resp
        ]
        maxfl_weights_sum = np.sum([maxfl_w for _, maxfl_w in results])
        
        aggregated_weights = [
            np.zeros_like(layer) for layer in results[0][0]
        ]
        for parameters, _ in results:
            for i, layer in enumerate(parameters):
                aggregated_weights[i] += layer

        learning_rate = 0.09 / (maxfl_weights_sum+1e-8)

        aggregated_weights = [
            old_y + learning_rate * dw
            for old_y, dw in zip(self.last_parameters, aggregated_weights)
        ]

        self.last_parameters = aggregated_weights.copy()

        return aggregated_weights 
        
    def fedavgm_aggregate_fit(self, server_round: int, participant_resp):
        """
            Retirado do framework: https://flower.ai/docs/baselines/fedavgm.html
        """
        fedavg_result = aggregate(participant_resp)
        pseudo_gradient = [
            x - y
            for x, y in zip(
                self.last_parameters, fedavg_result
            )
        ]

        if server_round > 1:
            assert self.momentum_vector, "Momentum should have been created on round 1."

            self.momentum_vector = [
                self.server_momentum * v + w
                for w, v in zip(pseudo_gradient, self.momentum_vector)
            ]
        else:  # Round 1
            # Initialize server-side model
            assert (
                self.last_parameters is not None
            ), "When using server-side optimization, model needs to be initialized."
            # Initialize momentum vector
            self.momentum_vector = pseudo_gradient

        # Applying Nesterov
        pseudo_gradient = [
            g + self.server_momentum * v
            for g, v in zip(pseudo_gradient, self.momentum_vector)
        ]

        # Federated Averaging with Server Momentum
        fedavgm_result = [
            w - self.server_learning_rate * v
            for w, v in zip(
                self.last_parameters, pseudo_gradient
            )
        ]

        self.last_parameters = fedavgm_result

    def fed_yogi(self,
        fedavg_agg,
    ):
        """
            Retirado do framework: https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedYogi.html
        """
        delta_t = [
            x - y for x, y in zip(fedavg_agg, self.last_parameters)
        ]

        # m_t
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
            for x, y in zip(self.m_t, delta_t)
        ]

        # v_t
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            x - (1.0 - self.beta_2) * np.multiply(y, y) * np.sign(x - np.multiply(y, y))
            for x, y in zip(self.v_t, delta_t)
        ]

        new_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.last_parameters, self.m_t, self.v_t)
        ]

        self.last_parameters = new_weights

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        config = {
            'rnd': server_round,
            'selected_by_server': ','.join([str(cid) for cid in list(range(self.n_clients))])
        }

        evaluate_ins = fl.common.EvaluateIns(parameters, config)
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        return [ (client, evaluate_ins) for client in clients ]

    def aggregate_evaluate(self, server_round: int, results, failures):
        parameters_loss = []
        self.losses.clear()
        for _, eval_res in results:
            cid = eval_res.metrics['cid']
            val_loss = eval_res.metrics['val_loss']
            self.losses.append((cid, val_loss)) # Loss de validação
            parameters_loss.append((
                eval_res.num_examples,
                eval_res.loss, # Loss de teste
            ))
        
        losses = [loss for _, loss in self.losses]

        self.avg_loss = sum(losses)/(len(losses)+1e-6)

        weighted_losses = [
            (eval_res.num_examples, eval_res.loss) for _, eval_res in results
        ]
        
        self.last_evaluate_loss = fl.server.strategy.aggregate.weighted_loss_avg(weighted_losses)

        weighted_accuracies = [
            (eval_res.num_examples, eval_res.metrics['eval_accuracy']) for _, eval_res in results
        ]
        self.avg_acc = weighted_accuracies
        total_examples = sum(num_examples for num_examples, _ in weighted_accuracies)
        weighted_accuracy = sum(num_examples * accuracy for num_examples, accuracy in weighted_accuracies) / total_examples

        self.last_evaluate_accuracy = weighted_accuracy
        self.save_server_info(server_round)
        print(f'Round {server_round} Evaluate loss: {self.last_evaluate_loss} Evaluate accuracy: {weighted_accuracy}, MODEL SIZE: {sum([layer.nbytes for layer in self.last_parameters])}')
        return self.last_evaluate_loss, {"accuracy": weighted_accuracy}

    def save_server_info(self, server_rnd ):
        performance = {
            'rnd': [server_rnd],
            'cid_p': [self.cid_p],
            "cid_np": [self.cid_np],
            'selected_clients': [self.selected_clients],
            'self.participant_set': [self.participant_set],
            'self.non_participant_set': [self.non_participant_set],
            'sel': [self.sel],
            'how_many_time_non_participating': f"{' '.join([str(i) for i in self.how_many_time_non_participating])}",
            'evaluate_loss': [self.last_evaluate_loss],
            'evaluate_accuracy': [self.last_evaluate_accuracy],
            # 'model_size': [sum([layer.nbytes for layer in self.last_parameters])],
        }

        if os.path.exists(f'data/performance_server.csv'):
            df = pd.read_csv(f'data/performance_server.csv')
            df = pd.concat([df, pd.DataFrame(performance)], ignore_index=True)
        else:
            df = pd.DataFrame(performance)

        df.to_csv(f'data/performance_server.csv', index=False)