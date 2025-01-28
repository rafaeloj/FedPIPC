import random as rd
import typing as T
from math import ceil
import numpy as np

def random(num_clients: int, perc: float) -> T.List[str]:
    list_of_clients = list(range(num_clients))
    perc = int(len(list_of_clients)*perc)
    if perc <= 1:
        perc = 2
    selected_clients = rd.sample(list_of_clients, perc)
    return [str(cid) for cid in selected_clients]

def poc(losses: T.List[T.Tuple[int, float]], perc: float) -> T.List[str]:
    lc = losses.copy()
    lc.sort(key=lambda x: x[1], reverse=True)
    # print(lc)
    selected_clients = []
    for cid, _ in lc:
        selected_clients.append(cid)

    clients2select        = int(float(len(losses)) * float(perc))
    if clients2select <= 1:
        clients2select = 2
    selected_clients  = [
        str(cid) for cid, _ in lc[:clients2select]
    ]

    # print(selected_clients)
    return selected_clients

def deev(rnd: int, losses: T.List[T.Tuple[int, float]], avg_loss: float, decay: float = 0.05) -> T.List[str]:
    lc = losses.copy()
    # lc = [(cid, loss) for cid, loss in losses]
    lc.sort(key=lambda x: x[1], reverse=True)
    selected_clients = []
    for cid, loss in lc:
        if loss > avg_loss:
            selected_clients.append(cid)

    if decay > 0.0:
        the_chosen_ones  = len(selected_clients) * (1 - decay)**int(rnd)
        selected_clients = selected_clients[ : ceil(the_chosen_ones)]
    return [str(cid) for cid in selected_clients]

    # return selected_clients

def r_robin(num_clients: int, how_many_time_selected: T.List[int], perc: float) -> T.Tuple[T.List[str], T.List[int]]:
    list_of_clients = list(range(num_clients))
    clients_cid_int = [int(cid) for cid in list_of_clients]

    # Pega a quantidade de clientes que querem participar dentro do contador
    how_many_time_selected_client = how_many_time_selected[clients_cid_int]

    # Aqui basicamente eu to pegando um array de indices ordenados pelos valorres do how_many_time_selected_client
    sort_cids = np.argsort(how_many_time_selected_client)
    
    # Pegando o top menos chamados
    top_values_of_cid  = sort_cids[:int(len(how_many_time_selected_client)*perc)]

    # Update score
    for cid_value_index in top_values_of_cid:
        how_many_time_selected[
            cid_value_index
        ] += 1
    
    # To pegando indice dos clientes que foram selecionados
    top_clients = [str(cid) for cid in top_values_of_cid]
    return top_clients, how_many_time_selected

def random_from(clients: T.List[int], perc: float) -> T.List[str]:
    perc = int(len(clients)*perc)
    if perc <= 1 and len(clients) > 1:
        perc = 2
    selected_clients = rd.sample(clients, perc)
    return [str(cid) for cid in selected_clients]

def r_robin_from(clients: T.List[int], how_many_time_selected: T.List[int], perc: float) -> T.Tuple[T.List[str], T.List[int]]:
    # Pega a quantidade de clientes que querem participar dentro do contador
    how_many_time_selected_client = how_many_time_selected[clients]

    # Aqui basicamente eu to pegando um array de indices ordenados pelos valorres do how_many_time_selected_client
    sort_cids = np.argsort(how_many_time_selected_client)
    
    # Pegando o top menos chamados
    top_values_of_cid  = sort_cids[:int(len(how_many_time_selected_client)*perc)]

    # Update score
    for cid_value_index in top_values_of_cid:
        how_many_time_selected[
            cid_value_index
        ] += 1
    
    # To pegando indice dos clientes que foram selecionados
    top_clients = [str(cid) for cid in top_values_of_cid]
    return top_clients, how_many_time_selected

