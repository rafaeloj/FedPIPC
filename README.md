# FedPIPC: Como Agregar e Não Influenciar Modelos - Controlando a Influência de Clientes no Aprendizado Federado Dinâmico

**Resumo:** Sistemas distribuídos têm se mostrado uma abordagem essencial para o aprendizado de máquina, especialmente em cenários com grande quantidade de dispositivos conectados, como a Internet das Coisas (IoT) e Cidades Inteligentes. No entanto, a disponibilidade desses dispositivos é crucial para um treinamento eficaz, e não é garantida durante todo o treinamento devido a limitações como bateria, largura de banda ou requisitos de conformidade. Para enfrentar esses desafios, propomos {\name}, um método de agregação que utiliza a participação dos clientes como fatores de influência no modelo global, combinado com um mecanismo de preservação de conhecimento para mitigar os impactos das oscilações na participação ao longo do treinamento federado. Os resultados empíricos indicam que a solução proposta melhora o equilíbrio entre precisão e transmissão de dados em até 50\% e reduz o volume de transmissão em até 89\%, comprovando sua eficácia em cenários dinâmicos.


## 🔧 Configuração de Hiperparâmetros

Esta seção descreve os hiperparâmetros configuráveis utilizados no ambiente de **Aprendizado Federado**. Ajuste esses parâmetros no arquivo de configuração para personalizar o processo de treinamento.

## 📊 Configurações Gerais

- **`N_CLIENTS`** (`int`, padrão: `10`)  
  Define o número total de clientes participantes no treinamento federado.

- **`MODEL_TYPE`** (`str`, padrão: `'dnn'`)  
  Especifica o tipo de modelo a ser utilizado no treinamento. Opções:  
  - `'dnn'`: Rede Neural Profunda (Deep Neural Network)  
  - `'cnn'`: Rede Neural Convolucional (Convolutional Neural Network)  

- **`N_ROUNDS`** (`int`, padrão: `10`)  
  Define o número de rodadas de comunicação entre o servidor e os clientes durante o treinamento.

## 📚 Configuração de Dataset

- **`DATASET`** (`str`, padrão: `'fashion_mnist'`)  
  Define o conjunto de dados utilizado para o treinamento. Opções disponíveis estão documentadas no framework [Flower Datasets](https://flower.ai/docs/datasets/index.html):

  **Atenção:** Ao trocar o dataset, lembre-se de ajustar a distribuição de Dirichlet.

## 🤝 Estratégia de Seleção de Clientes

- **`SEL_INVITATION`** (`str`, padrão: `'random'`)  
  Define a estratégia para seleção de clientes. `'random'` indica uma seleção de Randômica.

## ⚙️ Métodos de Treinamento e Agregação

- **`FIT_METHOD`** (`str`, padrão: `'avg'`)  
  Especifica o método utilizado para o ajuste dos modelos locais.  
  - `'avg'`: Média simples das atualizações locais.
  - `'prox'`: Média simples das atualizações locais.

- **`AGG_METHOD`** (`str`, padrão: `'yogi'`)  
  Define o algoritmo de agregação usado para combinar as atualizações dos clientes.  
  - `'yogi'`
  - `'avg'`
  - `'avgm'`
  - `'FedPIPC'`

---

## 🚀 Exemplo de Configuração

```python
N_CLIENTS: 50
MODEL_TYPE: 'dnn' # CNN or DNN
N_ROUNDS: 100
DATASET: 'fashion_mnist' ## QUANDO MUDAR DE DATASET LEMBRA DE MUDAR O DIRICHLET
SEL_INVITATION: 'poc' 
FIT_METHOD: 'avg'
AGG_METHOD: 'yogi'
AWARNESSE: True
```
