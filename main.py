import gym
import numpy as np
import pickle
from colorama import init, Fore

init(autoreset=True)

env = gym.make("ALE/Riverraid-v5", render_mode="human", obs_type="rgb")
env.seed(42)

print("Ambiente:", env.observation_space)

alpha = 0.5  # Taxa de aprendizado
gamma = 0.9  # Fator de desconto
k = 100  # Parâmetro para a função f(u, n)

num_states = 10000  # Número de estados discretos
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))
N = np.zeros((num_states, num_actions))

estado_anterior = None
total_iteracoes = 0

def funcao_f(u, n):
    return k / (k + u)

def discretizar_estado(estado):
    # Hashing do estado para um valor inteiro
    return hash(estado.tostring()) % num_states

# Função para salvar os dados de treinamento
def salvar_dados(filename):
    with open(filename, 'wb') as f:
        dados = {
            'Q': Q,
            'N': N,
            'total_iteracoes': total_iteracoes
        }
        pickle.dump(dados, f)

# Carrega os dados de treinamento
def carregar_dados(filename):
    with open(filename, 'rb') as f:
        dados = pickle.load(f)
        Q = dados['Q']
        N = dados['N']
        total_iteracoes = dados['total_iteracoes']
        return Q, N, total_iteracoes

# Ações aleatórias antes de começar o episódio
for _ in range(100):
    env.reset()
    ambiente, _, _, _, _ = env.step(env.action_space.sample())

while True:
    if estado_anterior is not None:
        estado_discreto = discretizar_estado(estado_anterior)
        u = N[estado_discreto].sum()
        probabilidade_exploracao = funcao_f(u, total_iteracoes)
        if np.random.rand() < probabilidade_exploracao:
            acao = env.action_space.sample()
        else:
            acao = np.argmax(Q[estado_discreto])
    else:
        acao = env.action_space.sample()

    ambiente, recompensa, finalizado, paralizado, info = env.step(acao)
    recompensa_txt = "Recompensa: [%s]" % recompensa
    fim_txt = "Finalizado: [%s]" % finalizado
    acao_txt = "Acao: [%s]" % acao

    if recompensa < 0:
        recompensa_txt = Fore.RED + recompensa_txt + Fore.RESET
    elif recompensa >= 0:
        recompensa_txt = Fore.GREEN + recompensa_txt + Fore.RESET

    if finalizado:
        fim_txt = Fore.RED + fim_txt + Fore.RESET

    print(recompensa_txt, fim_txt, acao_txt)

    if finalizado or paralizado:
        break

    if estado_anterior is not None:
        estado_discreto = discretizar_estado(estado_anterior)
        N[estado_discreto, acao] += 1
        proximo_estado_discreto = discretizar_estado(ambiente)
        
        # Atualizar o valor Q do estado atual e ação com base na fórmula do Q-Learning
        Q[estado_discreto, acao] = (1 - alpha) * Q[estado_discreto, acao] + alpha * (
            recompensa + gamma * np.max(Q[proximo_estado_discreto]) - Q[estado_discreto, acao])

    estado_anterior = ambiente
    total_iteracoes += 1

# Salvando os dados de treinamento em arquivo
salvar_dados('dados_treinamento.pkl')

# Carregando os dados de treinamento de um arquivo
Q, N, total_iteracoes = carregar_dados('dados_treinamento.pkl')

env.close()
