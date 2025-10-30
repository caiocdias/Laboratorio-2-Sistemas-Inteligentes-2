# ====== Imports (mesmos do seu esqueleto) ======
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

# ====== Utilitários ======
def distancia(p1, p2):
    # p1, p2: arrays/listas 2D
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def make_circles(n_samples=2000, noise=0.05, seed=7):
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    t = rng.uniform(0, 2*np.pi, n)
    outer = np.stack([np.cos(t), np.sin(t)], axis=1)
    inner = 0.5 * np.stack([np.cos(t), np.sin(t)], axis=1)
    X = np.vstack([outer, inner])
    X += rng.normal(0, noise, X.shape)
    return X

# ====== Dataset ======
seed_global = 11  # para reprodutibilidade em tudo
X = make_circles(n_samples=3000, noise=0.05, seed=seed_global)

plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], s=2, alpha=0.5)
plt.axis('equal')
plt.title("Dataset - Círculos Concêntricos")
plt.show()

# ======================================================================================
#                                     GNG
# ======================================================================================
# Hiperparâmetros do enunciado
max_nodes    = 180
max_age      = 70
lambda_steps = 120
eps_b        = 0.06
eps_n        = 0.008
alpha        = 0.5
d_err        = 0.996
rng = np.random.default_rng(seed_global)

# ----- Inicialização do grafo com 2 nós em posições amostradas dos dados -----
G = nx.Graph()
idx = rng.choice(len(X), size=2, replace=False)
for i, idp in enumerate(idx, start=1):
    G.add_node(i, pos=X[idp].astype(float), error=0.0)
G.add_edge(1, 2, idade=0)

# ----- Funções auxiliares do GNG -----
def nearest_two_nodes(G, x):
    # retorna (s1, s2) ids dos dois nós mais próximos de x
    dists = []
    for node in G.nodes:
        p = G.nodes[node]["pos"]
        dists.append((node, np.linalg.norm(x - p)))
    dists.sort(key=lambda t: t[1])
    return dists[0][0], dists[1][0]

def envelhece_arestas_de(G, s1):
    for v in list(G.neighbors(s1)):
        G[s1][v]["idade"] += 1

def remove_arestas_antigas_e_nos_isolados(G, max_age):
    antigas = [(u,v) for u,v,a in G.edges(data=True) if a.get("idade", 0) > max_age]
    G.remove_edges_from(antigas)
    isolados = [n for n in G.nodes if G.degree(n) == 0]
    G.remove_nodes_from(isolados)

def insere_no(G):
    # q = nó com maior erro
    q = max(G.nodes, key=lambda n: G.nodes[n]["error"])
    viz_q = list(G.neighbors(q))
    if len(viz_q) == 0:
        # se isolado, conecta ao mais próximo e volta (caso raro)
        # encontra f como nó mais próximo a q
        qpos = G.nodes[q]["pos"]
        candidatos = [(n, np.linalg.norm(G.nodes[n]["pos"] - qpos)) for n in G.nodes if n != q]
        if len(candidatos) == 0:
            return
        f = min(candidatos, key=lambda t: t[1])[0]
    else:
        # f = vizinho de q com maior erro
        f = max(viz_q, key=lambda n: G.nodes[n]["error"])

    # cria novo nó r no meio de q e f
    r_id = (max(G.nodes) + 1) if len(G.nodes) > 0 else 1
    pos_q = G.nodes[q]["pos"]
    pos_f = G.nodes[f]["pos"]
    pos_r = 0.5 * (pos_q + pos_f)

    # remove aresta q-f e adiciona r
    if G.has_edge(q, f):
        G.remove_edge(q, f)
    G.add_node(r_id, pos=pos_r.astype(float), error=0.0)
    G.add_edge(q, r_id, idade=0)
    G.add_edge(r_id, f, idade=0)

    # reduz erro de q e f, inicializa de r
    G.nodes[q]["error"] *= alpha
    G.nodes[f]["error"] *= alpha
    G.nodes[r_id]["error"] = G.nodes[q]["error"].copy()

def treina_gng(G, X, max_nodes, lambda_steps, eps_b, eps_n, max_age, d_err, rng):
    t = 0
    while len(G.nodes) < max_nodes:
        x = X[rng.integers(len(X))]

        # encontra dois mais próximos
        s1, s2 = nearest_two_nodes(G, x)

        # aumenta idade das arestas a partir de s1
        envelhece_arestas_de(G, s1)

        # atualiza erro do vencedor
        p1 = G.nodes[s1]["pos"]
        dist2 = np.sum((x - p1)**2)
        G.nodes[s1]["error"] += dist2

        # move s1 e seus vizinhos
        G.nodes[s1]["pos"] = p1 + eps_b * (x - p1)
        for v in G.neighbors(s1):
            pv = G.nodes[v]["pos"]
            G.nodes[v]["pos"] = pv + eps_n * (x - pv)

        # conecta s1 e s2 (idade=0) ou reseta idade se já existir
        if G.has_edge(s1, s2):
            G[s1][s2]["idade"] = 0
        else:
            G.add_edge(s1, s2, idade=0)

        # remove arestas antigas e nós isolados
        remove_arestas_antigas_e_nos_isolados(G, max_age)

        # decai erros globais
        for n in G.nodes:
            G.nodes[n]["error"] *= d_err

        # a cada lambda_steps insere novo nó na região de maior erro
        t += 1
        if t % lambda_steps == 0 and len(G.nodes) < max_nodes:
            insere_no(G)

    return G

# ----- Treinamento GNG -----
G = treina_gng(G, X, max_nodes, lambda_steps, eps_b, eps_n, max_age, d_err, rng)

# ----- Plot final GNG sobre os dados -----
pos = {n: G.nodes[n]["pos"] for n in G.nodes}
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], s=2, alpha=0.25, label="dados")
nx.draw(G, pos, node_size=12, width=0.6, with_labels=False)
plt.axis('equal')
plt.title(f"GNG final (nós={G.number_of_nodes()}, arestas={G.number_of_edges()})")
plt.legend(loc="upper right")
plt.show()

# ======================================================================================
#                                     SOM 10x10
# ======================================================================================
def som_train(X, m=10, n=10, lr0=0.5, sigma0=None, steps=80000, seed=11):
    rng = np.random.default_rng(seed)
    if sigma0 is None:
        sigma0 = max(m, n) / 2.0

    # inicializa W com amostras dos dados (mais estável que uniforme bruto)
    idx = rng.choice(len(X), size=m*n, replace=False)
    W = X[idx].reshape(m, n, X.shape[1]).astype(float)

    # coordenadas da grade para vizinhança gaussiana
    ii, jj = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    grid = np.stack([ii, jj], axis=2)  # (m,n,2)

    # constantes de decaimento
    tau_sigma = steps / np.log(sigma0 + 1e-8)
    tau_lr    = steps

    for t in range(steps):
        x = X[rng.integers(len(X))]

        # BMU
        dif = W - x  # (m,n,2)
        dist2 = np.sum(dif*dif, axis=2)  # (m,n)
        bmu = np.unravel_index(np.argmin(dist2), (m, n))

        # taxas com decaimento
        lr_t = lr0 * np.exp(-t / tau_lr)
        sigma_t = sigma0 * np.exp(-t / tau_sigma)

        # vizinhança gaussiana no grid
        dgrid2 = (grid[:,:,0]-bmu[0])**2 + (grid[:,:,1]-bmu[1])**2
        h = np.exp(-dgrid2 / (2.0 * (sigma_t**2) + 1e-12))  # (m,n)

        # atualização
        W += lr_t * h[:,:,None] * (x - W)

    return W

# ----- Treinamento SOM -----
m, n = 10, 10
W = som_train(X, m=m, n=n, lr0=0.5, sigma0=max(m,n)/2, steps=80000, seed=seed_global)

# ----- Plot SOM sobre os dados, com malha 10×10 -----
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], s=2, alpha=0.25, label="dados")
# neurônios
plt.scatter(W[:,:,0].ravel(), W[:,:,1].ravel(), s=15, label="neurônios SOM")

# conexões horizontais e verticais da grade
for i in range(m):
    for j in range(n):
        if i+1 < m:
            xline = [W[i,j,0], W[i+1,j,0]]
            yline = [W[i,j,1], W[i+1,j,1]]
            plt.plot(xline, yline, linewidth=0.6)
        if j+1 < n:
            xline = [W[i,j,0], W[i,j+1,0]]
            yline = [W[i,j,1], W[i,j+1,1]]
            plt.plot(xline, yline, linewidth=0.6)

plt.axis('equal')
plt.title("SOM 10×10 final sobre os dados")
plt.legend(loc="upper right")
plt.show()
