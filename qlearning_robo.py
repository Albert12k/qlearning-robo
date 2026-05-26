import random
import time

# ─────────────────────────────────────────
# 1. CONFIGURAÇÃO DO AMBIENTE
# ─────────────────────────────────────────

TAMANHO = 5  # grid 5x5

# Posições especiais
INICIO    = (0, 0)
OBJETIVO  = (4, 4)
OBSTACULOS = [(0, 3), (1, 1), (2, 4), (3, 1), (0, 1)]

# Ações possíveis: (delta_linha, delta_coluna)
ACOES = {
    0: (-1,  0),  # cima
    1: ( 1,  0),  # baixo
    2: ( 0, -1),  # esquerda
    3: ( 0,  1),  # direita
}
NOMES_ACOES = {0: "cima", 1: "baixo", 2: "esquerda", 3: "direita"}

# Recompensas
RECOMPENSA_OBJETIVO  =  10
RECOMPENSA_OBSTACULO = -10
RECOMPENSA_MOVIMENTO =  -1

# ─────────────────────────────────────────
# 2. PARÂMETROS DO Q-LEARNING
# ─────────────────────────────────────────

ALPHA   = 0.5   # taxa de aprendizado
GAMMA   = 0.9   # fator de desconto
EPSILON = 0.9   # exploração inicial (decai ao longo do treino)
EPISODIOS = 1000
MAX_PASSOS = 100  # limite por episódio

# ─────────────────────────────────────────
# 3. FUNÇÕES AUXILIARES
# ─────────────────────────────────────────

def dentro_do_grid(r, c):
    return 0 <= r < TAMANHO and 0 <= c < TAMANHO

def obter_recompensa(r, c):
    if (r, c) == OBJETIVO:
        return RECOMPENSA_OBJETIVO
    if (r, c) in OBSTACULOS:
        return RECOMPENSA_OBSTACULO
    return RECOMPENSA_MOVIMENTO

def escolher_acao(Q, r, c, epsilon):
    """Política epsilon-greedy: explora aleatoriamente ou escolhe a melhor ação."""
    if random.random() < epsilon:
        return random.randint(0, 3)  # exploração
    return max(ACOES.keys(), key=lambda a: Q[r][c][a])  # explotação

def exibir_ambiente(agent_pos=None, caminho=None):
    """Exibe o grid no terminal com legendas visuais."""
    print()
    for r in range(TAMANHO):
        linha = ""
        for c in range(TAMANHO):
            pos = (r, c)
            if caminho and pos in caminho and pos != INICIO and pos != OBJETIVO:
                celula = "[·]"
            elif agent_pos and pos == agent_pos:
                celula = "[R]"
            elif pos == INICIO:
                celula = "[S]"
            elif pos == OBJETIVO:
                celula = "[G]"
            elif pos in OBSTACULOS:
                celula = "[X]"
            else:
                celula = "[ ]"
            linha += celula + " "
        print(linha)
    print()
    print("  Legenda: [S]=início  [G]=objetivo  [X]=obstáculo  [R]=robô  [·]=caminho")
    print()

def exibir_tabela_q(Q):
    """Exibe a tabela Q formatada."""
    print("\n=== TABELA Q (valores por estado) ===")
    print(f"{'Estado':<12} {'Cima':>8} {'Baixo':>8} {'Esquerda':>10} {'Direita':>8}")
    print("-" * 50)
    for r in range(TAMANHO):
        for c in range(TAMANHO):
            estado = f"({r},{c})"
            vals = Q[r][c]
            print(f"{estado:<12} {vals[0]:>8.2f} {vals[1]:>8.2f} {vals[2]:>10.2f} {vals[3]:>8.2f}")

# ─────────────────────────────────────────
# 4. INICIALIZAÇÃO DA TABELA Q
# ─────────────────────────────────────────

# Q[linha][coluna][acao] = valor estimado
Q = [[[0.0] * 4 for _ in range(TAMANHO)] for _ in range(TAMANHO)]

# ─────────────────────────────────────────
# 5. EXIBIR AMBIENTE INICIAL
# ─────────────────────────────────────────

print("=" * 55)
print("  APRENDIZADO POR REFORÇO — Q-LEARNING")
print("=" * 55)
print(f"\nAmbiente: grid {TAMANHO}x{TAMANHO}")
print(f"Início:   {INICIO}  |  Objetivo: {OBJETIVO}")
print(f"Obstáculos: {OBSTACULOS}")
print(f"\nParâmetros:")
print(f"  α (alpha)   = {ALPHA}   — taxa de aprendizado")
print(f"  γ (gamma)   = {GAMMA}   — fator de desconto")
print(f"  ε (epsilon) = {EPSILON}   — exploração inicial")
print(f"  Episódios   = {EPISODIOS}")

print("\n--- Ambiente inicial ---")
exibir_ambiente()

# ─────────────────────────────────────────
# 6. TREINAMENTO COM Q-LEARNING
# ─────────────────────────────────────────

print("=" * 55)
print("  TREINAMENTO")
print("=" * 55)

recompensas_por_episodio = []
epsilon = EPSILON
epsilon_decay = EPSILON / EPISODIOS  # decaimento linear

inicio_treino = time.time()

for ep in range(EPISODIOS):
    r, c = INICIO
    recompensa_total = 0

    for passo in range(MAX_PASSOS):
        acao = escolher_acao(Q, r, c, epsilon)
        dr, dc = ACOES[acao]
        nr, nc = r + dr, c + dc

        # Se sair do grid, fica no mesmo lugar
        if not dentro_do_grid(nr, nc):
            nr, nc = r, c

        recompensa = obter_recompensa(nr, nc)
        recompensa_total += recompensa

        # Atualização Q (fórmula de Bellman)
        melhor_proximo = max(Q[nr][nc])
        Q[r][c][acao] += ALPHA * (recompensa + GAMMA * melhor_proximo - Q[r][c][acao])

        r, c = nr, nc

        # Termina o episódio se chegou ao objetivo ou caiu em obstáculo
        if (r, c) == OBJETIVO or (r, c) in OBSTACULOS:
            break

    recompensas_por_episodio.append(recompensa_total)
    epsilon = max(0.05, epsilon - epsilon_decay)  # decaimento com mínimo de 5%

    # Log a cada 100 episódios
    if (ep + 1) % 100 == 0:
        media = sum(recompensas_por_episodio[-100:]) / 100
        print(f"  Episódio {ep+1:>4}/{EPISODIOS}  |  Recompensa média (últimos 100): {media:>7.2f}  |  ε = {epsilon:.3f}")

tempo_treino = time.time() - inicio_treino
print(f"\nTreinamento concluído em {tempo_treino:.2f}s")

# ─────────────────────────────────────────
# 7. EVOLUÇÃO DO APRENDIZADO
# ─────────────────────────────────────────

print("\n=== EVOLUÇÃO DAS RECOMPENSAS (média por bloco de 100 episódios) ===")
print(f"{'Bloco':<10} {'Recompensa média':>18} {'Barra'}")
print("-" * 50)

for i in range(0, EPISODIOS, 100):
    bloco = recompensas_por_episodio[i:i+100]
    media = sum(bloco) / len(bloco)
    barra_len = max(0, int((media + 15) * 1.5))  # normaliza para visualização
    barra = "█" * min(barra_len, 40)
    print(f"  {i+1:>3}-{i+100:<5}  {media:>10.2f}       {barra}")

# ─────────────────────────────────────────
# 8. DEMONSTRAÇÃO DO CAMINHO APRENDIDO
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("  DEMONSTRAÇÃO — CAMINHO APRENDIDO PELO AGENTE")
print("=" * 55)

r, c = INICIO
caminho = [(r, c)]
acoes_tomadas = []
recompensa_demo = 0
sucesso = False

for passo in range(MAX_PASSOS):
    acao = escolher_acao(Q, r, c, epsilon=0)  # sem exploração — apenas explotação
    dr, dc = ACOES[acao]
    nr, nc = r + dr, c + dc

    if not dentro_do_grid(nr, nc):
        nr, nc = r, c

    recompensa = obter_recompensa(nr, nc)
    recompensa_demo += recompensa
    acoes_tomadas.append(NOMES_ACOES[acao])

    r, c = nr, nc
    caminho.append((r, c))

    if (r, c) == OBJETIVO:
        sucesso = True
        break
    if (r, c) in OBSTACULOS:
        break

print(f"\nCaminho percorrido: {' → '.join(str(p) for p in caminho)}")
print(f"Ações tomadas:      {' → '.join(acoes_tomadas)}")
print(f"Passos:             {len(acoes_tomadas)}")
print(f"Recompensa total:   {recompensa_demo}")
print(f"Resultado:          {'✓ OBJETIVO ALCANÇADO' if sucesso else '✗ Não alcançou o objetivo'}")

print("\n--- Ambiente com caminho final ---")
exibir_ambiente(agent_pos=(r, c), caminho=set(caminho))

# ─────────────────────────────────────────
# 9. TABELA Q FINAL
# ─────────────────────────────────────────

exibir_tabela_q(Q)

print("\n" + "=" * 55)
print("  FIM DA EXECUÇÃO")
print("=" * 55)
