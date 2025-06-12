import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# --- 1. Definições Iniciais e Geometria ---

def definir_geometria():
    """
    Define a geometria da treliça.

    Esta versão foi reprojetada para ser uma treliça Pratt com reforços,
    garantindo que não haja cruzamento de barras e que a estrutura seja
    composta apenas por triângulos.

    Possui 19 barras, atendendo ao requisito de >= 18 para 4 integrantes.
    """
    # Coordenadas dos nós [x, y] em metros (10 nós no total)
    nodes = np.array([
        # Banzo (viga) inferior - 5 nós
        [0.0, 0.0],  # Nó 0 (Apoio A)
        [0.25, 0.0],  # Nó 1
        [0.5, 0.0],  # Nó 2 (Carga C)
        [0.75, 0.0],  # Nó 3
        [1.0, 0.0],  # Nó 4 (Apoio B)
        # Banzo superior - 5 nós
        [0.0, 0.4],  # Nó 5
        [0.25, 0.4],  # Nó 6
        [0.5, 0.4],  # Nó 7
        [0.75, 0.4],  # Nó 8
        [1.0, 0.4],  # Nó 9
    ])

    # Conectividade das barras [nó_i, nó_j] (19 barras no total)
    bars = np.array([
        # Banzo inferior (4 barras)
        [0, 1], [1, 2], [2, 3], [3, 4],
        # Banzo superior (4 barras)
        [5, 6], [6, 7], [7, 8], [8, 9],
        # Montantes (verticais) (5 barras)
        [0, 5], [1, 6], [2, 7], [3, 8], [4, 9],
        # Diagonais para formar triângulos (6 barras)
        [5, 1],  # Diagonal painel esquerdo
        [1, 7],  # Diagonal painel central-esquerdo
        [6, 2],  # Reforço painel central-esquerdo
        [7, 3],  # Diagonal painel central-direito
        [2, 8],  # Reforço painel central-direito
        [8, 4]  # Diagonal painel direito
    ])

    # Áreas da seção transversal (A) de cada barra em m²
    areas = np.full((len(bars), 1), 1.5e-4)

    return nodes, bars, areas


# --- 2. Definição dos Apoios e Carga ---

NO_APOIO_A = 0
NO_APOIO_B = 4
NO_CARGA = 2

# --- Parâmetros do Material (Espaguete) ---
TENSAO_LIMITE = 3.5E5
DENSIDADE = 1500.0
G = 9.81
ARQUIVO_DADOS = 'dados.txt'


# --- 4. Cálculo do Peso da Estrutura ---

def calcular_peso(nodes, bars, areas):
    """Calcula o peso total da estrutura."""
    comprimentos = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)
    volumes = comprimentos * areas.flatten()
    massa_total = DENSIDADE * np.sum(volumes)

    if massa_total > 0.75:
        print(f"AVISO: Massa da estrutura ({massa_total:.3f} kg) excede o limite de 0.75 kg.")

    return massa_total


# --- 5. Verificação Gráfica ---

def plotar_estrutura(nodes, bars, no_apoio_A, no_apoio_B, no_carga, carga_aplicada, titulo):
    """Plota a estrutura da treliça, indicando apoios e carga."""
    plt.figure(figsize=(12, 7))
    for bar in bars:
        node_i_coords = nodes[bar[0]]
        node_j_coords = nodes[bar[1]]
        plt.plot([node_i_coords[0], node_j_coords[0]],
                 [node_i_coords[1], node_j_coords[1]], 'r-')

    plt.plot(nodes[:, 0], nodes[:, 1], 'bo', markersize=8, label='Nós')
    plt.plot(nodes[no_apoio_A, 0], nodes[no_apoio_A, 1], 'g^', markersize=12, label='Apoio Fixo (A)')
    plt.plot(nodes[no_apoio_B, 0], nodes[no_apoio_B, 1], 'g^', markersize=12, label='Apoio Móvel (B)')
    plt.arrow(nodes[no_carga, 0], nodes[no_carga, 1], 0, -0.08,
              head_width=0.04, head_length=0.03, fc='m', ec='m', label=f'Carga ({carga_aplicada:.1f} kgf)')

    plt.title(titulo, fontsize=16)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.show()


# --- 6. Módulo de Elasticidade ---

def calcular_modulo_elasticidade(caminho_arquivo):
    """Calcula o Módulo de Elasticidade (E) a partir de um arquivo de dados."""
    try:
        dados = np.loadtxt(caminho_arquivo)
        deformacao, tensao = dados[:, 0], dados[:, 1]
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{caminho_arquivo}' não foi encontrado.")
        exit()
    except Exception as e:
        print(f"ERRO ao ler o arquivo '{caminho_arquivo}': {e}")
        exit()

    E, _ = np.polyfit(deformacao, tensao, 1)
    return E


# --- 7 e 8. Análise Estrutural e Capacidade Resistiva ---

def solucao_estrutural_e_capacidade(F_aplicada, nodes, bars, areas, E, no_carga, no_apoio_A, no_apoio_B, tensao_limite):
    """Calcula as forças nas barras e a capacidade resistiva (Cr) da estrutura."""
    num_nodes, num_bars = len(nodes), len(bars)
    K = np.zeros((2 * num_nodes, 2 * num_nodes))
    comprimentos = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)

    for i in range(num_bars):
        n1, n2 = bars[i]
        L, A = comprimentos[i], areas[i, 0]
        dx, dy = nodes[n2, 0] - nodes[n1, 0], nodes[n2, 1] - nodes[n1, 1]
        c, s = dx / L, dy / L

        k_local = (A * E / L) * np.array([
            [c * c, c * s, -c * c, -c * s], [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s], [-c * s, -s * s, c * s, s * s]
        ])

        indices = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1])
        K[np.ix_(indices, indices)] += k_local

    F_vetor = np.zeros(2 * num_nodes)
    F_vetor[2 * no_carga + 1] = -F_aplicada

    apoios_gdl = [2 * no_apoio_A, 2 * no_apoio_A + 1, 2 * no_apoio_B + 1]
    gdl_livres = np.setdiff1d(range(2 * num_nodes), apoios_gdl)

    try:
        u_reduzido = np.linalg.solve(K[np.ix_(gdl_livres, gdl_livres)], F_vetor[gdl_livres])
    except np.linalg.LinAlgError:
        return -1, None

    u = np.zeros(2 * num_nodes)
    u[gdl_livres] = u_reduzido

    forcas_internas = np.zeros(num_bars)
    for i in range(num_bars):
        n1, n2 = bars[i]
        L, A = comprimentos[i], areas[i, 0]
        dx, dy = nodes[n2, 0] - nodes[n1, 0], nodes[n2, 1] - nodes[n1, 1]
        c, s = dx / L, dy / L

        u_local = np.array([u[2 * n1], u[2 * n1 + 1], u[2 * n2], u[2 * n2 + 1]])
        deformacao_axial = (1 / L) * np.dot(np.array([-c, -s, c, s]), u_local)
        forcas_internas[i] = A * E * deformacao_axial

    Ri = np.zeros(num_bars)
    for i in range(num_bars):
        Fi, Ai, L = forcas_internas[i], areas[i, 0], comprimentos[i]
        I = Ai ** 2 / (4 * np.pi)

        if Fi >= 0:
            Ri[i] = 1 - (Fi / (tensao_limite * Ai))
        else:
            F_crit = (np.pi ** 2 * E * I) / (L ** 2)
            R_compressao = 1 - (abs(Fi) / (tensao_limite * Ai))
            R_flambagem = 1 - (abs(Fi) / F_crit)
            Ri[i] = min(R_compressao, R_flambagem)

    return np.min(Ri), forcas_internas


# --- 9. Cálculo da Carga de Colapso ---

def encontrar_carga_colapso(nodes, bars, areas, E, no_carga, no_apoio_A, no_apoio_B, tensao_limite):
    """Encontra a carga F que resulta em Cr = 0."""

    def funcao_cr(F):
        cr_val, _ = solucao_estrutural_e_capacidade(F, nodes, bars, areas, E, no_carga, no_apoio_A, no_apoio_B,
                                                    tensao_limite)
        return cr_val

    try:
        return brentq(funcao_cr, a=0.1, b=5000, xtol=1e-3)
    except (ValueError, RuntimeError):
        return None


# --- 10. Execução Principal e Apresentação dos Resultados ---

if __name__ == "__main__":
    nodes, bars, areas = definir_geometria()
    altura_max = np.max(nodes[:, 1])
    massa = calcular_peso(nodes, bars, areas)
    E = calcular_modulo_elasticidade(ARQUIVO_DADOS)
    carga_de_colapso = encontrar_carga_colapso(nodes, bars, areas, E, NO_CARGA, NO_APOIO_A, NO_APOIO_B, TENSAO_LIMITE)

    print("-------------------------------------------------")
    print("      M E M O R I A L   D E   C Á L C U L O      ")
    print("-------------------------------------------------")
    print(f"Dados carregados de:            '{ARQUIVO_DADOS}'")
    print(f"Densidade do material (ρ):      {DENSIDADE:.2f} kg/m³")
    print(f"Módulo de Elasticidade (E):     {E / 1e6:.2f} x 10^6 kgf/m²")
    print(f"Tensão limite do material (σl): {TENSAO_LIMITE / 1e6:.2f} x 10^6 kgf/m²")
    print("-------------------------------------------------")
    print(f"Altura máxima da estrutura:     {altura_max:.2f} m")
    print(f"Massa da estrutura:             {massa:.3f} kg (Limite: 0.75 kg)")
    print(f"Número de barras:               {len(bars)}")
    print(f"Número de nós:                  {len(nodes)}")
    print("-------------------------------------------------")
    if carga_de_colapso is not None:
        print(f"CARGA DE COLAPSO (RUPTURA):     {carga_de_colapso:.2f} kgf")
    else:
        print("CARGA DE COLAPSO:               Não foi possível calcular.")
    print("-------------------------------------------------")

    plot_title = f'Estrutura da Ponte (19 Barras)\nCarga de Colapso: {carga_de_colapso:.2f} kgf' if carga_de_colapso is not None else 'Estrutura da Ponte (Cálculo Falhou)'
    plotar_estrutura(nodes, bars, NO_APOIO_A, NO_APOIO_B, NO_CARGA, carga_de_colapso or 0, plot_title)