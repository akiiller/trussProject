import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


# --- 1. Definições Iniciais e Geometria ---

def definir_geometria():
    """
    Define a geometria da treliça: coordenadas dos nós, conectividade das barras e áreas.
    A estrutura deve vencer um vão de 1m, com altura máxima de 0.5m.
    Nós obrigatórios: A(0,0), B(1,0) e C(0.5,0).
    """
    # Coordenadas dos nós [x, y] em metros
    nodes = np.array([
        [0.0, 0.0],  # Nó 0 (Apoio A)
        [0.25, 0.0],  # Nó 1
        [0.5, 0.0],  # Nó 2
        [0.75, 0.0],  # Nó 3 (Carga C)
        [1.0, 0.0],  # Nó  4
        [1.25, 0.0], # Nó  5
         [1.5, 0.0],# Nó 6 (Apoio B)
        [0.25, 0.25],  # Nó 7
        [0.5, 0.25],  # Nó 8
        [0.75, 0.25],  # Nó 9
        [1.0, 0.25], # Nó 10
         [1.25, 0.25] # Nó 11
    ])

    # Conectividade das barras [nó_i, nó_j]
    # Mínimo de 18 barras para 8 integrante (10 + 2*1)
    bars = np.array([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],  # Banzos inferiores
        [7, 8], [8, 9],[9, 10], [ 10, 11], # Banzos superiores
        [0, 7], [1, 7], [2, 7], [2,8], # Diagonais e montantes esquerdos
        [2, 9], [3, 9], [4, 9],  # Diagonais e montantes centrais e direitos
        [4, 10] ,[4, 11] , [5, 11] ,[6, 11] ,  # Montante direito
    ])

    # Áreas da seção transversal (A) de cada barra em m²
    # Considera-se seção circular
    areas = np.full((len(bars), 1), 1.5e-4)  # Área uniforme inicial

    return nodes, bars, areas


# --- 2. Definição dos Apoios e Carga ---

# Nós de apoio A e B
NO_APOIO_A = 0
NO_APOIO_B = 6

# Nó de aplicação da carga e valor de teste
NO_CARGA = 3
CARGA_TESTE_KGF = 1.0  # 1.0 kgf para teste inicial

# --- Parâmetros do Material (Espaguete) ---

# Tensão limite do material em kgf/m²
TENSAO_LIMITE = 3.5E5

# Densidade do espaguete em kg/m³ (valor típico, pois não foi fornecido)
DENSIDADE = 1500.0
G = 9.81  # Aceleração da gravidade em m/s²

# Nome do arquivo com dados de tensão x deformação
ARQUIVO_DADOS = 'dados.txt'


# --- 4. Cálculo do Peso da Estrutura ---

def calcular_peso(nodes, bars, areas):
    """
    Calcula o peso total da estrutura com base na geometria.
    O peso total é a somatória de (densidade * g * L * A) para cada barra.
    """
    comprimentos = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)
    volumes = comprimentos * areas.flatten()
    massa_total = DENSIDADE * np.sum(volumes)
    peso_total_newtons = massa_total * G

    # O limite é de 0.75 kg de massa
    if massa_total > 0.75:
        print(f"AVISO: Massa da estrutura ({massa_total:.3f} kg) excede o limite de 0.75 kg.")

    return massa_total, peso_total_newtons


# --- 5. Verificação Gráfica ---

def plotar_estrutura(nodes, bars, no_apoio_A, no_apoio_B, no_carga, carga_aplicada, titulo):
    """
    Plota a estrutura da treliça, indicando apoios e carga.
    """
    plt.figure(figsize=(12, 6))
    for i, bar in enumerate(bars):
        node_i_coords = nodes[bar[0]]
        node_j_coords = nodes[bar[1]]
        plt.plot([node_i_coords[0], node_j_coords[0]],
                 [node_i_coords[1], node_j_coords[1]], 'r-')

    plt.plot(nodes[:, 0], nodes[:, 1], 'bo', markersize=8, label='Nós')

    # Marca os apoios
    plt.plot(nodes[no_apoio_A, 0], nodes[no_apoio_A, 1], 'g^', markersize=12, label='Apoio A')
    plt.plot(nodes[no_apoio_B, 0], nodes[no_apoio_B, 1], 'g^', markersize=12, label='Apoio B')

    # Marca a carga
    plt.arrow(nodes[no_carga, 0], nodes[no_carga, 1], 0, -0.1,
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
    """
    Calcula o Módulo de Elasticidade (E) por regressão linear de dados
    de um ensaio de tração lidos do arquivo 'dados.txt'.
    Tensão = E * Deformação
    """
    try:
        # Carrega os dados do arquivo. A primeira coluna é deformação, a segunda é tensão
        dados = np.loadtxt(caminho_arquivo)
        deformacao = dados[:, 0]
        tensao = dados[:, 1]
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{caminho_arquivo}' não foi encontrado.")
        print("Certifique-se de que ele está na mesma pasta que o script Python.")
        print("O programa será encerrado.")
        exit()
    except Exception as e:
        print(f"ERRO ao ler o arquivo '{caminho_arquivo}': {e}")
        print("O programa será encerrado.")
        exit()

    # Regressão linear para encontrar a inclinação (E)
    E, _ = np.polyfit(deformacao, tensao, 1)

    return E  # em kgf/m²


# --- 7 e 8. Análise Estrutural e Capacidade Resistiva ---

def solucao_estrutural_e_capacidade(F_aplicada, nodes, bars, areas, E, no_carga, no_apoio_A, no_apoio_B, tensao_limite):
    """
    Calcula as forças nas barras e a capacidade resistiva (Cr) da estrutura.
    """
    num_nodes = len(nodes)
    num_bars = len(bars)

    # Montagem da matriz de rigidez global K
    K = np.zeros((2 * num_nodes, 2 * num_nodes))

    comprimentos = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)

    for i in range(num_bars):
        n1, n2 = bars[i]
        L = comprimentos[i]
        A = areas[i, 0]

        # Cossenos diretores
        dx = nodes[n2, 0] - nodes[n1, 0]
        dy = nodes[n2, 1] - nodes[n1, 1]
        c, s = dx / L, dy / L

        # Matriz de rigidez local em coordenadas globais
        k_local = (A * E / L) * np.array([
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s]
        ])

        # Montagem na matriz global
        indices = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1])
        K[np.ix_(indices, indices)] += k_local

    # Vetor de forças F
    F_vetor = np.zeros(2 * num_nodes)
    F_vetor[2 * no_carga + 1] = -F_aplicada  # Carga vertical para baixo

    # Aplicação das condições de contorno (apoios)
    apoios_gdl = [2 * no_apoio_A, 2 * no_apoio_A + 1, 2 * no_apoio_B + 1]  # Apoio fixo em A, móvel em B
    gdl_livres = np.setdiff1d(range(2 * num_nodes), apoios_gdl)

    K_reduzida = K[np.ix_(gdl_livres, gdl_livres)]
    F_reduzido = F_vetor[gdl_livres]

    # Resolução para os deslocamentos
    try:
        u_reduzido = np.linalg.solve(K_reduzida, F_reduzido)
    except np.linalg.LinAlgError:
        return -1, None

    u = np.zeros(2 * num_nodes)
    u[gdl_livres] = u_reduzido

    # Cálculo das forças internas (Fi) em cada barra
    forcas_internas = np.zeros(num_bars)
    for i in range(num_bars):
        n1, n2 = bars[i]
        L = comprimentos[i]
        A = areas[i, 0]
        dx = nodes[n2, 0] - nodes[n1, 0]
        dy = nodes[n2, 1] - nodes[n1, 1]
        c, s = dx / L, dy / L

        u_local = np.array([u[2 * n1], u[2 * n1 + 1], u[2 * n2], u[2 * n2 + 1]])
        deformacao_axial = (1 / L) * np.dot(np.array([-c, -s, c, s]), u_local)
        forcas_internas[i] = A * E * deformacao_axial

        # Cálculo da Capacidade Resistiva (Cr)
    Ri = np.zeros(num_bars)
    for i in range(num_bars):
        Fi = forcas_internas[i]
        Ai = areas[i, 0]
        L = comprimentos[i]

        I = Ai ** 2 / (4 * np.pi)

        if Fi >= 0:  # Tração
            Ri[i] = 1 - (Fi / (tensao_limite * Ai))
        else:  # Compressão
            F_crit = (np.pi ** 2 * E * I) / (L ** 2)
            R_compressao = 1 - (abs(Fi) / (tensao_limite * Ai))
            R_flambagem = 1 - (abs(Fi) / F_crit)
            Ri[i] = min(R_compressao, R_flambagem)

    Cr = np.min(Ri)
    return Cr, forcas_internas


# --- 9. Cálculo da Carga de Colapso ---

def encontrar_carga_colapso(nodes, bars, areas, E, no_carga, no_apoio_A, no_apoio_B, tensao_limite):
    """
    Encontra a carga F que resulta em Cr = 0 usando um método de busca de raiz.
    """

    def funcao_cr(F):
        cr_val, _ = solucao_estrutural_e_capacidade(F, nodes, bars, areas, E, no_carga, no_apoio_A, no_apoio_B,
                                                    tensao_limite)
        return cr_val

    try:
        carga_colapso = brentq(funcao_cr, a=0.1, b=5000, xtol=1e-3)
        return carga_colapso
    except (ValueError, RuntimeError):
        return None


# --- 10. Execução Principal e Apresentação dos Resultados ---

if __name__ == "__main__":
    # 1. Geometria
    nodes, bars, areas = definir_geometria()
    altura_max = np.max(nodes[:, 1])

    # 4. Peso
    massa, peso_N = calcular_peso(nodes, bars, areas)

    # 6. Módulo de Elasticidade com base no arquivo de dados
    E = calcular_modulo_elasticidade(ARQUIVO_DADOS)

    # 9. Carga de Colapso
    carga_de_colapso = encontrar_carga_colapso(nodes, bars, areas, E, NO_CARGA, NO_APOIO_A, NO_APOIO_B, TENSAO_LIMITE)

    # 10. Resultados Finais
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
        print("CARGA DE COLAPSO:               Não foi possível calcular. Verifique o intervalo de busca ou os dados.")
    print("-------------------------------------------------")

    # 5. Plotagem Final
    if carga_de_colapso is not None:
        plot_title = f'Estrutura da Ponte de Espaguete\nCarga de Colapso: {carga_de_colapso:.2f} kgf'
        plotar_estrutura(nodes, bars, NO_APOIO_A, NO_APOIO_B, NO_CARGA, carga_de_colapso, plot_title)
    else:
        plotar_estrutura(nodes, bars, NO_APOIO_A, NO_APOIO_B, NO_CARGA, 0,
                         'Estrutura da Ponte (Cálculo da Carga de Colapso Falhou)')