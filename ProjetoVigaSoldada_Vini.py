import math
import numpy as np
import pandas as pd
import seaborn as sns
from skopt import gp_minimize
from decimal import Decimal, getcontext
from mealpy.evolutionary_based.DE import JADE
from mealpy import math_based
from mealpy import ( #Modelos do mealpy
    FloatVar,
    Problem,
    PSO,
    MFO,
    EFO,
    ASO,
    MA,
    ACOR,
    BBO,
    GSKA,
    SA,


)

getcontext().prec = 50 #define a precisão dos cálculos com números decimais

P = 6000            # Carga
L = 14              # Comprimento da viga
E = 30 * (10 ** 6)  # Módulo de elasticidade
G = 12 * (10 ** 6)  # Módulo de cisalhamento

def Calculate_M(parameters):
    return P * (L + (parameters[1] / 2))

def Calculate_R(parameters):
    return (1 / 2) * math.sqrt((parameters[1] ** 2) + ((parameters[0] + parameters[2]) ** 2))

def Calculate_J(parameters):
    return 2 * math.sqrt(2) * parameters[0] * parameters[1] * (((parameters[1] ** 2) / 12) + (((parameters[0] + parameters[2]) / 2) ** 2))

def Calculate_T(parameters):    #Tau ≤ 13600 (tensão de torção)
    def Calculate_Tl():    
        return P / (math.sqrt(2) * parameters[0] * parameters[1])
    
    def Calculate_Tll():
        return Calculate_M(parameters)  * Calculate_R(parameters) / Calculate_J(parameters)
    
    Tl = Calculate_Tl()
    Tll = Calculate_Tll()
    return math.sqrt((Tl ** 2) + (2 * Tl * Tll * (parameters[1] / (2 * Calculate_R(parameters)))) + (Tll ** 2))

def Calculate_Lowercase_Sigma(parameters): #Sigma ≤ 30000 (tensão axial)
    return ((6 * P * L) / (parameters[3] * ((parameters[2]) ** 2)))

def Calculate_Lowercase_Delta(parameters): #Delta ≤ 0,25 (deformação)
    return (4  * P * (L ** 3)) / (E * (parameters[2] ** 3) * parameters[3])

def Calculate_P_c(parameters): #Pc ≥ 6000000 (carga crítica)
    return ((4.013 * E) / (L ** 2)) * ((parameters[2] * (parameters[3] ** 3)) / 6) * (1 - (parameters[2] / (2 * L)) * math.sqrt(E / (4 * G)))

def violations(parameters): # Verifica violação de restrições
    tau   = Calculate_T(parameters)
    sigma = Calculate_Lowercase_Sigma(parameters)
    Pc    = Calculate_P_c(parameters)
    delta     = Calculate_Lowercase_Delta(parameters)

    return np.array([
        max(0.0, tau - 13600.0),
        max(0.0, sigma - 30000.0),
        max(0.0, (P) - (Pc)),
        max(0.0, delta - 0.25),
        max(0.0, parameters[0] - parameters[3])
    ])

def Calculate_Penalty(parameters):  # Penaliza soluções inválidas
    #penalty = (np.multiply(violations(parameters), 100)).sum()
    penalty = violations(parameters).sum() * 10
    #penalty = penalty * 100
    return penalty

def Cost_Minimize(parameters):  #Função custo total
    parameters = np.array(parameters, dtype=Decimal)
    return Calculate_Penalty(parameters) + (1.10471 * (parameters[0] ** 2) * parameters[1]) + (0.04811 * parameters[2] * parameters[3] * (14 + parameters[1]))

# Define figure parameters
FS = (8, 7)
DPI = 70
SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 26

# cmap = sns.diverging_palette(
#     h_neg=18,
#     h_pos=248,
#     s=140,
#     as_cmap=True,
# )

# sns.set_context("poster")
# sns.set_theme(font="Times New Roman", font_scale=2)
# sns.color_palette("husl", 9)

D = 4
ME = 10000
PS = 600

problem_dict1 = { #Configuração do Problema
    "bounds": FloatVar(lb=(0.125, 0.1, 0.1, 0.1), ub=(2, 10, 10, 10), name="delta"),
    "obj_func": Cost_Minimize,
    "minmax": "min",
    "log_to": "none",
}

modelList = [] #Lista de modelos
#modelList.append(PSO.AIW_PSO(epoch=ME, pop_size=PS, c1=2.05, c2=2.05, alpha=0.8))
modelList.append(PSO.OriginalPSO(epoch=ME, pop_size=PS, c1=2.05, c2=2.05, w_min=0.7, w_max=0.8)) #Modelo usado
#modelList.append(MFO.OriginalMFO(epoch=ME, pop_size=PS))
modelList.append(JADE(epoch=ME, pop_size=PS))

#modelList.append(EFO.DevEFO(epoch=ME, pop_size=PS, r_rate = 0.3, ps_rate = 0.95, p_field = 0.3, n_field = 0.5))
#modelList.append(ASO.OriginalASO(epoch=ME, pop_size=PS, alpha = 50, beta = 0.2))
# modelList.append(MA.OriginalMA(epoch=ME, pop_size=PS, pc = 0.85, pm = 0.15, p_local = 0.5, max_local_gens = 10, bits_per_param = 4))
# modelList.append(ACOR.OriginalACOR(epoch=ME, pop_size=PS, sample_count = 25, intent_factor = 0.5, zeta = 1.0))
# modelList.append(BBO.DevBBO(epoch=ME, pop_size=PS, p_m=0.01, n_elites=2))
# modelList.append(GSKA.DevGSKA(epoch=ME, pop_size=PS, pb = 0.1, kr = 0.9))
# modelList.append(SA.GaussianSA(epoch=ME, pop_size=2, temp_init = 100, cooling_rate = 0.99, scale = 0.1))

for model in modelList:     #Percorre a lista de modelos, tentando resolver o problema
    print('Initializing problem solving')
    g_best1 = model.solve(problem_dict1)
    # g_best = g_best1
    # nm = 1

    print(f"""
        Model: {model.name},
            Solution: {g_best1.solution},
            Fitness: {g_best1.target.fitness},
            tau: {Calculate_T(g_best1.solution)},
            sigma: {Calculate_Lowercase_Sigma(g_best1.solution)},
            Pc: {Calculate_P_c(g_best1.solution)},
            delta: {Calculate_Lowercase_Delta(g_best1.solution)},
            violations: {Calculate_Penalty(g_best1.solution)},
""")
    # print(f"Model: {model.name}, Fitness: {g_best1.target.fitness}")

    # print("Best model: ")
    # print(f"#{nm}, Model: {model.name}, Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")

    # dimensions = []
    # for i in range(D):
    #     dimensions.append((lb, ub))

    # res = gp_minimize(Cost_Minimize,   # the function to minimize
    #                 [(0.125, 2), (0.1, 10), (0.1, 10), (0.1, 10)],         # the bounds on each dimension of x
    #                 initial_point_generator="random",
    #                 acq_func="gp_hedge",      # the acquisition function
    #                 n_calls=100,         # the number of evaluations of f
    #                 n_random_starts=80,  # the number of random initialization points
    # #                  noise=0.1**2,       # the noise level (optional)
    #                 n_jobs=1,
    #                 random_state=42)   # the random seed

    # "x^*=%.14f, f(x^*)=%.14f" % (res.x[0], res.fun)
    # g_best.solution[0]