from pennylane import numpy as np
import argparse
import pandas as pd
import pickle
import pennylane as qml
from pennylane import qaoa
from tqdm import tqdm
from utils_dicke_states import dicke_state
from utils import (GenerateTermsMatrix, GenerateConstrainTerms, solve_qubo, from_binary_to_int, generate_cost_hamiltonian, GenerateGraph, normalized_performance, compute_score2)
np.random.seed(412)

parser = argparse.ArgumentParser()
#Adding optional parameters
parser.add_argument('-STRUCTURE',
                    type=str,
                    default = "WarrenTruss_5")

parser.add_argument('-MODES',
                    type=str,
                    default="0,1,4")

parser.add_argument('-GRANULARITY',
                    type=int,
                    default=100)

parser.add_argument('-SUP',
                    type=float,
                    default=np.pi)

parser.add_argument('-MIXER',
                    type=str,
                    default = "trad")

parser.add_argument('-N_S',
                    type=int,
                    default=2)

parser.add_argument('-ALPHA',
                    type=int,
                    default=1)

parser.add_argument('-CIRCUIT_DEPTHS',
                    type=str,
                    default="1,2,4,8")



#Parsing the argument
args=parser.parse_args()
print(args)
STRUCTURE = args.STRUCTURE
GRANULARITY = args.GRANULARITY
MODES = [int(item) for item in args.MODES.split(',')]
print(f"Modes: {MODES}")
SUP = args.SUP
MIXER = args.MIXER
ALPHA = args.ALPHA
N_S = args.N_S
CIRCUIT_DEPTHS = [int(item) for item in args.CIRCUIT_DEPTHS.split(',')]

# Read data
with open(f'data/{STRUCTURE}.pickle', 'rb') as handle:
        data = pickle.load(handle)

stiffness_matrix = data["K"]
mass_matrix = data["M"]
modal_matrix = data["Phi"][:, MODES] # Modes to reach 95% MMP
N = len(stiffness_matrix)

# Generate the MSE terms
A = GenerateTermsMatrix(modal_matrix, stiffness_matrix)
normalization_constant = (np.ones((N,1)).T @ A @ np.ones((N,1)))[0][0] # Heuristic: normalize by max energy (all dof with sensors), which is always easy to compute.
print(f"Normalization constant: {normalization_constant}")
A /= normalization_constant

# Generate constraint terms
constrain_terms, constrain_offset = GenerateConstrainTerms(N, N_S) 

# solve the QUBO
df = solve_qubo(A - ALPHA * constrain_terms)
df["f_obj + constraint offset"] = df.f_obj - ALPHA * constrain_offset
print(df.head(10))
print(df.tail(10))

# Generate the DF for the normalized_performance score
df_p = df.copy()
df_p["fulfill"] = df_p["Candidate Solutions"].apply(lambda x: sum(x)==N_S)
df_p = df_p[df_p.fulfill]
# get max and min for feasible 
best = df_p[df_p.fulfill == 1].f_obj.max().item()
worst = df_p[df_p.fulfill == 1].f_obj.min().item()
df_p["int_representation"] = df_p["Candidate Solutions"].apply(lambda x: from_binary_to_int(np.asarray(x)))
df_p["normalized_performance"] = df_p.f_obj.apply(lambda x: normalized_performance(x, best, worst))
print(df_p)

# Generate Hamiltonians, -1 indicate that we are maximizing
H_c, H_c_matrix = generate_cost_hamiltonian(-1 * (A - ALPHA * constrain_terms), N)
H_m = qaoa.x_mixer(range(N))
H_m_xy = qaoa.xy_mixer(GenerateGraph(N))

# Generate QAOA circuit
def qaoa_layer_trad(gamma, beta):
    qaoa.cost_layer(gamma, H_c)
    qaoa.mixer_layer(beta, H_m)
    
def qaoa_layer_xy(gamma, beta):
    qaoa.cost_layer(gamma, H_c)
    qaoa.mixer_layer(beta, H_m_xy)

# Function for trad
@qml.qnode(qml.device("lightning.gpu", wires=len(range(N))))
def qaoa_fn_trad(params, **kwargs):
    for w in range(N):
        qml.Hadamard(wires=w)
    P = len(params[0])
    qml.layer(qaoa_layer_trad, P, params[0], params[1])
    return qml.probs()

# function for XY
dicke_circuit = qml.from_qiskit(dicke_state(N, N_S))
@qml.qnode(qml.device("lightning.gpu", wires=len(range(N))))
def qaoa_fn_xy(params, **kwargs):
    dicke_circuit(wires=list(range(N))[::-1])
    P = len(params[0])
    qml.layer(qaoa_layer_xy, P, params[0], params[1])
    return qml.probs()

# Define function
if MIXER == "xy":
    qaoa_fn = qaoa_fn_xy
elif MIXER == "trad":
    qaoa_fn = qaoa_fn_trad
else:
    raise("MIXER IS NOT DEFINED!")


# Run heatmap cycle
heatmap = np.zeros((GRANULARITY, GRANULARITY, len(CIRCUIT_DEPTHS)))
for k, P in enumerate(CIRCUIT_DEPTHS):
    print(f"\n\nHeatmap with p = {CIRCUIT_DEPTHS[k]}")
    pbar = tqdm(total=GRANULARITY**2)
    for i, MM_beta in enumerate(np.linspace(0, SUP, GRANULARITY)):
        for j, MM_gamma in enumerate(np.linspace(0, SUP, GRANULARITY)):
            pbar.update(1)
            gamma = list(MM_gamma * np.asarray([(2*m-1)/(2*P) for m in range(1,P+1)]))
            beta = list(MM_beta * np.asarray([1 - (2*m-1)/(2*P) for m in range(1,P+1)]))
            point = np.array([gamma, beta])
            heatmap[i,j,k] = compute_score2(qaoa_fn(point), df_p)
    pbar.close()
            
            
file_name = f"heatmap_{STRUCTURE}_{MIXER}"            
with open(f'results/{file_name}.pickle', 'wb') as handle:
    pickle.dump({"heatmap": heatmap,
                 "circuit_depths": CIRCUIT_DEPTHS,
                 "SUP": SUP,
                 "GRANULARITY": GRANULARITY,
                 "df_exact_solution": df}, handle, protocol=pickle.HIGHEST_PROTOCOL)       