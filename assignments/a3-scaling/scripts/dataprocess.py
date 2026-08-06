import json
from pathlib import Path
import matplotlib.pyplot as plt

data_path = Path("data/isoflops_curves.json")

with data_path.open("r", encoding="utf-8") as file:
    runs = json.load(file)

print(type(runs))
print(len(runs))
print(runs[0])

counts = {}

for run in runs:
    compute_budget = run["compute_budget"]
    counts[compute_budget] = counts.get(compute_budget, 0) + 1

for compute_budget in sorted(counts):
    print(compute_budget, counts[compute_budget])
    
best_by_budget = {}

for run in runs:
    C = run["compute_budget"]
    
    if C not in best_by_budget:
        best_by_budget[C] = run
    else:
        if run["final_loss"] < best_by_budget[C]["final_loss"]:
            best_by_budget[C] = run

for C in sorted(best_by_budget):
    best_run = best_by_budget[C]
    
    N_opt = best_run["parameters"]
    minimus_loss = best_run["final_loss"]
    D_opt = C / (6*N_opt)
    
    print(f"Compute budget: {C}, Optimal parameters: {N_opt}, Minimus loss: {minimus_loss}, Optimal depth: {D_opt}")


compute_budgets = []
optimal_parameters = []
minimus_losses = []
optimal_data = []

for C in sorted(best_by_budget):
    run = best_by_budget[C]
    N_opt = run["parameters"]
    loss = run["final_loss"]
    D_opt = C / (6*N_opt)
    
    compute_budgets.append(C)
    optimal_parameters.append(N_opt)
    minimus_losses.append(loss)
    optimal_data.append((C, N_opt, loss, D_opt))


# 图1

plt.figure(figsize=(7, 5))
plt.plot(compute_budgets, optimal_parameters, marker='o')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Compute Budget")
plt.ylabel("Optimal Parameters N_opt")
plt.title("Optimal Parameters vs Compute Budget")
plt.grid(True)
plt.tight_layout()
plt.show()

# 图2
plt.figure(figsize=(7, 5))
plt.plot(compute_budgets, optimal_data, marker='o')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Compute Budget")
plt.ylabel("Optimal Data D_opt")
plt.title("Minimum loss vs compute Budget")
plt.grid(True)
plt.tight_layout()
plt.show()

