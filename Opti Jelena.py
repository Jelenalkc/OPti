import pandas as pd
import numpy as np
from itertools import product

def read_excel(file_path):
    """Read the cost, supply, and demand data from an Excel file."""
    data = pd.read_excel(file_path, header=None)
    cost_matrix = data.iloc[:3, :3].values
    supply_values = data.iloc[:3, 3].values
    demand_values = data.iloc[3, :3].values
    return cost_matrix, supply_values, demand_values

def northwest_corner_method(cost_matrix, supply_values, demand_values):
    """Apply the Northwest Corner Method to find an initial allocation."""
    rows, cols = cost_matrix.shape
    allocation_matrix = np.zeros((rows, cols))
    row, col = 0, 0

    while row < rows and col < cols:
        allocation = min(supply_values[row], demand_values[col])
        allocation_matrix[row, col] = allocation
        supply_values[row] -= allocation
        demand_values[col] -= allocation

        if supply_values[row] == 0:
            row += 1
        elif demand_values[col] == 0:
            col += 1

    return allocation_matrix

def minimum_cost_method(cost_matrix, supply_values, demand_values):
    """Apply the Minimum Cost Method to find an initial allocation."""
    rows, cols = cost_matrix.shape
    allocation_matrix = np.zeros((rows, cols))
    cost_indices = sorted(product(range(rows), range(cols)), key=lambda x: cost_matrix[x])

    for row, col in cost_indices:
        if supply_values[row] == 0 or demand_values[col] == 0:
            continue
        allocation = min(supply_values[row], demand_values[col])
        allocation_matrix[row, col] = allocation
        supply_values[row] -= allocation
        demand_values[col] -= allocation

    return allocation_matrix

def vogels_approximation_method(cost_matrix, supply_values, demand_values):
    """Apply Vogel's Approximation Method to find an initial allocation."""
    rows, cols = cost_matrix.shape
    allocation_matrix = np.zeros((rows, cols))
    remaining_supply = supply_values.copy()
    remaining_demand = demand_values.copy()
    available_rows = list(range(rows))
    available_cols = list(range(cols))

    while available_rows and available_cols:
        penalties = []

        for row in available_rows:
            row_costs = [cost_matrix[row][col] for col in available_cols]
            if len(row_costs) >= 2:
                penalties.append((sorted(row_costs)[1] - sorted(row_costs)[0], row, 'row'))
            elif row_costs:
                penalties.append((row_costs[0], row, 'row'))

        for col in available_cols:
            col_costs = [cost_matrix[row][col] for row in available_rows]
            if len(col_costs) >= 2:
                penalties.append((sorted(col_costs)[1] - sorted(col_costs)[0], col, 'col'))
            elif col_costs:
                penalties.append((col_costs[0], col, 'col'))

        if not penalties:
            break

        _, index, penalty_type = max(penalties, key=lambda x: x[0])

        if penalty_type == 'row':
            row = index
            col = min(available_cols, key=lambda x: cost_matrix[row][x])
        else:
            col = index
            row = min(available_rows, key=lambda x: cost_matrix[x][col])

        allocation = min(remaining_supply[row], remaining_demand[col])
        allocation_matrix[row, col] = allocation
        remaining_supply[row] -= allocation
        remaining_demand[col] -= allocation

        if remaining_supply[row] <= 1e-10:
            available_rows.remove(row)
        if remaining_demand[col] <= 1e-10:
            available_cols.remove(col)

    return allocation_matrix

def calculate_total_cost(allocation_matrix, cost_matrix):
    """Calculate the total cost for a given allocation."""
    return np.sum(allocation_matrix * cost_matrix)

def main():
    # Replace with your actual file path
    file_path = "C:\\Users\\jelen\\OneDrive\\Documents\\UVic\\Classeur Opti.xlsx"
    cost_matrix, supply_values, demand_values = read_excel(file_path)

    print("Cost Matrix:")
    print(cost_matrix)
    print("Supply Values:")
    print(supply_values)
    print("Demand Values:")
    print(demand_values)

    print("\nNorthwest Corner Method Allocation:")
    northwest_allocation = northwest_corner_method(cost_matrix, supply_values.copy(), demand_values.copy())
    print(northwest_allocation)
    print(f"Total Cost: {calculate_total_cost(northwest_allocation, cost_matrix)}")

    print("\nMinimum Cost Method Allocation:")
    min_cost_allocation = minimum_cost_method(cost_matrix, supply_values.copy(), demand_values.copy())
    print(min_cost_allocation)
    print(f"Total Cost: {calculate_total_cost(min_cost_allocation, cost_matrix)}")

    print("\nVogel's Approximation Method Allocation:")
    vogels_allocation = vogels_approximation_method(cost_matrix, supply_values.copy(), demand_values.copy())
    print(vogels_allocation)
    print(f"Total Cost: {calculate_total_cost(vogels_allocation, cost_matrix)}")

if __name__ == "__main__":
    main()
