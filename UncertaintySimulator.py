#
# (2024/09) author: Zhengshu Zhou (shu@ertl.jp)
#
# Method: Uncertainty simulator
# Input:  Mobility-as-a-Service system function parameters (csv file)
# Output: Key functions for the MaaS system
#

import pandas as pd
import random

# Define a function for simulating uncertainty
def simulate_uncertainty(agent, method, environment):
    agent_uncertainty = 0
    method_uncertainty = 0
    environment_uncertainty = 0

    # Uncertainty of the execution agent
    if agent == 'human':
        agent_uncertainty = random.uniform(1, 8)
    elif agent == 'organizational':
        agent_uncertainty = random.uniform(1, 4)
    elif agent == 'technical':
        agent_uncertainty = random.uniform(1, 2)
    else:
        raise ValueError(f"Unknown execution agent type: {agent}")

    # Uncertainty of the execution method 
    if method == 'laissez-faire':
        method_uncertainty = random.uniform(1, 8)
    elif method == 'monitoring':
        method_uncertainty = random.uniform(1, 4)
    elif method == 'controlling':
        method_uncertainty = random.uniform(1, 2)
    else:
        raise ValueError(f"Unknown execution method type: {method}")

    # Execution environment uncertainty
    if environment == 'harsh':
        environment_uncertainty = random.uniform(1, 8)
    elif environment == 'normal':
        environment_uncertainty = random.uniform(1, 4)
    elif environment == 'gentle':
        environment_uncertainty = random.uniform(1, 2)
    else:
        raise ValueError(f"Unknown execution environment type: {environment}")

    return agent_uncertainty, method_uncertainty, environment_uncertainty

# Simulate the timing impact based on the type of executing agent
def simulate_time_impact(agent, time_columns, func):
    if agent == 'human':
        prob = random.random()
        if prob <= 0.5:
            time_impact = func[time_columns[1]]  # 'on time'
        elif prob <= 0.7:
            time_impact = func[time_columns[0]]  # 'too early'
        elif prob <= 0.9:
            time_impact = func[time_columns[2]]  # 'too late'
        else:
            time_impact = func[time_columns[3]]  # 'omission'
    elif agent == 'organizational':
        prob = random.random()
        if prob <= 0.8:
            time_impact = func[time_columns[1]]  # 'on time'
        elif prob <= 0.85:
            time_impact = func[time_columns[0]]  # 'too early'
        elif prob <= 0.95:
            time_impact = func[time_columns[2]]  # 'too late'
        else:
            time_impact = func[time_columns[3]]  # 'omission'
    elif agent == 'technical':
        prob = random.random()
        if prob <= 0.9:
            time_impact = func[time_columns[1]]  # 'on time'
        elif prob <= 0.94:
            time_impact = func[time_columns[0]]  # 'too early'
        elif prob <= 0.98:
            time_impact = func[time_columns[2]]  # 'too late'
        else:
            time_impact = func[time_columns[3]]  # 'omission'
    else:
        raise ValueError(f"Unknown execution agent type: {agent}")
    
    return time_impact

# Simulate the impact of precision based on the type of execution agent
def simulate_precision_impact(agent, precision_columns, func):
    if agent == 'human':
        prob = random.random()
        if prob <= 0.3:
            precision_impact = func[precision_columns[0]]  # 'precise'
        elif prob <= 0.8:
            precision_impact = func[precision_columns[1]]  # 'acceptable'
        else:
            precision_impact = func[precision_columns[2]]  # 'imprecise'
    elif agent == 'organizational':
        prob = random.random()
        if prob <= 0.6:
            precision_impact = func[precision_columns[0]]  # 'precise'
        elif prob <= 0.9:
            precision_impact = func[precision_columns[1]]  # 'acceptable'
        else:
            precision_impact = func[precision_columns[2]]  # 'imprecise'
    elif agent == 'technical':
        prob = random.random()
        if prob <= 0.8:
            precision_impact = func[precision_columns[0]]  # 'precise'
        elif prob <= 0.95:
            precision_impact = func[precision_columns[1]]  # 'acceptable'
        else:
            precision_impact = func[precision_columns[2]]  # 'imprecise'
    else:
        raise ValueError(f"Unknown execution agent type: {agent}")
    
    return precision_impact

# Function for calculating functional uncertainty
def calculate_uncertainty(function_id, df, calculated_uncertainty):
    func = df[df['ID'] == function_id].iloc[0]
    
    agent = func['agent']
    method = func['method']
    environment = func['environment']

    agent_uncertainty, method_uncertainty, environment_uncertainty = simulate_uncertainty(agent, method, environment)

    time_columns = [
        'upstream_function_1_too_early', 'upstream_function_1_on_time',
        'upstream_function_1_too_late', 'upstream_function_1_omission'
    ]
    precision_columns = [
        'upstream_function_1_precise', 'upstream_function_1_acceptable', 'upstream_function_1_imprecise'
    ]

    # Calculate the timing impact using the simulation method
    time_impact = simulate_time_impact(agent, time_columns, func)

    # Calculate the precision impact using the simulation method
    precision_impact = simulate_precision_impact(agent, precision_columns, func)

    # intrinsic_uncertainty = agent_uncertainty + method_uncertainty + environment_uncertainty + time_impact + precision_impact
    intrinsic_uncertainty = agent_uncertainty + method_uncertainty + environment_uncertainty

    upstream_uncertainty = 1
    cumulative_upstream_impact = 1
    upstream_details = []

    for i in range(1, 5):
        upstream_function = func.get(f'upstream_function_{i}', None)
        if pd.notna(upstream_function):
            if upstream_function not in calculated_uncertainty:
                calculated_uncertainty[upstream_function] = calculate_uncertainty(upstream_function, df, calculated_uncertainty)
            upstream_function_uncertainty = calculated_uncertainty[upstream_function]['intrinsic']
            upstream_time_impact = simulate_time_impact(func['agent'], time_columns, func)
            upstream_precision_impact = simulate_precision_impact(func['agent'], precision_columns, func)

            cumulative_upstream_impact *= upstream_time_impact * upstream_precision_impact

            print(f"Upstream function {upstream_function} of function {function_id}:")
            print(f"  Intrinsic uncertainty: {upstream_function_uncertainty}")
            print(f"  Timing impact: {upstream_time_impact}")
            print(f"  Precision impact: {upstream_precision_impact}")

            upstream_uncertainty += upstream_function_uncertainty

            upstream_details.append({
                "function_id": upstream_function,
                "intrinsic_uncertainty": upstream_function_uncertainty,
                "time_impact": upstream_time_impact,
                "precision_impact": upstream_precision_impact
            })

    total_uncertainty = cumulative_upstream_impact * intrinsic_uncertainty + upstream_uncertainty

    print(f"Function {function_id} intrinsic uncertainty details:")
    print(f"  Agent uncertainty: {agent_uncertainty}")
    print(f"  Method uncertainty: {method_uncertainty}")
    print(f"  Environment uncertainty: {environment_uncertainty}")
    print(f"Total intrinsic uncertainty of function {function_id}: {intrinsic_uncertainty}")
    print(f"Cumulative upstream timing and precision impact of function {function_id}: {cumulative_upstream_impact}")
    print(f"Total uncertainty of direct upstream functions of function {function_id}: {upstream_uncertainty}")
    print(f"Total uncertainty of function {function_id}: {total_uncertainty}\n")

    return {'intrinsic': intrinsic_uncertainty, 'total': total_uncertainty, 'cumulative_upstream_impact': cumulative_upstream_impact, 'upstream_details': upstream_details}

df = pd.read_csv("sim.csv")

calculated_uncertainty = {}

function_top_four_count = {func_id: 0 for func_id in df['ID']}

for _ in range(1000):
    total_uncertainties = {}

    for function_id in df['ID']:
        uncertainty = calculate_uncertainty(function_id, df, calculated_uncertainty)
        total_uncertainties[function_id] = uncertainty['total']

    sorted_functions = sorted(total_uncertainties.items(), key=lambda x: x[1], reverse=True)

    for i in range(5):
        function_top_four_count[sorted_functions[i][0]] += 1

for function_id, count in function_top_four_count.items():
    print(f"Function {function_id} appears in top 5 times: {count}")
