# %%

import pyspa

list_total_intensities = []

for i in range(0, 115):
    sc = pyspa.get_spa(
        target_ID = i,
        max_stage = 10,
        a_matrix ='A_matrix_template.csv',
        infosheet='Infosheet_template.csv',
        thresholds='Thresholds_template_perc.csv',
        thresholds_as_percentages=True,
        zero_indexing=True,
    )
    list_total_intensities.append(sc.root_node.total_intensities)


# %%

import numpy as np
import pandas as pd

df_infosheet = pd.read_csv(
    filepath_or_buffer='infosheet_template.csv',
    header=0,    
)
sector_names: pd.DataFrame = df_infosheet[['Sector number', 'Name']]
df_B_matrix: pd.DataFrame = df_infosheet['DR_GHG_emissions_(kgCO2e)']
B_matrix: np.ndarray = df_B_matrix.to_numpy()

df_A_matrix = pd.read_csv(
    filepath_or_buffer='A_matrix_template.csv',
    header=0,
    index_col=None,
)
A_matrix: np.ndarray = df_A_matrix.to_numpy()
I_matrix = np.identity(A_matrix.shape[0])

def generate_final_demand_vector(
    number_of_sectors: int,
    sector_index: int,
    demand_amount: float
) -> np.ndarray:
    f_vector = np.zeros(number_of_sectors)
    f_vector[sector_index] = demand_amount
    return f_vector


# %%

#B_matrix @ np.linalg.inv(I_matrix - A_matrix) @ f_vector

# https://numpy.org/doc/2.0/reference/generated/numpy.linalg.solve.html
# https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/

list_total_int_matrix = []

for i in range(0, 115):
    f_vector = generate_final_demand_vector(
        number_of_sectors=A_matrix.shape[0],
        sector_index=i,
        demand_amount=1
    )
    list_total_int_matrix.append(B_matrix @ np.linalg.solve(I_matrix - A_matrix, f_vector))

# %%

df_total_intensities = pd.DataFrame({
    'Sector number': sector_names['Sector number'],
    'Sector name': sector_names['Name'],
    'Total intensities (SPA)': list_total_intensities,
    'Total intensities (Matrix)': list_total_int_matrix
})

df_total_intensities['Total intensities (SPA)'] = df_total_intensities['Total intensities (SPA)'].apply(lambda x: x['GHG_emissions'])

df_total_intensities['Difference (%)'] = (df_total_intensities['Total intensities (SPA)'] - df_total_intensities['Total intensities (Matrix)']) / df_total_intensities['Total intensities (Matrix)'] * 100

import matplotlib
import time

df_total_intensities.plot.bar(x='Sector name', y='Difference (%)', figsize=(20, 10))


# %%



# %%

plt.figure(figsize=(10, 5))
plt.plot(list_comp_time, list_spa_coverage, label='SPA Coverage')
plt.xlabel('Computation Time (s)')
plt.ylabel('SPA Coverage')
#plt.xscale('log')
plt.legend()
plt.title('SPA Coverage vs Computation Time')
plt.show()
