# %%

import numpy as np
import pandas as pd
import pyspa

df_infosheet = pd.read_csv(
    filepath_or_buffer='../Infosheet_template.csv',
    header=0,    
)
sector_names: pd.DataFrame = df_infosheet[['Sector number', 'Name']]

list_total_int_spa = []

for i in range(0, sector_names.shape[0]):
    sc = pyspa.get_spa(
        target_ID=i,
        max_stage=20,
        a_matrix='../A_matrix_template.csv',
        infosheet='../Infosheet_template.csv',
        thresholds={'GHG_emissions': 0.1},
        thresholds_as_percentages=True,
        zero_indexing=True,
    )
    list_total_int_spa.append(sc.root_node.total_intensities['GHG_emissions'])


# %%

df_A_matrix = pd.read_csv(
    filepath_or_buffer='../A_matrix_template.csv',
    header=0,
    index_col=None,
)
A_matrix: np.ndarray = df_A_matrix.to_numpy()
I_matrix = np.identity(A_matrix.shape[0])

df_B_matrix: pd.DataFrame = df_infosheet['DR_GHG_emissions_(kgCO2e)']
B_matrix: np.ndarray = df_B_matrix.to_numpy()

def generate_final_demand_vector(
    number_of_sectors: int,
    sector_index: int,
    demand_amount: float
) -> np.ndarray:
    f_vector = np.zeros(number_of_sectors)
    f_vector[sector_index] = demand_amount
    return f_vector

list_total_int_matrix = []

for i in range(0, sector_names.shape[0]):
    f_vector = generate_final_demand_vector(
        number_of_sectors=A_matrix.shape[0],
        sector_index=i,
        demand_amount=1
    )
    list_total_int_matrix.append(B_matrix @ np.linalg.solve(I_matrix - A_matrix, f_vector))

# %%

df_total_intensities = pd.DataFrame({
    'Sector Number': sector_names['Sector number'],
    'Sector Name': sector_names['Name'],
    'Total Intensities (SPA)': list_total_int_spa,
    'Total Intensities (Matrix)': list_total_int_matrix
})
df_total_intensities['Difference (%)'] = ((df_total_intensities['Total Intensities (SPA)'] / df_total_intensities['Total Intensities (Matrix)']) * 100) - 100

import pickle
with open('df_total_intensities.pkl', 'wb') as f:
    pickle.dump(df_total_intensities, f)