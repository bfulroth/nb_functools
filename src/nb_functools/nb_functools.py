"""Module containing useful functions for documenting Biophysics experiments in Jupyter Notebook"""

import pandas as pd
import numpy as np
from glob import glob
import math
import os


def find_nearest(array, value):
    
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
    
    
def get_steady_state_pts(df_all_data):
    
    """
    Method that does the work of determining the steady-state RU value for each sensorgram in a dose response.
    
    param: df_all_data: DataFrame containing the sensorgram data points for each compound concentration tested.
    
    """

    # Find the index of nearest time point for 60 seconds
    x = df_all_data['x'].values
    idx_low = find_nearest(array=x, value=60)
    idx_low = idx_low

    # Find the index of nearest time point for 75 seconds
    idx_high = find_nearest(array=x, value=75)
    idx_high = idx_high

    Y = df_all_data.drop('x', axis=1)

    ls_ss_pts = []

    for col in Y:

        active_y = Y[col].values
        steady_range = active_y[idx_low:idx_high]

        ls_ss_pts.append(np.median(steady_range))
    
    return ls_ss_pts


def files_to_process(folder_path):
    
    """
    Method that returns a sorted list of text files to process.
    
    Note: In order for the sorting to work each file needs to be named with the order appended to the end as follows FILE_NAME_1, FILE_NAME_2
    
    param: folder_pather: String path of the file folder where the text files to process exist.
    """

    ls_files = glob(folder_path + '/*.txt')

    file_base_name = []
    file_order = []

    for file in ls_files:
        

        file_name = os.path.basename(file)
        file_name = file_name.replace('.txt', '')
        file_base_name.append(file_name)

        order = int(file_name.split('_')[-1])
        file_order.append(order)


    files_to_process = list(zip(ls_files, file_base_name, file_order))
    
    sorted_files_to_process = sorted(files_to_process, key=lambda x: x[2])
    
    print('Processing the following files in order...\n')
    
    for file in sorted_files_to_process:
        print(file[1])
        
    return sorted_files_to_process


def agg_spr_single_traces_raw_data(files_to_agg):
    
    """
    This method takes an ordered list of text file paths where each file is a single Biacore 8k sensogrgram with time X values and RU Y values.
    In addition the method combines all of the single traces into one DataFrame.  This is really helpful for dose response data.
    
    """

    df = pd.DataFrame()
    
    for i, file_tup in enumerate(files_to_agg):
        

        if i == 0:

            df = pd.read_csv(file_tup[0], sep='\t')
            df = df.iloc[:, [0, 1]]

            df.columns = ['x', f'y_{i + 1}']

        else:

            df_temp = pd.read_csv(file_tup[0], sep='\t')
            df_temp = df_temp.iloc[:, [0, 1]]

            df_temp.columns = ['x', f'y_{i + 1}']

            df[f'y_{i + 1}'] = df_temp[f'y_{i + 1}']
            
    return df


def dilution_scheme(top, num_pts, fold=2, sort=False):
    
    """
    Creates a concentration dilution scheme based on the top concentration, number of points, and the dilute fold between each point.
    
    param: top: Top concentration tested.
    param: num_pts: Number of points in the scheme.
    param: fold: Fold dilution between each point.
    param: sort: Option to sort the list in ascending order.  Default is descending.
    """

    ls_conc = []

    for i in range(num_pts):

        if i == 0:
            ls_conc.append(top)

        else:
            top = top/fold
            ls_conc.append(top)
    
    if sort:
        ls_conc = sorted(ls_conc)
        
    return ls_conc


def get_spr_sol_corr_tbl(conc_in_assay=1, num_pts=5, step=0.25, tube_vol=2000):
    
    """
    Method that generates a DMSO solvent correction table for SPR.
    
    param: conc_in_assay: Concentration of DMSO in the assay.
    param: num_pts: Number of DMSO concentrations to use for the standard curve.
    param: step: The percent DMSO difference between each concentration point in the calibration curve.
    param: tube_vol: The volume to prepare for each DMSO concentration point.
    """
    
    min_conc = conc_in_assay
    
    # Find the minimum concentration of DMSO
    for i in range(round(num_pts/2)):
        
        min_conc = min_conc - step
    
    # Take the start and generate the entire concentration series
    ls_conc_series = []
    ls_conc_series.append(min_conc)
    
    # Create the concentration series
    for i in range(num_pts-1):
        
        ls_conc_series.append(ls_conc_series[-1] + step)
        
    # Create a list of the volume for each DMSO conc.
    ls_tube_vol = [tube_vol for i in range(num_pts)]
    
    # Amount of DMSO to add
    ls_dmso_to_add = list((np.array(ls_conc_series) / 100) * np.array(ls_tube_vol))
    
    # Create a list of buffer uL to add
    ls_buffer_to_add = list(np.array(ls_tube_vol) - np.array(ls_dmso_to_add))
    
    # Zip all lists together and create a DataFrame 
    ls_all = list(zip(ls_conc_series, ls_tube_vol, ls_buffer_to_add, ls_dmso_to_add))
    
    df_sol_corr_tbl = pd.DataFrame(ls_all, columns = ['% DMSO', 'Tube Volume (uL)', 'Buffer to Add (uL)', 'DMSO to Add (uL)'])
    
    df_sol_corr_tbl.sort_values(by='% DMSO', ascending=False, inplace=True)
    df_sol_corr_tbl.reset_index(inplace=True, drop=True)
    
    return df_sol_corr_tbl


def Rmax_theoretical(mw_ligand, ru_immob_ligand, mw_analyte, stoichiometry=1):
    """
    This method calculates the theoretical Rmax response for an analyte binding to an immobilized ligand in and SPR experiment.
    
    """

    return round((ru_immob_ligand * (mw_analyte/mw_ligand)) * stoichiometry, 3)


def calc_recon_from_pow(fw, final_vol_uL, final_conc_mM):
    
    """Method that calclates the amount of solvent to add to a chemical in powder form to achieve the desired concentration in mM"""
    
    return (fw / (1000**2) * final_vol_uL * 1000) * final_conc_mM


def cmpd_dilutions(VRT, stock_cmpd_mM=10, top_conc_uM=50, vol_uL=200):
    """
    Method that calculates the preparation of the top concentration of compound tested.
    
    """
    
    return {'VRT': VRT,
            'Top Conc. (uM)': top_conc_uM,
            'Top volume (uL)': vol_uL,
            'Stock [Cmpd.] (mM)': stock_cmpd_mM,
            'Buffer to Add (uL)': vol_uL - (top_conc_uM * vol_uL)/(stock_cmpd_mM * 1000),
            'Stock Cmpd to Add (uL)':(top_conc_uM * vol_uL)/(stock_cmpd_mM * 1000)}


def calc_m1v1_m2v2(m1, v1, m2, v2):

    v1 = round((m2 * v2)/ m1, 3)
    
    return v1




