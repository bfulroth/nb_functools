"""Module containing useful functions for documenting Biophysics experiments in Jupyter Notebook"""

from glob import glob

import math
import numpy as np
import os
import pandas as pd

def agg_spr_single_traces_raw_data(files_to_agg):

    """
    This method takes an ordered list of text file paths where each file is a single Biacore 8k sensogrgram with time X values and RU Y values.
    In addition the method combines all single traces into one DataFrame.  This is really helpful for dose response data.

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


def calc_conc_oligo(moles, conc_M):
    """
    Method that calculates the number of microliters of solvent to add to a lyophilized oligo tube to reach the desired concentration.

    :param moles: Moles of oligo.
    :param conc_M: Desired molarity of reconstituted oligo.
    :returns Solvent amount in microliters to reach desired concentration.

    """

    solvent_to_add_ul = (moles/conc_M)*1000000

    return solvent_to_add_ul


def calc_conc_protein_mg_ml_from_Molar(molar_conc, MW_protein):
    """
    Function that calculates protein concentration in units of mg/mL from molarity

    :param molar_conc: Concentration of the protein in Molarity.
    :param MW_protein: The molecular weight of the protein in daltons.
    :returns The concentration of the protein in mg/mL

    """

    return molar_conc * MW_protein


def calc_conc_protein_uM_from_mg_ml(mg_ml, MW_protein):
    """
    Function that calculates the protein concentration in uM from mg/mL

    :param mg_ml: Protein concentration in mg/mL
    :param MW_protein: The molecular weight of the protein in daltons.
    :returns The concentration of the protein in uM.

    """

    return (mg_ml/MW_protein) * 1000000


def calc_m1v1_m2v2(m1, v1, m2, v2):
    """
    Method that calculates the classic molarity 1 * volume 1 = molarity 2 * volume 2

    :param m1: Stock concentration in Molarity
    :type m1: float
    :param v1:Volume need from Stock m1
    :type v1: Consistent units (uL, mL, etc.)
    :param m2: Desired concentration of final solution in Molarity.
    :type m2: float
    :param v2: Desired volume of final solution. (Units consistent with v1)
    :type v2: float
    :return: Unknown parameter (m1, v1, m2, or v2).
    :rtype: float

    """

    if v1 is None:
        v1 = round((m2 * v2)/ m1, 3)
        return v1

    elif m1 is None:
        m1 = round((m2 * v2) / v1, 3)
        return m1

    elif m2 is None:
        m2 = round((m1 * v1) / v2, 3)
        return m2

    else:
        v2 = round((m1 * v1) / m2)
        return v2


def calc_mRNA_prep_for_immob(rna_name, ext_coef, A260_1_100_dil, final_conc_RNA_nM, final_vol_needed_RNA_uL):
    """"
    Calculates the amounts needed to create mRNA final working samples for SPR immobilization.

    :param rna_name: Name of the RNA Sequence of interest.
    :param ext_coef: RNA oligo extension coefficient.
    :param A260_1_100_dil: Nanodrop A260 reading using RNA mode on a 1:100 dilution from stock.
    :param final_conc_RNA_nM: Final desired concentration of RNA in nanomolar.
    :param final_vol_needed_RNA_uL: Final volume for an RNA sample that is needed for SPR immobilization (Units uL)
    :returns Dictionary specifying the recipe for how to reconstitute the RNA oligo for SPR immobilization.

    Note: When preparing many RNA oligos for immobilization, this method can be used in a loop to create one dictionary
    for each RNA oligo and add it to a list.  Next, the list of Dictionaries can be made into a table using
    pd.DataFrame()

    """

    # extension coefficient
    ext_coef_RNA = ext_coef

    # A260 for RNA stocks diluted into RNAase free H20
    A260_1_100_dil_RNA = A260_1_100_dil

    # Concentration of 1:100 diluted stocks
    conc_1_100_RNA = ((A260_1_100_dil_RNA / ext_coef_RNA) * 1000000) * 1000

    # Final concentration to immobilize
    final_conc_RNA_nM = final_conc_RNA_nM

    # Volume needed for each RNA
    final_vol_needed_RNA_uL = final_vol_needed_RNA_uL

    # Calculated Concentration of RNA diluted 1:10 in buffer
    conc_1_10_RNA = conc_1_100_RNA / 10

    stock_RNA_needed_uL = (final_vol_needed_RNA_uL * final_conc_RNA_nM) / conc_1_10_RNA

    return {'RNA Sequence Name': rna_name,
            'Ext. Coeff': ext_coef_RNA,
            '[RNA] 1:100': round(conc_1_100_RNA, 2),
            '[RNA] 1:10': round(conc_1_10_RNA, 2),
            'Final Vol. Need (uL)': final_vol_needed_RNA_uL,
            'Buffer (uL)': round(final_vol_needed_RNA_uL - stock_RNA_needed_uL, 2),
            'Stock 1:10 RNA for Immob (uL)': round(stock_RNA_needed_uL, 2)
            }

def calc_recon_from_pow(fw, mg, mM):
    """
    Method that calculates the preparation of the top concentration of compound tested.

    :param fw:Formula weight of the compound to reconstitute.
    :type fw: float
    :param final_vol_uL: Final desired volume for the reconstituted sample.
    :type final_vol_uL: float
    :param final_conc_mM: Final desired concentration in mM for the reconstituted sample.
    :type final_conc_mM: float
    :return: Amount of solvent to add to reach the desired concentration.
    :rtype: float
    """

    return round(mg/1000/fw/(mM/1000)*1000*1000, 2)


def calc_recon_from_pow_return_mg(fw, final_vol_mL, final_conc_mM):
    """
    Method that calculates the amount of solvent to add to a chemical in powder form to achieve the desired concentration in mM

    :param fw: formula weight of the compound.
    :param final_vol_mL: final volume of the reconstituted stock in mL.
    :param final_conc_mM: Final desired concentration of reconstituted solution in mM.
    :returns Amount of solvent to add in mL to reach desired concentration.
    """

    return (fw / (1000 ** 2) * final_vol_mL * 1000) * final_conc_mM


def cmpd_dilutions(compound_id, stock_cmpd_mM=10.0, dmso_conc_percent=1.0, top_conc_uM=50.0, vol_uL=200.0):
    """
    Method that calculates the preparation of the top concentration of compound tested.

    :param compound_id: Compound ID
    :type str
    :param stock_cmpd_mM: Concentration of the stock in mM.
    :type float
    :param dmso_conc_percent: Final percent of DMSO in the assay.
    :type: float
    :param top_conc_uM: Top concentration to prepare the compound for the assay.
    :type float
    :param vol_uL: Volume to prepare for the assay (Units uL)
    :type vol_uL: float
    :return: Dictionary specifying the recipe for how to prepare a compound for testing in assay buffer.
    :rtype: dict

    Note: When preparing many compounds for testing, this method can be used in a loop to create one dictionary
    for each compound and add it to a list.  Next, the list of Dictionaries can be made into a table using
    pd.DataFrame()

    Return:

    return {'Compound ID': compound_id,
            'Top Conc. (uM)': top_conc_uM,
            'Top volume (uL)': vol_uL,
            'Stock [Cmpd.] (mM)': stock_cmpd_mM,
            'Buffer to Add (uL)': (vol_uL - (top_conc_uM * vol_uL)/(stock_cmpd_mM * 1000)) - dmso_to_add_uL,
            'DMSO to Add (uL)': dmso_to_add_uL,
            'Stock Cmpd to Add (uL)': stock_cmpd_to_add_uL}
    """
    stock_cmpd_to_add_uL = (top_conc_uM * vol_uL)/(stock_cmpd_mM * 1000)
    dmso_to_add_uL = ((vol_uL * (dmso_conc_percent / 100)) - stock_cmpd_to_add_uL)

    return {'Compound ID': compound_id,
            'Top Conc. (uM)': top_conc_uM,
            'Top volume (uL)': vol_uL,
            'Stock [Cmpd.] (mM)': stock_cmpd_mM,
            'Buffer to Add (uL)': (vol_uL - (top_conc_uM * vol_uL)/(stock_cmpd_mM * 1000)) - dmso_to_add_uL,
            'DMSO to Add (uL)': dmso_to_add_uL,
            'Stock Cmpd to Add (uL)': stock_cmpd_to_add_uL}


def dilution_scheme(top, num_pts, fold=2, sort=False):
    """
    Creates a concentration dilution scheme based on the top concentration, number of points, and the dilute fold between each point.

    param: top: Top concentration tested.
    param: num_pts: Number of points in the scheme.
    param: fold: Fold dilution between each point.
    param: sort: Option to sort the list in ascending order.  Default is descending.
    :returns Sorted list of concentrations.
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


def files_to_process(folder_path, file_ext='txt'):

    """
    Method that returns a sorted list of text files to process.

    Note: In order for the sorting to work each file needs to be named with the order appended to the end as follows FILE_NAME_1, FILE_NAME_2

    param: folder_pather: String path of the file folder where the text files to process exist.
    :returns: Sorted list of files to process. Tuple (file path, base file name, order of file; used for sorting)

    """

    ls_files = glob(folder_path + '/*.' + file_ext)

    file_base_name = []
    file_order = []

    for file in ls_files:
        file_name = os.path.basename(file)
        file_name = file_name.replace('.' + file_ext, '')
        file_base_name.append(file_name)

        order = int(file_name.split('_')[-1])
        file_order.append(order)

    files_to_process = list(zip(ls_files, file_base_name, file_order))

    sorted_files_to_process = sorted(files_to_process, key=lambda x: x[2])

    print('Processing the following files in order...\n')

    for file in sorted_files_to_process:
        print(file[1])

    return sorted_files_to_process


def find_nearest(array, target):
    """"
    Finds the index of the closest value to a target value in an array.

    :param array: Array to search for the nearest value to the target value.
    :param target: The target value to find the nearest value.
    :returns The index of the nearest value to the target value.

    """

    idx = np.searchsorted(array, target, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(target - array[idx-1]) < math.fabs(target - array[idx])):
        return idx-1
    else:
        return idx


def get_spr_sol_corr_tbl(conc_in_assay=1, num_pts=5, step=0.25, tube_vol=2000):
    """
    Method that generates a DMSO solvent correction table for SPR.

    param: conc_in_assay: Concentration of DMSO in the assay.
    param: num_pts: Number of DMSO concentrations to use for the standard curve.
    param: step: The percent DMSO difference between each concentration point in the calibration curve.
    param: tube_vol: The volume to prepare for each DMSO concentration point.
    :returns Recipe in a Dataframe that specifies how to prepare solvent correction samples.
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


def get_steady_state_pts(df_all_data, name_x_series='Time', start_sec=60, stop_sec=75):
    """
    Method that does the work of determining the steady-state RU value for each sensorgram in a dose response.

    :param df_all_data: DataFrame containing the sensorgram data points for each compound concentration tested.
    :type df_all_data: DataFrame
    :param start_sec: The starting time in seconds of where to find the steady state point.
    :type start_sec: int
    :param stop_sec: The ending time in seconds of where to find the steady state point.
    :type stop_sec: int
    :return: list of steady state points
    :rtype: list
    """

    # Find the index of nearest time point for start time
    x = df_all_data[name_x_series].values
    idx_low = find_nearest(array=x, target=start_sec)
    idx_low = idx_low

    # Find the index of nearest time point for end time
    idx_high = find_nearest(array=x, target=stop_sec)
    idx_high = idx_high

    Y = df_all_data.drop(name_x_series, axis=1)

    ls_ss_pts = []

    for col in Y:

        active_y = Y[col].values
        steady_range = active_y[idx_low:idx_high]

        ls_ss_pts.append(np.median(steady_range))

    return ls_ss_pts


def itc_calc_pre_rinse(num_titration_samples, num_titration_ctrls, dmso_prec=1):
    """
    Method that calculates the amount of pre-rinse solution to make for an ITC experiment.

    :param num_titration_samples:
    :type num_titration_samples:
    :param num_titration_ctrls:
    :type num_titration_ctrls:
    :param dmso_prec:
    :type dmso_prec:
    :return:
    :rtype:
    """
    pre_rinse_samples_ul = (400 * 1) * num_titration_samples
    pre_rinse_ctrl_ul = (400 * 2) * num_titration_ctrls

    total_pre_rinse_ul = pre_rinse_samples_ul + pre_rinse_ctrl_ul + 0.2 * (pre_rinse_samples_ul + pre_rinse_ctrl_ul)

    # Calculate final amounts
    dmso_pre_rinse_ul = total_pre_rinse_ul * (dmso_prec / 100)
    pre_rinse_buffer_ul = total_pre_rinse_ul - dmso_pre_rinse_ul

    dict_pre_rinse = {'Buffer (uL)': pre_rinse_buffer_ul, 'DMSO (uL)': dmso_pre_rinse_ul}

    return dict_pre_rinse


def itc_syringe_samples(sample_id, stock_conc_um=10000, test_conc_um=250, vol_ul=250, dmso_per=1, in_dmso=True):

    """
    Method that calculates the amounts of each reagent to added for a sample that is titrated by ITC.

    :param sample_id:
    :type sample_id:
    :param stock_conc_um:
    :type stock_conc_um:
    :param test_conc_um:
    :type test_conc_um:
    :param vol_ul:
    :type vol_ul:
    :param dmso_per:
    :type dmso_per:
    :param in_dmso:
    :type in_dmso: bool
    :return:
    :rtype: dict
    """

    # Initialize a dictionary to store all calculations
    dict_itc_syr = {'Sample ID': sample_id,
                    'Stock (uM)': stock_conc_um,
                    'Test Conc. (uM)': test_conc_um,
                    'Total vol. (uL)': vol_ul,
                    '% DMSO': dmso_per}

    # Calculate total DMSO needed
    total_dmso_ul = vol_ul * (dmso_per / 100)
    dict_itc_syr['Sample to Add (uL)'] = (vol_ul * test_conc_um) / (stock_conc_um)

    if in_dmso:
        dict_itc_syr['DMSO to Add (uL)'] = total_dmso_ul - dict_itc_syr['Sample to Add (uL)']
        dict_itc_syr['Buffer to Add (uL)'] = vol_ul - sum(
            [dict_itc_syr['Sample to Add (uL)'], dict_itc_syr['DMSO to Add (uL)']])

    if not in_dmso:
        dict_itc_syr['DMSO to Add (uL)'] = total_dmso_ul
        dict_itc_syr['Buffer to Add (uL)'] = vol_ul - sum(
            [dict_itc_syr['Sample to Add (uL)'], dict_itc_syr['DMSO to Add (uL)']])

    return dict_itc_syr


def logistic4PL_inhib_vs_resp_var_slope(x, Top, Bottom, IC50, HillSlope):
    """
    4PL lgoistic equation.

    Reference: https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_dr_stim_variable.htm

    x: x values. Typically concentration.
    Top: Top asymptote of the curve.
    Bottom: Bottom asymptote of the curve. Also refered to as the offset.
    IC50:  is the concentration of Inhibitor that gives a response half way between Bottom and Top.
    Hillslope: describes the steepness of the family of curves. A HillSlope of -1.0 is standard,
    and you should consider constraining the Hill Slope to a constant value of -1.0.
    A Hill slope more negative than -1 (say -2) is steeper.

    Example of use:

    fit = leastsq(residuals, [0, 0, 0, 0], maxfev=4000, args=(conc_ctrl, data.values, logistic4PL_inhib_vs_resp_var_slope))

    To access optimized parameters:

    IC50 = 10**fit[0][2]

    Note: 10** reverses the log operation.


    """

    y_hat = Bottom + (Top - Bottom) / (1 + 10 ** ((IC50 - x) * HillSlope))

    return y_hat


def peval(x, p, logistic4LP_funct):
    """
    Evaluated value at x with current parameters.

    x: x values. Typically concentration.
    p: vector of optimized parameters determined using a scipy optimization function.  Usually minimizing the SSR.
    logistic4LP_funct: Function being used to generated the predicted y values from the best fit.
    return: results from logistic4LP_funct which are the predicted y values from the optimized parameters.

    """

    Top, Bottom, IC50, HillSlope = p

    return logistic4LP_funct(x, Top, Bottom, IC50, HillSlope)


def residuals(p, x, y, logistic4LP_funct):
    """
    Deviations of data from fitted 4PL curve.

    p: vector of parameters
    x: x values. Typically concentration.
    y: Actual response values.
    logistic4LP_funct: Function being used to generated the predicted y values from the best fit.

    return: The difference from your actual y values and the predicted y values point by point.

    """

    Top, Bottom, IC50, HillSlope = p

    err = y - logistic4LP_funct(x, Top, Bottom, IC50, HillSlope)

    return err


def Rmax_theoretical(mw_ligand, ru_immob_ligand, mw_analyte, stoichiometry=1):
    """
    This method calculates the theoretical Rmax response for an analyte binding to an immobilized ligand in and SPR experiment.

    :param mw_ligand: Molecular weight of the ligand in daltons
    :type mw_ligand: float
    :param ru_immob_ligand: Measured stable relative response (RU) units of the immobilized ligand.
    :type ru_immob_ligand: float
    :param mw_analyte: Molecular weight of the analyte in daltons.
    :type mw_analyte: float
    :param stoichiometry: Number of molecules of analyte expected to bind to a ligand (e.g. 1:1, 2:1, etc.)
    :type stoichiometry: int
    :return: The theoretical maximum signal (RU) expected from a analyte/ ligand binding event.
    :rtype: float
    """

    return round((ru_immob_ligand * (mw_analyte/mw_ligand)) * stoichiometry, 3)


def gen_conc_series(num_pts, top, fold_dil=2, ascending=True):
    """
    Function that generates a concentration series from the top concentration and a fold dilution parameter.

    :param num_pts: Total number of points that the dilution series spans.
    :type num_pts: int
    :param top: The top concentration that the compound titration starts at
    :type top: float
    :param fold_dil: The fold of dilution between each point in the titration curve.
    :type fold_dil: float
    :param ascending: Option that indicates if the dilution series should be sorted in ascending order.
    :type ascending: bool
    :return: Calculated dilution series list.
    :rtype: list
    """

    curr_top = top

    ls_conc = []
    ls_conc.append(top)

    for i in range(1, num_pts):
        curr_top = curr_top / fold_dil
        ls_conc.append(curr_top)

    if ascending:
        ls_conc.sort()

    return ls_conc


def reject_outliers(data, fold_div=3):
    """
    Method that removes outliers from an array of values.

    - Get the deviation for each value by subtracting each value from the median value.
    - Get the median of the deviations.
    - Get the fold deviation for each element as it relates to the median of deviations.
    - Remove those elements with folds of deviation greater than the fold `fold_div` parameter.

    param: data: numpy array that you would like to check for outliers.
    param: fold_div: The threshold for the fold deviation from the median diviation.
    returns: array with outliers removed.

    """

    # Store the size of data.
    data_size = len(data)

    dev_median = np.abs(data - np.median(data))
    mdev = np.median(dev_median)
    fold_mdev = dev_median / mdev if mdev else np.zeros(data_size)

    keep = []
    keep_ct = 0

    for i, val in enumerate(data):

        if fold_mdev[i] < fold_div:
            keep.append(val)
            keep_ct += 1

        # Keep the original data array length by appending nan if the value is an outlier.
        else:
            keep.append(np.nan)

    # Warning if > 50% of the data points are knocked out.
    if (keep_ct / data_size) < 0.5:
        print("Warning! Over 50% of the points were knocked out!")

    return np.array(keep)
