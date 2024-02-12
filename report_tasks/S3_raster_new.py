import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
import matplotlib.ticker as ticker
import datetime
from datetime import datetime, timedelta, date
from fpdf import FPDF
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import logging
import warnings



def daily_report_S3_raster_new(df, save_path, date):

    # replace all nans with 0
    df = df.replace(np.nan, 0)

    # replace all nans with 0
    df = df.replace(np.nan, 0)

    # New columns (variables)
    df['trial_duration'] = df['TRIAL_END'] - df['TRIAL_START']
    df['sum_s_trial_duration'] = df['trial_duration'].sum()
    df['session_duration'] = (df['sum_s_trial_duration'].iloc[0])/ 60
    formatted_session_duration = "{:.2f}".format(df['session_duration'].iloc[0])
    df['center_light_duration'] = df['STATE_center_light_END'] - df['STATE_center_light_START']
    df['side_light_duration'] = df['STATE_side_light_END'] - df['STATE_side_light_START']
    df['duration_drink_delay'] = df['STATE_drink_delay_END'] - df['STATE_drink_delay_START']
    df['centre_response_latency'] = df['center_light_duration']
    df['side_response_latency'] = df['side_light_duration']
    df['duration_water_delivery'] = df['STATE_water_delivery_END'] - df['STATE_water_delivery_START']
    df['duration_timeout'] =  df['STATE_timeout_END'] - df['STATE_timeout_START']
    df['duration_wrong_side'] = df['STATE_wrong_side_END'] - df['STATE_wrong_side_START']
    df['first_response_right'] = df['Port5In_START'].str.split(',').str[0].astype(float)
    df['first_response_left'] = df['Port2In_START'].str.split(',').str[0].astype(float)


    df = df.replace(np.nan, 0)

    # List of conditions for teat well the NANs

    conditions = [
        (df.first_response_left == 0) & (df.first_response_right == 0),
        df.first_response_left == 0,
        df.first_response_right == 0,
        df.first_response_left <= df.first_response_right,
        df.first_response_left > df.first_response_right,
    ]

        # BPOD port5 ->  right
        # BPOD port3 -> (central)
        # BPOD port2 -> left
    # List of values to return

    choices = ["no_response",
               "right",
               "left",
               "left",
               "right"]
    # create a new column in the DF based on the conditions

    df["first_trial_response"] = np.select(conditions, choices)

    df["correct_outcome_bool"] = df["first_trial_response"] == df[
        "side"]  # this is for having boolean variables (true/false)
    df['true_count'] = (df['correct_outcome_bool']).value_counts()[True]
    df["correct_outcome"] = np.where(df["first_trial_response"] == df["side"], "correct",
                                     "incorrect")  # (true = correct choice, false = incorrect side)
    df["correct_outcome_int"] = np.where(df["first_trial_response"] == df["side"], 1,
                                         0)  # (1 = correct choice, 0= incorrect side)

    # Other calculations and Averaging

    df['center_response_latency_median'] = df['centre_response_latency'].median()  # median latency to first response

    df['side_response_latency_median'] = df['side_response_latency'].median()  # median latency to first response

    #misses: no reward from the sides but they can poke anyway
    max_duration_side_light = 300  # max time in side_light (for extrac missed trial)
    df['missed_reward'] = (df['side_light_duration'] == max_duration_side_light).astype(int)
    count_missed_trials = df['missed_reward'].sum()
    missed_reward_rate = (df['missed_reward'].iloc[0] / df['trials_max'].iloc[0]) * 100
    formatted_missed_reward_rate = "{:.2f}".format(missed_reward_rate)

    #Central miss: max time in center_light state
    max_duration_centre_light = 300
    df['missed_center_poke'] = (df['center_light_duration'] == max_duration_centre_light).astype(int)
    count_missed_center_trials = df['missed_center_poke'].sum()

    # omission: no RESPONSE. NO POKE. it's general
    df["omission_center_bool"] = (df['Port2In_START'] == 0) & (df['Port3In_START'] == 0) & (df['Port5In_START'] == 0)
    df["omission_int"] = df["omission_center_bool"].astype(int)
    df['omission_sum'] = df["omission_int"].sum()
    tot_center_omission = df.omission_sum.iloc[0]

    df["omission_sides_bool"] = (df['Port2In_START'] == 0) & (df['Port3In_START'] != 0) & (df['Port5In_START'] == 0)
    df["omission_int"] = df["omission_center_bool"].astype(int)
    df['omission_sum'] = df["omission_int"].sum()
    tot_center_omission = df.omission_sum.iloc[0]


    df['tot_correct_choices'] = df['correct_outcome_int'].sum()  # correct choice rate
    correct_choice_rate = (df['tot_correct_choices'].iloc[0] / df['trials_max'].iloc[0]) * 100
    formatted_correct_choice_rate = "{:.2f}".format(correct_choice_rate)

    df['right_choices'] = (df['side'] == 'right').sum()  # number of right choices
    right_choice_rate = (df['right_choices'].iloc[0] / df['trials_max'].iloc[0]) * 100
    formatted_right_choice_rate = "{:.2f}".format(right_choice_rate)

    df['left_choices'] = (df['side'] == 'left').sum()  # number of left choices
    left_choice_rate = (df['left_choices'].iloc[0] / df['trials_max'].iloc[0]) * 100
    formatted_left_choice_rate = "{:.2f}".format(left_choice_rate)

    # BPOD port5 ->  right
    # BPOD port3 -> (central)
    # BPOD port2 -> left

    columns_of_interest = ['trial', 'STATE_side_light_END', 'Port2In_START']
    columns_of_interest1 = ['trial','STATE_side_light_END', 'Port3In_START']
    columns_of_interest2 = ['trial','STATE_side_light_END', 'Port5In_START']

        # Crea un nuovo DataFrame con solo le colonne di interesse
    exploded_port2_df = df[columns_of_interest].copy()
    exploded_port3_df = df[columns_of_interest1].copy() if 'Port3In_START' in df else 0
    exploded_port5_df = df[columns_of_interest2].copy()

        # Suddividi le colonne utilizzando la virgola come delimitatore
    exploded_port2_df['Port2In_START'] = df['Port2In_START'].str.split(',')
    exploded_port3_df['Port3In_START'] = df['Port3In_START'].astype(str).str.split(',') if 'Port3In_START' in df else 0
    exploded_port5_df['Port5In_START'] = df['Port5In_START'].str.split(',') if 'Port5In_START' in df else 0

        # Esploa le colonne con liste in righe separate
    exploded_port2_df = exploded_port2_df.explode('Port2In_START')
    exploded_port3_df = exploded_port3_df.explode('Port3In_START')
    exploded_port5_df = exploded_port5_df.explode('Port5In_START')

        #explode le colonne con liste in righe separate
    exploded_port2_df = exploded_port2_df.explode('Port2In_START')
    exploded_port3_df = exploded_port3_df.explode('Port3In_START')
    exploded_port5_df = exploded_port5_df.explode('Port5In_START')

    # replace all nans with 100
    exploded_port2_df = exploded_port2_df.replace(np.nan, 190898697687982)
    exploded_port3_df = exploded_port3_df.replace(np.nan, 190898697687982)
    exploded_port5_df = exploded_port5_df.replace(np.nan, 190898697687982)

    #  'PortIn_START' in float
    exploded_port2_df['Port2In_START'] = pd.to_numeric(exploded_port2_df['Port2In_START'], errors='coerce')
    exploded_port3_df['Port3In_START'] = pd.to_numeric(exploded_port3_df['Port3In_START'], errors='coerce')
    exploded_port5_df['Port5In_START'] = pd.to_numeric(exploded_port5_df['Port5In_START'], errors='coerce')

    # BPOD port5 ->  right
    # BPOD port3 -> (central)
    # BPOD port2 -> left

    #count total pokes in each trial

    # Numero specifico da confrontare
    fake_value = 190898697687982

    # Definire una funzione per contare i pokes
    def count_pokes(value):
        return 1 if value != fake_value else 0


    exploded_port2_df['left_poke'] = exploded_port2_df['Port2In_START'].apply(count_pokes)
    exploded_port3_df['central_poke'] = exploded_port3_df['Port3In_START'].apply(count_pokes)
    exploded_port5_df['right_poke'] = exploded_port5_df['Port5In_START'].apply(count_pokes)

    #count each poke in each trial
    df['total_poke_left'] = exploded_port2_df.groupby(exploded_port2_df.index)['left_poke'].sum()
    df['total_poke_centre'] = exploded_port3_df.groupby(exploded_port3_df.index)['central_poke'].sum()
    df['total_poke_right'] = exploded_port5_df.groupby(exploded_port5_df.index)['right_poke'].sum()

    # comparison to find poke before the correct one
    exploded_port2_df['result'] = np.where(
        exploded_port2_df['Port2In_START'] <= exploded_port2_df['STATE_side_light_END'], 1, 0)
    exploded_port3_df['result'] = np.where(
        exploded_port3_df['Port3In_START'] <= exploded_port3_df['STATE_side_light_END'], 1, 0)
    exploded_port5_df['result'] = np.where(
        exploded_port5_df['Port5In_START'] <= exploded_port5_df['STATE_side_light_END'], 1, 0)

    # Creazione di nuove colonne in df per i risultati sommati raggruppati per l'indice
    df['poke_before_correct_left'] = exploded_port2_df.groupby(exploded_port2_df.index)['result'].sum()
    df['poke_before_correct_centre'] = exploded_port3_df.groupby(exploded_port3_df.index)['result'].sum()
    df['poke_before_correct_right'] = exploded_port5_df.groupby(exploded_port5_df.index)['result'].sum()

    poke_df = df[['trial','poke_before_correct_left', 'poke_before_correct_centre','poke_before_correct_right',
                'total_poke_left', 'total_poke_centre', 'total_poke_right']].copy()
    poke_df['total_poke_before_correct'] = poke_df.poke_before_correct_left + poke_df.poke_before_correct_centre + poke_df.poke_before_correct_right
    poke_df['total_poke_each_trial'] = poke_df.total_poke_left + poke_df.total_poke_centre + poke_df.total_poke_right
    poke_df['total_poke'] = poke_df['total_poke_each_trial'].sum()

    df['water_intake'] = (df['tot_correct_choices'].iloc[0]) * 0.01 #microliters each poke, but i want the result in milliliters

    poke_df['total_n_poke_left'] = poke_df['total_poke_left'].sum()
    poke_df['total_n_poke_centre'] = poke_df['total_poke_centre'].sum()
    poke_df['total_n_poke_right'] = poke_df['total_poke_right'].sum()

    # PLOT PARAMETERS

    mpl.rcParams['lines.linewidth'] = 0.7
    mpl.rcParams['lines.markersize'] = 1
    mpl.rcParams['font.size'] = 7
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    label_kwargs = {'fontsize': 9, 'fontweight': 'roman'}

    # PLOT COLORS:

    left_c = 'aquamarine'
    right_c = 'orange'
    correct_c = 'green'
    incorrect_c = 'red'
    label_kwargs = {'fontsize': 9}

    plt.figure(figsize=(8, 11))

    # Add session summary text
    session_summary = f'''
     S3 Sessions info
     Date: {df['date'].iloc[0]}, Animal ID: {df['subject'].iloc[0]}, Animal weight: {df['subject_weight'].iloc[0]},Box number: {df['box'].iloc[0]}, Trials: {df['trial'].iloc[-1]}, Session 
     duration: {formatted_session_duration} min, Center latency (median): {str(round(df.centre_response_latency.iloc[1], 2))} s, Side latency (median): {str(round(df.side_response_latency.iloc[1], 2))} s, Missed (sides): {count_missed_trials}, 
     Omission (sides): {tot_center_omission}, Missed (center): {count_missed_center_trials}, Omission (center): {tot_center_omission}, Water intake: {df['water_intake'].iloc[0]}ml, Total pokes: {poke_df['total_poke'].iloc[0]}, 
     (R: {poke_df['total_n_poke_right'].iloc[0]}, C: {poke_df['total_n_poke_centre'].iloc[0]}, L: {poke_df['total_n_poke_left'].iloc[0]})'''

    plt.figtext(0.03, 0.93, session_summary, fontsize=10)

    # Creiamo il plot con spaziatura tra le linee
    axes = plt.subplot2grid((1600, 100), (1, 1), rowspan=1800, colspan=200)

    # Impostiamo una scala logaritmica sull'asse delle x
    axes.set_xscale('log')

    # Impostiamo i tick sull'asse delle x come richiesto: 1, 10, 100, ...
    axes.set_xticks([1, 10, 100])
    axes.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Colori per le diverse fasi
    colors = ['grey', 'yellow', 'cyan', 'purple']

    # Crea il grafico a barre cumulativo


    # Crea una mappa dei colori per le fasi
    color_map = {
        'side_light_duration': 'lightpink',
        'duration_water_delivery': 'black',
        'duration_drink_delay': 'lightskyblue',
        'center_light_duration': 'yellow',
        'duration_timeout': 'orange',
        'duration_wrong_side': 'seagreen'
    }


    columns_of_interest = df[['trial', 'center_light_duration','side_light_duration', 'duration_water_delivery', 'duration_drink_delay', 'duration_timeout', 'duration_wrong_side']]
    df_trial_subgroup = columns_of_interest.copy()
    df_trial_subgroup = pd.melt(df_trial_subgroup, id_vars=['trial'], var_name='phase', value_name='duration')
    df_trial_subgroup['cumulative_duration'] = df_trial_subgroup.groupby(['trial', 'phase'])['duration'].cumsum()

    # Calcola la somma cumulativa per ciascuna fase separatamente
    df_trial_subgroup['cumulative_duration'] = df_trial_subgroup.groupby(['trial', 'phase'])['duration'].cumsum()

    # axes.set_xscale('log')  # Usa la scala logaritmica sull'asse x

    legend_labels = []  # Lista per le etichette della legenda

    for trial in df_trial_subgroup['trial'].unique():
        trial_data = df_trial_subgroup[df_trial_subgroup['trial'] == trial]
        for phase in df_trial_subgroup['phase'].unique():
            phase_data = trial_data[trial_data['phase'] == phase]
            # Rappresenta la fase con barre
            plt.barh([trial] * len(phase_data), phase_data['duration'],
                     left=phase_data['cumulative_duration'].shift(1, fill_value=0),
                     color=color_map[phase])
            legend_labels.append(phase)
    # Aumenta la trasparenza e riduci la larghezza delle barre
    bar_width = 0.1  # Regola questo valore per cambiare la larghezza delle barre
    alpha_value = 0.7  # Regola questo valore per cambiare la trasparenza

    legend_labels = set()  # Usa un set per evitare duplicati nella legenda

    for trial in df_trial_subgroup['trial'].unique():
        trial_data = df_trial_subgroup[df_trial_subgroup['trial'] == trial]
        y_positions = trial - bar_width / 2  # Calcola la posizione y per le barre
        for phase in df_trial_subgroup['phase'].unique():
            phase_data = trial_data[trial_data['phase'] == phase]
            plt.barh(y_positions, phase_data['duration'],
                     height=bar_width, alpha=alpha_value,
                     left=phase_data['cumulative_duration'].shift(1, fill_value=0),
                     color=color_map[phase])
            legend_labels.add(phase)

    # Aggiungi la griglia
    axes.grid(True, linestyle='--', alpha=0.5)

    # Impostiamo le etichette e il titolo
    axes.set_xlabel('Time (sec)')
    axes.set_ylabel('Trial')
    axes.set_title('Trial Raster')



    pdf_pages = PdfPages(save_path)

    # Save the plot to the PDF
    pdf_pages.savefig()

    # Close the PdfPages object to save the PDF file
    pdf_pages.close()


