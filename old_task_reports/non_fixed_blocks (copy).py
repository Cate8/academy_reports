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


def daily_report_non_fixed_blocks(df, save_path, date):


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

    df['Port5In_START'] = df['Port5In_START'].astype(str)
    df['Port2In_START'] = df['Port2In_START'].astype(str)
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
    df['total_poke_left'] = exploded_port2_df.groupby(exploded_port2_df.trial)['left_poke'].sum()
    df['total_poke_centre'] = exploded_port3_df.groupby(exploded_port3_df.trial)['central_poke'].sum()
    df['total_poke_right'] = exploded_port5_df.groupby(exploded_port5_df.trial)['right_poke'].sum()

    # comparison to find poke before the correct one
    exploded_port2_df['result'] = np.where(
        exploded_port2_df['Port2In_START'] <= exploded_port2_df['STATE_side_light_END'], 1, 0)
    exploded_port3_df['result'] = np.where(
        exploded_port3_df['Port3In_START'] <= exploded_port3_df['STATE_side_light_END'], 1, 0)
    exploded_port5_df['result'] = np.where(
        exploded_port5_df['Port5In_START'] <= exploded_port5_df['STATE_side_light_END'], 1, 0)

    # Creazione di nuove colonne in df per i risultati sommati raggruppati per l'indice
    df['poke_before_correct_left'] = exploded_port2_df.groupby(exploded_port2_df.trial)['result'].sum()
    df['poke_before_correct_centre'] = exploded_port3_df.groupby(exploded_port3_df.trial)['result'].sum()
    df['poke_before_correct_right'] = exploded_port5_df.groupby(exploded_port5_df.trial)['result'].sum()

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

    # Selecting specific columns to create error_df
    columns = df[["trial", "side", "STATE_wrong_side_START", 'correct_outcome_int']]
    error_df = columns.copy()

    # Initialize new columns
    error_df['error'] = 0
    error_df['error_L'] = 0
    error_df['error_R'] = 0

    error_df['STATE_wrong_side_START'] = error_df['STATE_wrong_side_START'].astype(float)

    # Iterate and update rows based on conditions
    for trial, param in enumerate(error_df['STATE_wrong_side_START']):
        if param != 0.00000:
            try:
                error_df.at[trial, 'error'] = 1
            except:
                pass
        try:
            if ((param != 0.00000) & (error_df.at[trial, 'side'] == "right")):
                error_df.at[trial, 'error_R'] = 1
        except:
            pass
        try:
            if ((param != 0.00000) & (error_df.at[trial, 'side'] == "left")):
                error_df.at[trial, 'error_L'] = 1
        except:
            pass

    # Conteggio dei valori uguali a 1 in ogni colonna
    error_df['error_count'] = error_df['error'].sum()
    error_df['error_L_count'] = error_df['error_L'].sum()
    error_df['error_R_count'] = error_df['error_R'].sum()

    error_df['error_rate'] = (error_df['error_count'].iloc[0]/error_df['trial'].iloc[-1]) * 100
    error_df['error_rate'] = error_df['error_rate'].round(2)

    formatted_probability = "{:.2f}".format(df['probability_r'].iloc[0])

    # Add session summary text
    session_summary = f'''
     S4 Sessions info
     Date: {df['date'].iloc[0]}, Animal ID: {df['subject'].iloc[0]}, Animal weight: {df['subject_weight'].iloc[0]},Box number: {df['box'].iloc[0]}, Trials: {df['trial'].iloc[-1]}, Session duration: {formatted_session_duration} min, 
     Center latency (median): {str(round(df.centre_response_latency.iloc[1], 2))} s, Side latency (median): {str(round(df.side_response_latency.iloc[1], 2))} s, Missed (sides): {count_missed_trials}, Omission (sides): {tot_center_omission}, Missed (center): {count_missed_center_trials}, 
     Omission (center): {tot_center_omission}, Water intake: {df['water_intake'].iloc[0]}ml, Total pokes: {poke_df['total_poke'].iloc[0]}, (R: {poke_df['total_n_poke_right'].iloc[0]}, C: {poke_df['total_n_poke_centre'].iloc[0]}, L: {poke_df['total_n_poke_left'].iloc[0]}), Errors: {error_df['error_count'].iloc[0]} (R: {error_df['error_R_count'].iloc[0]}, L: {error_df['error_L_count'].iloc[0]}), Error rate: {error_df['error_rate'].iloc[0]},
     Trials in block: 30, P(right):{formatted_probability}, Alternation blocks mode: {df['Prob_block_type'].iloc[0]}, Block type:{df['Block_type'].iloc[0]}, R/L probability:{df['Probability_L_R_blocks'].iloc[0]}
     '''


    plt.figtext(0.00, 0.91, session_summary, fontsize=9)

    # 1 PLOT: Probability computed from the first trial outcome to get the reward omissions and misses
    df['first_response_center'] = df['Port3In_START'].str.split(',').str[0].astype(float)

    #calculate missing and omission per port
    column_names = ['trial', 'side','correct_outcome_int', 'first_response_left', 'first_response_center', 'first_response_right']
    omission_df = df[column_names].copy()
    # replace all nans with 0
    omission_df = omission_df.replace(np.nan, 0)

    #OMISSION
    #general omission: no response in centre and side ports (it's at the same time a central omission).
    omission_df['general_omission'] = (
            (omission_df['first_response_left'] == 0) &
            (omission_df['first_response_right'] == 0) &
            (omission_df['first_response_center'] == 0)
    ).astype(int)
    # left omission: no response in left port when reward it's on side left, and no response in right port too.
    omission_df['left_omission'] = np.where(
        (omission_df['side'] == "left") &
        (omission_df['first_response_left'] == 0) &
        (omission_df['first_response_right'] == 0) &
        (omission_df['first_response_center'] != 0),
        1,  # true
        0  # false
    )
    # right omission: no response in right port when reward it's on side right, and no response in right port too.
    omission_df['right_omission'] = np.where(
        (omission_df['side'] == "right") &
        (omission_df['first_response_left'] == 0) &
        (omission_df['first_response_right'] == 0) &
        (omission_df['first_response_center'] != 0),
        1,  # true
        0  # false
    )

    #MISSES
    #central miss: when no poke in center but poke in left and right or in at least one of them
    omission_df['central_miss'] = (
            (omission_df['first_response_center'] == 0)&
            (omission_df['first_response_left'] != 0) &
            (omission_df['first_response_right'] == 0) |
            (omission_df['first_response_center'] == 0) &
            (omission_df['first_response_left'] == 0) &
            (omission_df['first_response_right'] != 0) |
            (omission_df['first_response_center'] == 0) &
            (omission_df['first_response_left'] != 0) &
            (omission_df['first_response_right'] != 0)
    ).astype(int)
    #left miss: when no poke in left when reward it's on left side but poke in centre and right or in at least one of them
    omission_df['left_miss'] = (
            ((omission_df['side'] == "left") & (df['STATE_timeout_START'] == 0)) &
            (omission_df['first_response_left'] == 0) &
            (omission_df['first_response_center'] != 0) &
            (omission_df['first_response_right'] != 0)
    ).astype(int)
    #right miss: when no poke in right when reward it's on right side but poke in centre and left or in at least one of them
    omission_df['right_miss'] = (
            ((omission_df['side'] == "right") & (df['STATE_timeout_START'] == 0)) &
            (omission_df['first_response_right'] == 0) &
            (omission_df['first_response_center'] != 0) &
            (omission_df['first_response_left'] != 0)
    ).astype(int)


    # Lista per raccogliere le probabilità

    df['rolling_prob'] = df['correct_outcome_int'].rolling(window=5, min_periods=1).mean()

    prob_colums = df[["trial", "side", "first_trial_response", "correct_outcome_int"]]
    prob_df = prob_colums.copy()
    prob_df["right_rewards"] = ((prob_df['side'] == 'right') & (prob_df['correct_outcome_int'] == 1)).astype(int)

    prob_df['rolling_avg_right_reward'] = prob_df["right_rewards"].rolling(window=5, min_periods=1).mean()

    # Creazione del subplot con dimensioni specifiche nella griglia (1600, 50)
    ax = plt.subplot2grid((1600, 50), (1, 1), rowspan=450, colspan=90)

    line_data = prob_df['rolling_avg_right_reward']
    plt.plot(df['trial'], df['probability_r'], '-', color='black', linewidth=1, alpha= 0.7)

    # Imposta i limiti dell'asse y con un margine più ampio
    ax.set_ylim(-0.5, 1.5)

    # Imposta i valori delle tacche sull'asse y
    ax.set_yticks([0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1])
    plt.axhline(y=1, linestyle='solid', color='black', alpha=0.7)
    plt.axhline(y=0.5, linestyle='--', color='lightgray', alpha=0.7)
    plt.axhline(y=0, linestyle= 'solid', color='black', alpha=0.7)

    # Grafico a linee

    ax.plot(df.trial, line_data, linewidth=2, color='mediumturquoise')

    column = ['trial', 'side', 'correct_outcome_int','first_response_left', 'first_response_right']
    first_lick_df = df[column].copy()

    conditions = [
        (first_lick_df.first_response_left == 0) & (first_lick_df.first_response_right == 0),
        first_lick_df.first_response_left == 0,
        first_lick_df.first_response_right == 0,
        first_lick_df.first_response_left <= first_lick_df.first_response_right,
        first_lick_df.first_response_left > first_lick_df.first_response_right,
    ]

    choices = ["no_response",
               "right",
               "left",
               "left",
               "right"]
    # create a new column in the DF based on the conditions

    first_lick_df["first_trial_response"] = np.select(conditions, choices)

    # Crea la colonna 'first_resp_left'
    first_lick_df["first_resp_left"] = first_lick_df["first_trial_response"].apply(lambda x: 1 if x == "left" else 0)

    # Crea la colonna 'first_resp_right'
    first_lick_df["first_resp_right"] = first_lick_df["first_trial_response"].apply(lambda x: 1 if x == "right" else 0)

    # Trova le posizioni dei tick per 'left' e 'right'
    left_ticks = first_lick_df[first_lick_df["first_resp_left"] == 1].trial
    right_ticks = first_lick_df[first_lick_df["first_resp_right"] == 1].trial

    # Plotta i tick marks per 'left' e 'right'
    for i, row in first_lick_df.iterrows():
        # Determina la dimensione del marker in base alla corrispondenza
        markersize = 15 if row["first_resp_left"] == row["correct_outcome_int"] else 5
        y_coord = -0.35 if markersize == 5 else -0.15
        trial = row['trial']
        # Per 'left'
        if row["first_resp_left"] == 1:
            ax.plot(trial, y_coord, marker='|', color='green', markersize=markersize)


    for i, row in first_lick_df.iterrows():
            # Determina la dimensione del marker in base alla corrispondenza
        markersize = 15 if row["first_resp_right"] == row["correct_outcome_int"] else 5
        y_coord = 1.35 if markersize == 5 else 1.15
        trial = row['trial']
        # Per 'right'
        if row["first_resp_right"] == 1:
            ax.plot(trial, y_coord, marker='|', color='purple', markersize=markersize)

    # Identifica i punti in cui l'indice di blocco cambia
    block_changes = df['Block_index'].diff().fillna(0).abs()
    change_points = df[block_changes > 0].trial

    # Itera su ciascun blocco unico
    for block in df['Block_index'].unique():
        block_data = df[df['Block_index'] == block]
        start = block_data['trial'].min()  # Inizio del blocco
        end = block_data['trial'].max()  # Fine del blocco
        block_center = (start + end) / 2  # Calcola il punto centrale del blocco
        block_prob = block_data['probability_r'].iloc[0]  # Probabilità del blocco

        # Scegli il colore in base alla probabilità
        if block_prob > 0.5:
            color = 'purple'
        elif block_prob == 0.5:
            color = 'blue'
        else:
            color = 'green'

        # Disegna una linea orizzontale per il blocco e la probabilità
        ax.hlines(y=1.5, xmin=start, xmax=end, colors=color, linestyles='solid', linewidth=10)
        ax.text(block_center, 1.6, f'{block_prob:.2f}', ha='center', va='center',
                backgroundcolor='white', fontsize=5)

    # Aggiungi linee tratteggiate per ogni cambio di blocco
    for point in change_points:
        ax.axvline(x=point, color='gray', linestyle='--')

    # Right: omission and misses
    for trial, param in enumerate(omission_df['right_omission']):
        if param == 1:
            plt.scatter(trial, -0.2, color='purple', marker="X", s=30)  # Sostituisci 'trial' con la posizione x corretta
        for trial, param in enumerate(omission_df['right_miss']):
            if param == 1:
                plt.scatter(trial, 1.3, color='purple',  marker="x", s=30)  # Sostituisci 'trial' con la posizione x corretta

    # left: omission and misses
    for trial, param in enumerate(omission_df['left_omission']):
        if param == 1:
            plt.scatter(trial, 1.4, color='green', marker="X", s=30)  # Sostituisci 'trial' con la posizione x corretta

    # Aggiungi i ticks per i parametri di sinistra
    for trial, param in enumerate(omission_df['left_miss']):
        if param == 1:
            plt.scatter(trial, 1.3, color='green', marker="x", s=30)  # Sostituisci 'trial' con la posizione x corretta

    # central thicks: omission and misses
    for trial, param in enumerate(omission_df['general_omission']):
        if param == 1:
            plt.scatter(trial, 0.5, color='black',  marker="X", s=30)  # Sostituisci 'trial' con la posizione x corretta

    # Aggiungi i ticks per i parametri di sinistra
    for trial, param in enumerate(omission_df['central_miss']):
        if param == 1:
            plt.scatter(trial, 0.5, color='black',  marker="x", s=30)  # Sostituisci 'trial' con la posizione x corretta

    ax.text(1.02, 0.1, 'L', ha='left', va='top', color='green', transform=ax.transAxes, fontsize=10)

    # Posiziona la lettera 'R' appena fuori dal bordo destro del grafico
    ax.text(1.02, 0.9, 'R', ha='left', va='bottom', color='purple', transform=ax.transAxes, fontsize=10)

    # Posiziona la lettera 'C' appena fuori dal bordo destro del grafico
    ax.text(1.02, 0.45557, 'C', ha='left', va='bottom', color='black', transform=ax.transAxes, fontsize=10)

    selected_trials = df.trial[::19]  # 20 trial
    ax.set_xticks(selected_trials)
    ax.set_xticklabels(selected_trials)
    ax.set_xlabel('trial')
    ax.set_ylabel('P(right)')
    plt.title('Probability right reward', pad=20)

    # 2 PLOT: latency to the first correct poke

    axes = plt.subplot2grid((1600, 50), (1150, 1), rowspan=450, colspan=90)

    plt.plot(df.trial, df.side_response_latency, color='dodgerblue', label='Side')
    plt.plot(df.trial, df.centre_response_latency, color='black', label='Centre')

    # Personalizzazione dei ticks sull'asse y

    custom_yticks = [0, 1, 10, 20, 50, 100, 200, 250, 300]  # I valori che desideri mostrare sull'asse y
    axes.set_yscale('log')
    axes.set_yscale('log')
    axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))

    for y in custom_yticks:
        plt.axhline(y=y, linestyle='--', color='lightgray', alpha=0.7)


    # Aggiungi titoli e etichette
    plt.legend()
    plt.title('Latency to fist correct poke')
    plt.xlabel('Trial')
    plt.ylabel('Latency (s)')

    plt.legend(loc='upper right', bbox_to_anchor=(1, -0.2))

    # 3 PLOT: computed probability

    axes = plt.subplot2grid((1600, 50), (600, 1), rowspan=400, colspan=90)
    df['first_response_center'] = df['Port3In_START'].str.split(',').str[0].astype(float)

    #calculate missing and omission per port
    column_names = ['trial', 'side','correct_outcome_int', 'first_response_left', 'first_response_center', 'first_response_right']
    omission_df = df[column_names].copy()
    # replace all nans with 0
    omission_df = omission_df.replace(np.nan, 0)

    #OMISSION
    #general omission: no response in centre and side ports (it's at the same time a central omission).
    omission_df['general_omission'] = (
            (omission_df['first_response_left'] == 0) &
            (omission_df['first_response_right'] == 0) &
            (omission_df['first_response_center'] == 0)
    ).astype(int)
    # left omission: no response in left port when reward it's on side left, and no response in right port too.
    omission_df['left_omission'] = np.where(
        (omission_df['side'] == "left") &
        (omission_df['first_response_left'] == 0) &
        (omission_df['first_response_right'] == 0) &
        (omission_df['first_response_center'] != 0),
        1,  # true
        0  # false
    )
    # right omission: no response in right port when reward it's on side right, and no response in right port too.
    omission_df['right_omission'] = np.where(
        (omission_df['side'] == "right") &
        (omission_df['first_response_left'] == 0) &
        (omission_df['first_response_right'] == 0) &
        (omission_df['first_response_center'] != 0),
        1,  # true
        0  # false
    )

    #MISSES
    #central miss: when no poke in center but poke in left and right or in at least one of them
    omission_df['central_miss'] = (
            (omission_df['first_response_center'] == 0)&
            (omission_df['first_response_left'] != 0) &
            (omission_df['first_response_right'] == 0) |
            (omission_df['first_response_center'] == 0) &
            (omission_df['first_response_left'] == 0) &
            (omission_df['first_response_right'] != 0) |
            (omission_df['first_response_center'] == 0) &
            (omission_df['first_response_left'] != 0) &
            (omission_df['first_response_right'] != 0)
    ).astype(int)
    #left miss: when no poke in left when reward it's on left side but poke in centre and right or in at least one of them
    omission_df['left_miss'] = (
            ((omission_df['side'] == "left") & (df['STATE_timeout_START'] == 0)) &
            (omission_df['first_response_left'] == 0) &
            (omission_df['first_response_center'] != 0) &
            (omission_df['first_response_right'] != 0)
    ).astype(int)
    #right miss: when no poke in right when reward it's on right side but poke in centre and left or in at least one of them
    omission_df['right_miss'] = (
            ((omission_df['side'] == "right") & (df['STATE_timeout_START'] == 0)) &
            (omission_df['first_response_right'] == 0) &
            (omission_df['first_response_center'] != 0) &
            (omission_df['first_response_left'] != 0)
    ).astype(int)


    # Lista per raccogliere le probabilità

    df['rolling_prob'] = df['correct_outcome_int'].rolling(window=5, min_periods=1).mean()


    # Creazione del subplot con dimensioni specifiche nella griglia (1600, 50)

    line_data = df['rolling_prob']
    plt.plot(df['trial'], df['probability_r'], '-', color='black', linewidth=1, alpha= 0.7)

    # Imposta i limiti dell'asse y con un margine più ampio
    axes.set_ylim(-0.5, 1.5)

    # Imposta i valori delle tacche sull'asse y
    axes.set_yticks([0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1])
    plt.axhline(y=1, linestyle='solid', color='black', alpha=0.7)
    plt.axhline(y=0.5, linestyle='--', color='lightgray', alpha=0.7)
    plt.axhline(y=0, linestyle= 'solid', color='black', alpha=0.7)

    # Grafico a linee

    axes.plot(df.trial, line_data, linewidth=2, color='orange')

    column = ['trial', 'side', 'correct_outcome_int','first_response_left', 'first_response_right']
    first_lick_df = df[column].copy()

    conditions = [
        (first_lick_df.first_response_left == 0) & (first_lick_df.first_response_right == 0),
        first_lick_df.first_response_left == 0,
        first_lick_df.first_response_right == 0,
        first_lick_df.first_response_left <= first_lick_df.first_response_right,
        first_lick_df.first_response_left > first_lick_df.first_response_right,
    ]

    choices = ["no_response",
               "right",
               "left",
               "left",
               "right"]
    # create a new column in the DF based on the conditions

    first_lick_df["first_trial_response"] = np.select(conditions, choices)

    # Crea la colonna 'first_resp_left'
    first_lick_df["first_resp_left"] = first_lick_df["first_trial_response"].apply(lambda x: 1 if x == "left" else 0)

    # Crea la colonna 'first_resp_right'
    first_lick_df["first_resp_right"] = first_lick_df["first_trial_response"].apply(lambda x: 1 if x == "right" else 0)

    # Trova le posizioni dei tick per 'left' e 'right'
    left_ticks = first_lick_df[first_lick_df["first_resp_left"] == 1].trial
    right_ticks = first_lick_df[first_lick_df["first_resp_right"] == 1].trial

    # Plotta i tick marks per 'left' e 'right'
    for i, row in first_lick_df.iterrows():
        # Determina la dimensione del marker in base alla corrispondenza
        markersize = 15 if row["first_resp_left"] == row["correct_outcome_int"] else 5
        y_coord = -0.35 if markersize == 5 else -0.15
        trial = row['trial']
        # Per 'left'
        if row["first_resp_left"] == 1:
            axes.plot(trial, y_coord, marker='|', color='green', markersize=markersize)


    for i, row in first_lick_df.iterrows():
            # Determina la dimensione del marker in base alla corrispondenza
        markersize = 15 if row["first_resp_right"] == row["correct_outcome_int"] else 5
        y_coord = 1.35 if markersize == 5 else 1.15
        trial = row['trial']

        # Per 'right'
        if row["first_resp_right"] == 1:
            axes.plot(trial, y_coord, marker='|', color='purple', markersize=markersize)


    # Calcola il numero di blocchi (segmenti di 20 trials ciascuno)

    blocks_lenght = 30
    num_blocks = len(df) // blocks_lenght

    df['mean_probabilities'] = np.nan

    # Calcola la media per ogni blocco e assegna i valori usando .loc
    for i in range(num_blocks):
        start = i * blocks_lenght
        end = start + blocks_lenght
        mean = df['rolling_prob'][start:end].mean()
        df.loc[start:end - 1, 'mean_probabilities'] = mean  # Nota: end-1 perché .loc è inclusivo

    # Gestisci l'ultimo blocco se i trials non sono un multiplo di 20
    if len(df) % blocks_lenght != 0:
        mean = df['rolling_prob'][end:].mean()
        df.loc[end:, 'mean_probabilities'] = mean

    # Assumendo che ax sia l'asse del tuo grafico Matplotlib
    num_blocks = len(df) // blocks_lenght

    for i in range(num_blocks):
        start = i * blocks_lenght
        end = start + blocks_lenght

        # Utilizza direttamente il valore da df['mean_probabilities']
        mean_prob = df['mean_probabilities'][start:end].mean()
        formatted_mean_prob = f"{mean_prob:.2f}"

        center = (start + end) / 2
        axes.text(center, 1.6, formatted_mean_prob, ha='center', va='center', backgroundcolor='white', fontsize=5)

    # Gestisci l'ultimo blocco se i trials non sono un multiplo di 20
    if len(df) % blocks_lenght != 0:
        start = num_blocks * blocks_lenght
        end = len(df)
        mean_prob = df['mean_probabilities'][start:end].mean()
        formatted_mean_prob = f"{mean_prob:.2f}"
        center = (start + end) / 2
        axes.text(center, 1.6, formatted_mean_prob, ha='center', va='center', fontsize=5)

    # Assumendo che il numero totale di trial sia accessibile
    num_trials = len(df)

    for x in range(0, num_trials, blocks_lenght ):
        plt.axvline(x=x, color='gray', linestyle='--')
    # Aggiungi anche una linea alla fine, se necessario
    if num_trials % blocks_lenght  != 0:
        plt.axvline(x=num_trials, color='gray', linestyle='--')

    # Assumendo che ax sia l'asse del tuo grafico Matplotlib
    num_blocks = len(df) // blocks_lenght

    for i in range(num_blocks):
        start = i * blocks_lenght
        end = start + blocks_lenght - 1  # -1 perché l'indice finale è inclusivo

        first_twenty_sides = df['side'][start:start + 20]

        # Controlla se almeno i primi 10 valori sono 'left'
        if (first_twenty_sides == 'right').sum() >= 10:
            color = 'purple'
        # Altrimenti, controlla se almeno 5 di questi sono 'right'
        elif (first_twenty_sides == 'left').sum() >= 10:
            color = 'green'
        # Se nessuna delle condizioni precedenti è soddisfatta, imposta un colore di default
        else:
            color = 'gray'
            # Disegna una linea orizzontale per il blocco

        # Disegna una linea orizzontale per il blocco
        axes.hlines(y=1.5, xmin=start, xmax=end, colors=color, linestyles='solid', linewidth=10)

    # Gestisci l'ultimo blocco se i trials non sono un multiplo di 20
    if len(df) % blocks_lenght != 0:
        start = num_blocks * blocks_lenght
        end = len(df) - 1
        block_start_side = df['side'].iloc[start]  ##change this!!!!!
        color = 'green' if block_start_side == 'left' else 'purple'
        axes.hlines(y=1.5, xmin=start, xmax=end, colors=color, linestyles='solid', linewidth=10)

    # Assumendo che ax sia l'asse del tuo grafico Matplotlib
    num_blocks = len(df) // blocks_lenght

    # Disegna una linea verticale tratteggiata all'inizio di ogni blocco di 20 trials
    for i in range(num_blocks + 1):  # +1 per includere la linea finale del grafico
        x_pos = i * blocks_lenght
        axes.vlines(x=x_pos, ymin=-0.2, ymax=1.2, colors='grey', linestyles='dashed', alpha=0.5)

    # Se il numero totale di trials non è un multiplo di 20, aggiungi una linea alla fine
    if len(df) % blocks_lenght != 0:
        axes.vlines(x=len(df), ymin=-0.2, ymax=1.2, colors='grey', linestyles='dashed', alpha=0.5)

    # Right: omission and misses
    for trial, param in enumerate(omission_df['right_omission']):
        if param == 1:
            plt.scatter(trial, -0.2, color='purple', marker="X", s=30)  # Sostituisci 'trial' con la posizione x corretta
        for trial, param in enumerate(omission_df['right_miss']):
            if param == 1:
                plt.scatter(trial, 1.3, color='purple',  marker="x", s=30)  # Sostituisci 'trial' con la posizione x corretta

    # left: omission and misses
    for trial, param in enumerate(omission_df['left_omission']):
        if param == 1:
            plt.scatter(trial, 1.4, color='green', marker="X", s=30)  # Sostituisci 'trial' con la posizione x corretta

    # Aggiungi i ticks per i parametri di sinistra
    for trial, param in enumerate(omission_df['left_miss']):
        if param == 1:
            plt.scatter(trial, 1.3, color='green', marker="x", s=30)  # Sostituisci 'trial' con la posizione x corretta

    # central thicks: omission and misses
    for trial, param in enumerate(omission_df['general_omission']):
        if param == 1:
            plt.scatter(trial, 0.5, color='black',  marker="X", s=30)  # Sostituisci 'trial' con la posizione x corretta

    # Aggiungi i ticks per i parametri di sinistra
    for trial, param in enumerate(omission_df['central_miss']):
        if param == 1:
            plt.scatter(trial, 0.5, color='black',  marker="x", s=30)  # Sostituisci 'trial' con la posizione x corretta



    selected_trials = df.trial[::19]  # 20 trial
    axes.set_xticks(selected_trials)
    axes.set_xticklabels(selected_trials)
    axes.set_xlabel('trial')
    axes.set_ylabel('P(reward)')
    plt.title('Probability Rewarded trials', pad=20)



    pdf_pages = PdfPages(save_path)

    # Save the plot to the PDF
    pdf_pages.savefig()

    # Close the PdfPages object to save the PDF file
    pdf_pages.close()




