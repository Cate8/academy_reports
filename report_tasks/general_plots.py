import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
import datetime
from datetime import datetime, timedelta, date
from fpdf import FPDF
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import logging
import warnings
from academy_reports.utils import compute_window

# Right thicks: omission and misses
for trial, param in enumerate(omission_df['right_omission']):
    if param == 1:
        plt.scatter(trial, 1.2, color='red', marker="X")  # Sostituisci 'trial' con la posizione x corretta
    for trial, param in enumerate(omission_df['right_miss']):
        if param == 1:
            plt.scatter(trial, 1.2, color='red', marker="v")  # Sostituisci 'trial' con la posizione x corretta

# left thicks: omission and misses
for trial, param in enumerate(omission_df['left_omission']):
    if param == 1:
        plt.scatter(trial, -0.05, color='blue', marker="X")  # Sostituisci 'trial' con la posizione x corretta
# Aggiungi i ticks per i parametri di sinistra
for trial, param in enumerate(omission_df['left_miss']):
    if param == 1:
        plt.scatter(trial, -0.05, color='blue', marker="v")  # Sostituisci 'trial' con la posizione x corretta

# central thicks: omission and misses
for trial, param in enumerate(omission_df['general_omission']):
    if param == 1:
        plt.scatter(trial, -0.5, color='gold', marker="X")  # Sostituisci 'trial' con la posizione x corretta
# Aggiungi i ticks per i parametri di sinistra
for trial, param in enumerate(omission_df['central_miss']):
    if param == 1:
        plt.scatter(trial, -0.5, color='gold', marker="v")  # Sostituisci 'trial' con la posizione x corretta



# FORTH PLOT: accuracy first poke/ accuracy last poke
# left/right licks and accuracy in a window of trials called "rolling average"
# Calculate moving averages for accuracy, left poke count and right poke count
# replace all nans with 0
df = df.replace(np.nan, 0)

# df["accuracy"] = ((df['correct_outcome_int'] / df['trial']) * 100).astype(float)
df['last_response_right'] = df['Port5In_START'].str.split(',').str[-1].astype(float)
df['last_response_left'] = df['Port2In_START'].str.split(',').str[-1].astype(float)

columns = df[['trial', 'side', 'first_response_right', 'first_response_left', 'Port5In_START', 'Port2In_START',
              'last_response_right', 'last_response_left']]
acc_df = columns.copy()
acc_df = acc_df.replace(np.nan, 0)

# convert string to float to compare the time
acc_df['last_response_left'] = acc_df['last_response_left'].astype(float)
acc_df['last_response_right'] = acc_df['last_response_right'].astype(float)
acc_df['first_response_left'] = acc_df['first_response_left'].astype(float)
acc_df['first_response_right'] = acc_df['first_response_right'].astype(float)


# first pokes
def confronta_valori(acc_df):
    first_left = []
    first_right = []
    for idx, row in df.iterrows():
        if row['first_response_left'] == 0:
            first_left.append(0)
            first_right.append(1)
        elif row['first_response_right'] == 0:
            first_left.append(1)
            first_right.append(0)
        else:
            # compare values != 0
            if row['first_response_left'] < row['first_response_right']:
                first_left.append(1)
                first_right.append(0)
            else:
                first_left.append(0)
                first_right.append(1)

    return first_left, first_right


# Utilizzo della funzione per confrontare i valori delle colonne nel DataFrame 'acc_df'
acc_df['first_left'], acc_df['first_right'] = confronta_valori(
    acc_df[['first_response_left', 'first_response_right']])


# last pokes
def confronta_valori(acc_df):
    last_left = []
    last_right = []
    for idx, row in df.iterrows():
        if row['last_response_left'] == 0:
            last_left.append(0)
            last_right.append(1)
        elif row['last_response_right'] == 0:
            last_left.append(1)
            last_right.append(0)
        else:
            # compare values != 0
            if row['last_response_left'] > row['last_response_right']:
                last_left.append(1)
                last_right.append(0)
            else:
                last_left.append(0)
                last_right.append(1)

    return last_left, last_right


acc_df['last_left'], acc_df['last_right'] = confronta_valori(
    acc_df[['last_response_left', 'last_response_right']])

acc_df['first_poke_rewarded'] = 0  # Impostiamo tutti i valori iniziali a 0

acc_df.loc[(acc_df['first_left'] == 1) & (acc_df['side'] == 'left'), 'first_poke_rewarded'] = 1
acc_df.loc[(acc_df['first_right'] == 1) & (acc_df['side'] == 'right'), 'first_poke_rewarded'] = 1
acc_df['first_poke_accuracy'] = acc_df['first_poke_rewarded'].copy()

acc_df['last_poke_rewarded'] = 0  # Impostiamo tutti i valori iniziali a 0
acc_df.loc[(acc_df['last_left'] == 1) & (acc_df['side'] == 'left'), 'last_poke_rewarded'] = 1
acc_df.loc[(acc_df['last_right'] == 1) & (acc_df['side'] == 'right'), 'last_poke_rewarded'] = 1
acc_df['last_poke_accuracy'] = acc_df['last_poke_rewarded'].copy()

ax1 = plt.subplot2grid((1600, 50), (800, 1), rowspan=250, colspan=100)

# Scatter plot per accuratezza
# sns.scatterplot(x=df['trial'], y=acc_df['first_poke_accuracy'], s=30, ax=ax1,
# color=['blueviolet' if x == 1 else 'gray' for x in acc_df['first_poke_accuracy']])
plt.plot(acc_df.trial, acc_df.first_poke_accuracy, color='blueviolet', linestyle='-', linewidth=1, alpha=0.3)
# sns.scatterplot(x=df['trial'], y=acc_df['last_poke_accuracy'], s=30, ax=ax1,
# color=['brown' if x == 1 else 'black' for x in acc_df['last_poke_accuracy']])

plt.plot(acc_df.trial, acc_df.last_poke_accuracy, color='brown', linestyle='-', linewidth=1, alpha=0.3)

axes.set_yticks([0, 1.5])
plt.xlabel('Trial')
plt.ylabel('rewarded = 1')
# plt.title('Poke accuracy: first and last poke rewarded')
plt.legend()

# Creazione di un'area colorata per ogni intervallo di 'side' costante
current_side = acc_df['side'][0]
start_trial = 0

for idx, side in enumerate(acc_df['side']):
    if current_side != side:
        plt.axvspan(start_trial, idx, color='blue' if current_side == 'left' else 'white', alpha=0.1)
        start_trial = idx
        current_side = side

# Aggiunta dell'ultimo intervallo di 'side'
plt.axvspan(start_trial, len(acc_df), color='blue' if current_side == 'left' else 'white', alpha=0.1)
plt.plot(acc_df.trial, acc_df.last_poke_accuracy, color='brown', linestyle='-', linewidth=1, alpha=0.3)

# Impostazioni per il grafico
plt.xlabel('Trial')
plt.ylabel('Side')
plt.title('Sequence of Side over Trials')
plt.xlim(0, len(acc_df))
plt.ylim(0, 1.5)

# FORTH PLOT: trial distribution over time

columns_of_interest = df[['trial', 'duration_water_light', 'duration_waiting_light', 'duration_drink_delay']]
df_trial_subgroup = columns_of_interest.copy()
df_trial_subgroup = pd.melt(df_trial_subgroup, id_vars=['trial'], var_name='phase', value_name='duration')
df_trial_subgroup['cumulative_duration'] = df_trial_subgroup.groupby(['trial', 'phase'])['duration'].cumsum()

# Crea il grafico a barre cumulativo
axes = plt.subplot2grid((1600, 100), (1300, 1), rowspan=700, colspan=50)

# Crea una mappa dei colori per le fasi
color_map = {
    'duration_water_light': 'black',
    'duration_waiting_light': 'lightpink',
    'duration_drink_delay': 'lightskyblue'
}

# Calcola la somma cumulativa per ciascuna fase separatamente
df_trial_subgroup['cumulative_duration'] = df_trial_subgroup.groupby(['trial', 'phase'])['duration'].cumsum()

# axes.set_xscale('log')  # Usa la scala logaritmica sull'asse x


legend_labels = []  # Lista per le etichette della legenda
legend_handles = []  # Lista per gli oggetti di legenda

for trial in df_trial_subgroup['trial'].unique():
    trial_data = df_trial_subgroup[df_trial_subgroup['trial'] == trial]
    for phase in df_trial_subgroup['phase'].unique():
        phase_data = trial_data[trial_data['phase'] == phase]
        if phase == 'duration_waiting_light' and phase_data['duration'].max() == 300:
            # Rappresenta con un simbolo se la durata massima è 300
            plt.scatter(phase_data['cumulative_duration'].max(), trial, marker='o', c=color_map[phase], label=phase,
                        s=50)
            legend_labels.append(phase)
        else:
            # Rappresenta la fase con barre
            plt.barh([trial] * len(phase_data), phase_data['duration'],
                     left=phase_data['cumulative_duration'].shift(1, fill_value=0),
                     color=color_map[phase])
            legend_labels.append(phase)

# Aggiungi la legenda
plt.xlabel('Seconds')  # Modifica l'etichetta dell'asse x per riflettere la scala logaritmica
plt.ylabel('Trials')
plt.title('Trial phases duration')  # Modifica il titolo per riflettere la scala logaritmica
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Crea gli oggetti di legenda
unique_phases = df_trial_subgroup['phase'].unique()
for phase in unique_phases:
    legend_handles.append(plt.Line2D([0], [0], color=color_map[phase], lw=4, label=phase))

# Aggiungi la legenda
plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), title='Phases', prop={'size': 8})


def daily_report_S1(path, save_path):
    df = pd.read_csv(path, sep=';')


#Use the following 2 lines to run manually the task


    #path = "C:\\academy_reports\\academy_reports\\sessions\\raton1\\raton1_S1-0-0_20230915-122439.csv"
    #save_path = path[:-3] + 'pdf'

    #df = pd.read_csv(path, sep=';')


    # New columns (variables)

    df['duration_waiting_light'] = df['STATE_waiting_light_END'] - df['STATE_waiting_light_START']
    df['duration_drink_delay'] = df['STATE_drink_delay_END'] - df['STATE_drink_delay_START']
    df['duration_water_light'] = df['STATE_water_light_END'] - df['STATE_water_light_START']
    df['response_latency'] = df['duration_waiting_light'] + df['duration_water_light']
    df['trial_duration'] = df['response_latency'] + df['duration_drink_delay']

    df['first_response_left'] = df['Port5In_START'].str.split(',').str[0].astype(float)
    df['first_response_right'] = df['Port2In_START'].str.split(',').str[0].astype(float)

    # List of conditions for teat well the NANs

    conditions = [df.first_response_left.isnull() & df.first_response_right.isnull(),
                  df.first_response_left.isnull(),
                  df.first_response_right.isnull(),
                  df.first_response_left <= df.first_response_right,
                  df.first_response_left > df.first_response_right, ]

    # List of values to return

    choices = ["no_response",
               "right",
               "left",
               "left",
               "right"]

    # create a new column in the DF based on the conditions

    df["first_trial_response"] = np.select(conditions, choices)

    df["correct_outcome_bool"] = df["first_trial_response"] == df["side"]  # this is for having boolean variables (true/false)
    df['true_count'] = (df['correct_outcome_bool']).value_counts()[True]
    df["correct_outcome"] = np.where(df["first_trial_response"] == df["side"], "correct",
                                     "incorrect")  # (true = correct choice, false = incorrect side)
    df["correct_outcome_int"] = np.where(df["first_trial_response"] == df["side"], 1,
                                     0)  # (1 = correct choice, 0= incorrect side)

    # PLOT COLORS:

    left_c = 'blue'
    right_c = 'purple'
    correct_c = 'green'
    incorrect_c = 'red'
    label_kwargs = {'fontsize': 9}

    # PLOT PARAMETERS

    mpl.rcParams['lines.linewidth'] = 0.7
    mpl.rcParams['lines.markersize'] = 1
    mpl.rcParams['font.size'] = 7
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    label_kwargs = {'fontsize': 9, 'fontweight': 'roman'}

    plt.figure(figsize=(8, 11))

    # Other calculations and Averaging

    df['response_latency_mean'] = df['response_latency'].mean() #average latency to first response

    max_time_waiting_light = 300.00000  # max time in waiting light (for extrac missed trial)
    df['misses'] = (df['duration_waiting_light'] == max_time_waiting_light).astype(int)
    count_missed_trials = df['misses'].sum()
    missed_rate = (df['misses'].iloc[0] / df['trials_max'].iloc[0]) * 100
    formatted_missed_rate = "{:.2f}".format(missed_rate)

    df['tot_correct_choices'] = df['correct_outcome_int'].sum()  #correct choice rate
    correct_choice_rate = (df['tot_correct_choices'].iloc[0] / df['trials_max'].iloc[0]) * 100
    formatted_correct_choice_rate = "{:.2f}".format(correct_choice_rate)

    df['right_choices'] = (df['side'] == 'right').sum() # number of right choices
    right_choice_rate = (df['right_choices'].iloc[0] / df['trials_max'].iloc[0]) * 100
    formatted_right_choice_rate = "{:.2f}".format(right_choice_rate)

    df['left_choices'] = (df['side'] == 'left').sum()# number of left choices
    left_choice_rate = (df['left_choices'].iloc[0] / df['trials_max'].iloc[0]) * 100
    formatted_left_choice_rate = "{:.2f}".format(left_choice_rate)

    df['session_duration_s']= df['TRIAL_END'].iloc[29] - df['TRIAL_START'].iloc[0]
    df['session_duration_min'] = df['session_duration_s'] / 60

    ### PORTS ###
    #BPOD port5 -> box III (left)
    #BPOD port3 -> box II (central)
    #BPOD port2 -> box I (right)

    # Port-in counts (number of animal's pokes)
    df['n_right_pokes'] = df['Port2In_START'].count()
    df['n_central_pokes'] = df['Port3In_START'].count()
    df['n_left_pokes'] = df['Port5In_START'].count()
    df['total_pokes'] = df['n_right_pokes'].iloc[0] + df['n_central_pokes'].iloc[0] +df['n_left_pokes'].iloc[0]

    # Add session summary text
    session_summary = f'''
    S1 Sessions info
    Date: {df['date'].iloc[0]}, Animal ID: {df['subject'].iloc[0]}, Animal weight: {df['subject_weight'].iloc[0]},Box number: {df['box'].iloc[0]}, Total numbers of trials: {df['trials_max'].iloc[0]}, 
    Session duration: {str(round(df.session_duration_min.iloc[1],2))} min, Average response latency: {str(round(df.response_latency_mean.iloc[1],2))}, Correct trials: {formatted_correct_choice_rate}%, Missed: {count_missed_trials} ({formatted_missed_rate}%), 
    Right choice: {df['right_choices'].iloc[0]} ({formatted_right_choice_rate}%), Left choice: {df['left_choices'].iloc[0]} ({formatted_left_choice_rate}%), Total pokes: {df['total_pokes'].iloc[0]}, (R: {df['n_right_pokes'].iloc[0]}, C: {df['n_central_pokes'].iloc[0]}, L: {df['n_left_pokes'].iloc[0]})'''

    plt.figtext(0.0, 0.93, session_summary, fontsize=10)


    # FIRST PLOT: latency to the first correct poke

    plot_summary = f"Latency to the first correct poke\n"

    axes = plt.subplot2grid((1000, 50), (1, 1), rowspan=130, colspan=90)

    sns.scatterplot(x=df.trial, y=df.response_latency, hue=df.side, hue_order=['left', 'right'],
                    palette=[left_c, right_c], s=30, ax=axes)

    sns.lineplot(x=df.trial, y=df.response_latency, hue=df.side, hue_order=['left', 'right'],
                 palette=[left_c, right_c], ax=axes)

    axes.set_title(plot_summary)

    # SECOND PLOT: Outcome (first response = correct outcome)

    summary = f"Outcome\n"

    axes = plt.subplot2grid((1600, 50), (340, 1), rowspan=130, colspan=90)

    sns.scatterplot(x=df.trial, y=df.side, hue=df.correct_outcome, hue_order=['correct', 'incorrect'],
                    palette=[correct_c, incorrect_c], s=30, ax=axes)

    axes.set_title(summary)

    #THIRD PLOT: session's misses over trials

    summary = f"Misses over trials\n"

    axes = plt.subplot2grid((1600, 50), (620, 1), rowspan=110, colspan=90)
    missed_trials_df = df[df['misses'] == 1]

    sns.scatterplot(x=df['trial'], y=df['misses'], s=30, ax=axes, color=['darkorange' if x == 1 else 'lightseagreen' for x in df['misses']])
    axes.set_yticks([0, 1])

    axes.set_title(summary)

    #FORTH PLOT: accuracy on TRIAL
    # left/right licks and accuracy in a window of trials called "rolling average"

    # Calculate moving averages for accuracy, left poke count and right poke count

    df["accuracy"] = ((df['correct_outcome_int'] / df['trial']) * 100).astype(float)
    df['right_choices_rate'] = (df['right_choices'] / df['total_pokes'] * 100) #wrong
    df['left_choices_rate'] = (df['n_left_pokes'] / df['total_pokes'] * 100) #wrong

    # Define a function to compute a rolling window statistic
    def compute_window(column, group_size):

        return column.rolling(window=5, min_periods=1).mean()

    df['rolling_accuracy'] = compute_window(df["correct_outcome_int"], 5) * 100


    # Plot of rolling average
    axes = plt.subplot2grid((1600, 50), (870, 1), rowspan=130, colspan=90)

    plt.scatter(df.index, df.rolling_accuracy, label='choice accuracy', color='slateblue', marker='o', s=10)
    plt.plot(df.index, df.rolling_accuracy, color='mediumslateblue', linestyle='-', linewidth=1)

    axes.set_yticks([0, 50, 100])

    # Add 50% line on y axes
    y_value = 50 # value

    plt.axhline(y=y_value, color='black', linestyle='--')
    plt.xlabel('Trial')
    plt.ylabel('%')
    plt.title('Trials accuracy (rolling average)')
    plt.legend()

    # FIFTH PLOT: accuracy on POKES

    ### PORTS ###
    # BPOD port2 -> box I (right)
    # BPOD port3 -> box II (central)
    # BPOD port5 -> box III (left)
    # Define a timestamp count function by considering points (Epoch Unix.)
    def count_timestamps_with_points(timestamp_list):
        if pd.notna(timestamp_list):
            return sum(1 for ts in timestamp_list if '.' in str(ts))
        return 0

    # Apply function
    df['n_right_licks'] = df['Port2In_START'].apply(count_timestamps_with_points)
    df['n_left_licks'] = df['Port5In_START'].apply(count_timestamps_with_points)
    df['n_centre_licks'] = df['Port3In_START'].apply(count_timestamps_with_points)
    df['total_licks'] = df['n_centre_licks'] + df['n_left_licks'] + df['n_right_licks']

    def compute_window(column, group_size):

        return column.rolling(window=5, min_periods=1).mean()

    # Converting lick values in rates
    #df['rolling_accuracy'] = compute_window(df["correct_outcome_int"], 5) * 100
    df['rolling_right_lick_rate'] = compute_window((df['n_right_licks'] / df['total_licks']), 5) * 100
    df['rolling_left_lick_rate'] = compute_window((df['n_left_licks'] / df['total_licks']),5) * 100
    df['rolling_center_lick_rate'] = compute_window((df['n_centre_licks'] / df['total_licks']),5) * 100

    # Plot of rolling average
    axes = plt.subplot2grid((1600, 50), (900, 1), rowspan=130, colspan=90)

    #plt.scatter(df.index, df.rolling_accuracy, label='choice accuracy', color='slateblue', marker='o', s=10)
    #plt.plot(df.index, df.rolling_accuracy, color='mediumslateblue', linestyle='-', linewidth=1)
    plt.scatter(df.index, df.rolling_left_lick_rate, label='left pokes', color='green', marker='o', s=3)
    plt.plot(df.index, df.rolling_left_lick_rate, color='green', linestyle='-', linewidth=1)
    plt.scatter(df.index, df.rolling_right_lick_rate, label='right pokes', color='red', marker='o', s=3)
    plt.plot(df.index, df.rolling_right_lick_rate, color='red', linestyle='-', linewidth=1)
    plt.scatter(df.index, df.rolling_center_lick_rate, label='centre pokes', color='gold', marker='o', s=3)
    plt.plot(df.index, df.rolling_center_lick_rate, color='gold', linestyle='-', linewidth=1)

    axes.set_yticks([0, 50, 100])

    # Add 50% line on y axes
    y_value = 50 # value

    plt.axhline(y=y_value, color='black', linestyle='--')
    plt.xlabel('Trial')
    plt.ylabel('%')
    plt.title(' Global pokes % (rolling average)')
    # Aggiungi la legenda
    plt.legend(loc='upper right', bbox_to_anchor=(0.0, -0.2))

    ###
    # 6 PLOT: negative accuracy (pokes did before the correct one)

    ### PORTS ###
    # BPOD port2 -> box I (right)
    # BPOD port3 -> box II (central)
    # BPOD port5 -> box III (left)

    # Specifica le colonne di interesse


    columns_of_interest = ['trial', 'STATE_waiting_light_END', 'Port2In_START']
    columns_of_interest1 = ['trial','STATE_waiting_light_END', 'Port3In_START']
    columns_of_interest2 = ['trial','STATE_waiting_light_END', 'Port5In_START']

    # Crea un nuovo DataFrame con solo le colonne di interesse
    exploded_port2_df = df[columns_of_interest].copy()
    exploded_port3_df = df[columns_of_interest1].copy()
    exploded_port5_df = df[columns_of_interest2].copy()

    # Suddividi le colonne utilizzando la virgola come delimitatore
    exploded_port2_df['Port2In_START'] = df['Port2In_START'].str.split(',')
    exploded_port3_df['Port3In_START'] = df['Port3In_START'].str.split(',')
    exploded_port5_df['Port5In_START'] = df['Port5In_START'].str.split(',')

    # Esploa le colonne con liste in righe separate
    exploded_port2_df = exploded_port2_df.explode('Port2In_START')
    exploded_port3_df = exploded_port3_df.explode('Port3In_START')
    exploded_port5_df = exploded_port5_df.explode('Port5In_START')

    # Esploa le colonne con liste in righe separate
    exploded_port2_df = exploded_port2_df.explode('Port2In_START')
    exploded_port3_df = exploded_port3_df.explode('Port3In_START')
    exploded_port5_df = exploded_port5_df.explode('Port5In_START')

    # Converte la colonna 'Port2In_START' in float
    exploded_port2_df['Port2In_START'] = pd.to_numeric(exploded_port2_df['Port2In_START'], errors='coerce')
    exploded_port3_df['Port3In_START'] = pd.to_numeric(exploded_port3_df['Port3In_START'], errors='coerce')
    exploded_port5_df['Port5In_START'] = pd.to_numeric(exploded_port5_df['Port5In_START'], errors='coerce')

    # Esegui il confronto tra le colonne
    exploded_port2_df['risultato'] = np.where(exploded_port2_df['STATE_waiting_light_END'] <= exploded_port2_df['Port2In_START'], 1, 0)
    exploded_port3_df['risultato'] = np.where(exploded_port3_df['STATE_waiting_light_END'] <= exploded_port3_df['Port3In_START'], 1, 0)
    exploded_port5_df['risultato'] = np.where(exploded_port5_df['STATE_waiting_light_END'] <= exploded_port5_df['Port5In_START'], 1, 0)

    # Creazione di nuove colonne in df per i risultati sommati raggruppati per l'indice
    df['risultato_left'] = exploded_port2_df.groupby(exploded_port2_df.index)['risultato'].sum()
    df['risultato_centre'] = exploded_port3_df.groupby(exploded_port3_df.index)['risultato'].sum()
    df['risultato_right'] = exploded_port5_df.groupby(exploded_port5_df.index)['risultato'].sum()

    # Seleziona solo le colonne di interesse in un nuovo DataFrame

    N_accuracy = df[['trial', 'risultato_left', 'risultato_centre', 'risultato_right', 'total_licks']]

    N_accuracy['total_incorrect_pokes'] = N_accuracy['risultato_left'] + N_accuracy['risultato_centre'] + N_accuracy['risultato_right']

    N_accuracy['N_accuracy'] = (N_accuracy['total_incorrect_pokes'] / N_accuracy['total_licks']) * 100

    N_accuracy['prop_correct'] = (1/(N_accuracy['total_licks'])) * 100

    path_ = '''
    def compute_window(column, window_size):
        return column.rolling(window=window_size, min_periods=1).mean()

    N_accuracy['N_rolling_accuracy'] = compute_window(N_accuracy['N_accuracy'], window_size=5)
    N_accuracy['rolling_prop_correct'] = compute_window(N_accuracy['prop_correct'],  window_size=5)

    # Plot of rolling average
    axes = plt.subplot2grid((1600, 50), (1215, 1), rowspan=150, colspan=70)

    plt.scatter(N_accuracy.index, N_accuracy['N_rolling_accuracy'], label='incorrect choice accuracy', color='orangered', marker='o', s=10)
    plt.plot(N_accuracy.index, N_accuracy['N_rolling_accuracy'], color='orangered', linestyle='-', linewidth=1)
    plt.scatter(N_accuracy.index, N_accuracy['rolling_prop_correct'], label='proportion of correct choices', color='yellowgreen', marker='o', s=10)
    plt.plot(N_accuracy.index, N_accuracy['rolling_prop_correct'], color='yellowgreen', linestyle='-', linewidth=1)

    axes.set_yticks([0, 100 ])

     #Add 50% line on y axes
    y_value = 50  # value

    plt.axhline(y=y_value, color='black', linestyle='--')
    plt.xlabel('Trial')
    plt.ylabel('%')
    plt.title('Rolling average incorrect pokes')
    plt.legend(loc='upper right', bbox_to_anchor=(0.0, -0.2))
'''

    # 7 PLOT: average side error
    def compute_window(column, window_size):
        return column.rolling(window=window_size, min_periods=1).mean()

    df['risultato_left'] = compute_window(df['risultato_left'], window_size=5)
    df['risultato_centre'] = compute_window(df['risultato_centre'], window_size=5)
    df['risultato_right'] = compute_window(df['risultato_right'], window_size=5)

    # Plot of rolling average
    axes = plt.subplot2grid((1600, 50), (1215, 1), rowspan=200, colspan=90)

    sns.lineplot(x=df.trial, y=df['risultato_left'], color='green', label='Left pokes', linestyle='-', linewidth=1,
                 ax=axes)
    sns.lineplot(x=df.trial, y=df['risultato_right'], color='red', label='Right pokes', linestyle='-', linewidth=1,
                 ax=axes)
    #sns.lineplot(x=df.trial, y=df['risultato_centre'], color='gold', label='Center pokes', linestyle='-', linewidth=1,
                 #ax=axes)

    axes.set_yticks([0, 5, 10])

    # Add 50% line on y axes
    y_value = 50  # value
    plt.xlabel('Trial')
    plt.ylabel('n° errors')
    plt.title('n° errors per side (rolling average)')
    #label = f'y = {y_value}'
    #labels = ['l poke', 'r poke', 'c poke']

    plt.legend(loc='upper right', bbox_to_anchor=(0.0, -0.2))


    # Create a PdfPages object to save plots to a PDF file

    pdf_pages = PdfPages(save_path)

    # Save the plot to the PDF
    pdf_pages.savefig()

    # Close the PdfPages object to save the PDF file
    pdf_pages.close()


root_folder = "C:\\academy_reports\\academy_reports\\sessions"

# Recursively search for CSV files
for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith(".csv"):
            print('csv file: ', file)
            path = os.path.join(root, file)
            print('csv path: ', path)
            save_path = path[:-3] + 'pdf'

            #if not "raw" in file and "S1" in file and not os.path.exists(save_path): # requires manually delete the generated pdf in order to generate a new one
            if not "raw" in file and "S1" in file: # overwrite the pdf to the previous one

                daily_report_S1(path, save_path)











