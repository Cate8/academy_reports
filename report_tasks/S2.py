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


def daily_report_S2(df, save_path, date):

    # replace all nans with 0
    df = df.replace(np.nan, 0)


    # New columns (variables)
    df['trial_duration'] = df['TRIAL_END'] - df['TRIAL_START']
    df['sum_s_trial_duration'] = df['trial_duration'].sum()
    df['session_duration'] = (df['sum_s_trial_duration'].iloc[0]) / 60
    formatted_session_duration = "{:.2f}".format(df['session_duration'].iloc[0])
    df['side_light_duration'] = df['STATE_side_light_END'] - df['STATE_side_light_START']
    df['duration_drink_delay'] = df['STATE_drink_delay_END'] - df['STATE_drink_delay_START']
    df['duration_water_delivery'] = df['STATE_water_delivery_END'] - df['STATE_water_delivery_START']
    df['response_latency'] = df['side_light_duration'].copy()

    # BPOD port5 ->  right
    # BPOD port3 -> (central)
    # BPOD port2 -> left

    df = df.replace(np.nan, 0)
    # BPOD port5 ->  right
    # BPOD port3 -> (central)
    # BPOD port2 -> left

    df['first_response_right'] = df['Port5In_START'].str.split(',').str[0].astype(float)
    df['first_response_left'] = df['Port2In_START'].str.split(',').str[0].astype(float)



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

    df["correct_outcome_bool"] = df["first_trial_response"] == df[
        "side"]  # this is for having boolean variables (true/false)
    df['true_count'] = (df['correct_outcome_bool']).value_counts()[True]
    df["correct_outcome"] = np.where(df["first_trial_response"] == df["side"], "correct",
                                     "incorrect")  # (true = correct choice, false = incorrect side)
    df["correct_outcome_int"] = np.where(df["first_trial_response"] == df["side"], 1,
                                         0)  # (1 = correct choice, 0= incorrect side)

    # Other calculations and Averaging

    df['response_latency_median'] = df['response_latency'].median()  # average latency to first response

    max_duration_side_light = 300.00000  # max time in side_light (for extrac missed trial)
    df['misses'] = (df['side_light_duration'] == max_duration_side_light).astype(int)
    count_missed_trials = df['misses'].sum()
    missed_rate = (df['misses'].iloc[0] / df['trials_max'].iloc[0]) * 100
    formatted_missed_rate = "{:.2f}".format(missed_rate)

    df['tot_correct_choices'] = df['correct_outcome_int'].sum()  # correct choice rate
    correct_choice_rate = (df['tot_correct_choices'].iloc[0] / df['trials_max'].iloc[0]) * 100
    formatted_correct_choice_rate = "{:.2f}".format(correct_choice_rate)

    df['right_choices'] = (df['side'] == 'right').sum()  # number of right choices
    right_choice_rate = (df['right_choices'].iloc[0] / df['trials_max'].iloc[0]) * 100
    formatted_right_choice_rate = "{:.2f}".format(right_choice_rate)

    df['left_choices'] = (df['side'] == 'left').sum()  # number of left choices
    left_choice_rate = (df['left_choices'].iloc[0] / df['trials_max'].iloc[0]) * 100
    formatted_left_choice_rate = "{:.2f}".format(left_choice_rate)

    df['session_duration_s'] = df['TRIAL_END'].iloc[:] - df['TRIAL_START'].iloc[0]


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

    right_rate= (poke_df['total_n_poke_right'].iloc[0] / poke_df['total_poke'].iloc[0]) * 100
    left_rate = (poke_df['total_n_poke_left'].iloc[0] / poke_df['total_poke'].iloc[0]) * 100
    centre_rate = (poke_df['total_n_poke_centre'].iloc[0] / poke_df['total_poke'].iloc[0]) * 100

    formatted_right_rate = "{:.2f}".format(right_rate)
    formatted_center_rate = "{:.2f}".format(centre_rate)
    formatted_left_rate = "{:.2f}".format(left_rate)

    # PLOT PARAMETERS

    mpl.rcParams['lines.linewidth'] = 0.7
    mpl.rcParams['lines.markersize'] = 1
    mpl.rcParams['font.size'] = 7
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    label_kwargs = {'fontsize': 9, 'fontweight': 'roman'}

    plt.figure(figsize=(8, 11))

    # PLOT COLORS:

    left_c = 'aquamarine'
    right_c = 'orange'
    correct_c = 'green'
    incorrect_c = 'red'
    label_kwargs = {'fontsize': 9}

    # Add session summary text
    session_summary = f'''
    S2 Sessions info
    Date: {df['date'].iloc[0]}, Animal ID: {df['subject'].iloc[0]}, Animal weight: {df['subject_weight'].iloc[0]},Box number: {df['box'].iloc[0]}, Trials: {df['trial'].iloc[-1]}, 
    Session duration: {formatted_session_duration} min, Response latency (median): {str(round(df.response_latency_median.iloc[1], 2))} s, Missed: {count_missed_trials} ({formatted_missed_rate}%), Water intake: {df['water_intake'].iloc[0]}ml,
    Right choice rate: ({formatted_right_rate}%), Left choice rate: ({formatted_left_rate}%), Center choice rate: ({formatted_center_rate}%), Total pokes: {poke_df['total_poke'].iloc[0]},
    (R: {poke_df['total_n_poke_right'].iloc[0]}, C: {poke_df['total_n_poke_centre'].iloc[0]}, L: {poke_df['total_n_poke_left'].iloc[0]})'''

    plt.figtext(0.0, 0.93, session_summary, fontsize=10)

    # FIRST PLOT: latency to the first correct poke

    plot_summary = f"Latency to the first correct poke\n"

    axes = plt.subplot2grid((1000, 50), (1, 1), rowspan=100, colspan=90)

    sns.scatterplot(x=df.trial, y=df.response_latency, hue=df.side, hue_order=['left', 'right'],
                    palette=[left_c, right_c], s=10, ax=axes)

    sns.lineplot(x=df.trial, y=df.response_latency, hue=df.side, hue_order=['left', 'right'],
                 palette=[left_c, right_c], ax=axes)

    axes.set_title(plot_summary)

    # SECOND PLOT: Outcome (first response = correct outcome)

    summary = f"First poke (side) \n"

    color_palette = {
        'no_response': 'gray',
        'right': 'lightseagreen',
        'left': 'orchid',
    }

    axes = plt.subplot2grid((1600, 50), (310, 1), rowspan=110, colspan=90)

    # Combina i dati da entrambe le colonne nel grafico a dispersione
    sns.scatterplot(x=df.trial, y=df['first_trial_response'], hue=df['first_trial_response'],
                    palette=color_palette, s=30, ax=axes)

    axes.set_title(summary)

    # THIRD PLOT: Outcome (first response = correct outcome)

    summary = f"Outcome\n"

    axes = plt.subplot2grid((1600, 50), (550, 1), rowspan=110, colspan=90)

    sns.scatterplot(x=df.trial, y=df.side, hue=df.correct_outcome, hue_order=['correct', 'incorrect'],
                    palette=[correct_c, incorrect_c], s=30, ax=axes)

    axes.set_title(summary)

    # THIRD PLOT: session's misses over trials

    summary = f"Misses over trials\n"

    axes = plt.subplot2grid((1600, 50), (800, 1), rowspan=110, colspan=90)
    missed_trials_df = df[df['misses'] == 1]

    sns.scatterplot(x=df['trial'], y=df['misses'], s=30, ax=axes,
                    color=['darkorange' if x == 1 else 'royalblue' for x in df['misses']])
    axes.set_yticks([0, 1])

    axes.set_title(summary)
    # FORTH PLOT: accuracy
    # FORTH PLOT: accuracy on TRIAL
    # left/right licks and accuracy in a window of trials called "rolling average"

    # Calculate moving averages for accuracy, left poke count and right poke count

    df["accuracy"] = ((df['correct_outcome_int'] / df['trial']) * 100).astype(float)


    # Define a function to compute a rolling window statistic
    def compute_window(column, group_size):
        return column.rolling(window=5, min_periods=1).mean()

    axes = plt.subplot2grid((1600, 50), (1025, 1), rowspan=150, colspan=90)

    df['rolling_accuracy'] = compute_window(df["correct_outcome_int"], 5) * 100
    plt.scatter(df.index, df.rolling_accuracy, label='choice accuracy', color='blueviolet', marker='o', s=5)
    plt.plot(df.index, df.rolling_accuracy, color='blueviolet', linestyle='-', linewidth=1)

    axes.set_yticks([0, 50, 100])

    # Add 50% line on y axes
    y_value = 50  # value

    plt.axhline(y=y_value, color='black', linestyle='--')
    plt.xlabel('Trial')
    plt.ylabel('%')
    plt.title('Trials accuracy (rolling average)')
    plt.legend()

    #FORTH PLOT: trial distribution over time
    columns_of_interest = df[['trial', 'side_light_duration', 'duration_water_delivery', 'duration_drink_delay']]
    df_trial_subgroup = columns_of_interest.copy()
    df_trial_subgroup = pd.melt(df_trial_subgroup, id_vars=['trial'], var_name='phase', value_name='duration')
    df_trial_subgroup['cumulative_duration'] = df_trial_subgroup.groupby(['trial', 'phase'])['duration'].cumsum()

    # Crea il grafico a barre cumulativo
    axes = plt.subplot2grid((1600, 100), (1300, 1), rowspan=700, colspan=50)

    # Crea una mappa dei colori per le fasi
    color_map = {
        'side_light_duration': 'lightpink',
        'duration_water_delivery': 'black',
        'duration_drink_delay': 'lightskyblue'
    }

    # Calcola la somma cumulativa per ciascuna fase separatamente
    df_trial_subgroup['cumulative_duration'] = df_trial_subgroup.groupby(['trial', 'phase'])['duration'].cumsum()

    #axes.set_xscale('log')  # Usa la scala logaritmica sull'asse x


    legend_labels = []  # Lista per le etichette della legenda
    legend_handles = []  # Lista per gli oggetti di legenda

    for trial in df_trial_subgroup['trial'].unique():
        trial_data = df_trial_subgroup[df_trial_subgroup['trial'] == trial]
        for phase in df_trial_subgroup['phase'].unique():
            phase_data = trial_data[trial_data['phase'] == phase]
            if phase == 'side_light_duration' and phase_data['duration'].max() == 300:
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


    pdf_pages = PdfPages(save_path)

    # Save the plot to the PDF
    pdf_pages.savefig()

    # Close the PdfPages object to save the PDF file
    pdf_pages.close()

