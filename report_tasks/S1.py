import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
import matplotlib.ticker as ticker


def daily_report_S1(df, save_path, date):


    df = df.replace(np.nan, 0)

    # create a new column in the DF based on the conditions

    # New columns (variables)
    df['trial_duration'] = df['TRIAL_END'] - df['TRIAL_START']
    df['sum_s_trial_duration'] = df['trial_duration'].sum()
    df['session_duration'] = (df['sum_s_trial_duration'].iloc[0])/ 60
    formatted_session_duration = "{:.2f}".format(df['session_duration'].iloc[0])
    df['duration_drink_delay'] = df['STATE_drink_delay_END'] - df['STATE_drink_delay_START']
    df['duration_waiting_light'] = df['STATE_waiting_light_END'] - df['STATE_waiting_light_START']
    df['duration_water_light']= df['STATE_water_light_END'] - df['STATE_water_light_START']
    df['response_latency'] = df['duration_waiting_light'].copy()

    df['Port5In_START'] = df['Port5In_START'].astype(str)
    df['Port2In_START'] = df['Port2In_START'].astype(str)
    df['first_response_right'] = df['Port5In_START'].str.split(',').str[0].astype(float)
    df['first_response_left'] = df['Port2In_START'].str.split(',').str[0].astype(float)

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
    df['response_latency_median'] = df['response_latency'].median()  # median latency to first response
    # misses: no reward from the sides but they can poke anyway
    max_duration_waiting_light = 300  # max time in side_light (for extrac missed trial)
    df['missed_reward'] = (df['duration_waiting_light'] == max_duration_waiting_light).astype(int)
    count_missed_trials = df['missed_reward'].sum()
    # missed_reward_rate = (df['missed_reward'].iloc[0] / df['trials_max'].iloc[0]) * 100
    # formatted_missed_reward_rate = "{:.2f}".format(missed_reward_rate)

    # omission: no RESPONSE. NO POKE. it's general
    df["omission_bool"] = (df['Port2In_START'] == 0) & (df['Port3In_START'] == 0) & (df['Port5In_START'] == 0)
    df["omission_int"] = df["omission_bool"].astype(int)
    df['omission_sum'] = df["omission_int"].sum()
    tot_omission = df.omission_sum.iloc[0]


    df['tot_correct_choices'] = df['correct_outcome_int'].sum()  # correct choice rate

    df['right_choices'] = (df['side'] == 'right').sum()  # number of right choices

    df['left_choices'] = (df['side'] == 'left').sum()  # number of left choices

    df['water_intake'] = (df['tot_correct_choices'].iloc[
        0]) * 0.01  # microliters each poke, but i want the result in milliliters

    # BPOD port5 ->  right
    # BPOD port3 -> (central)
    # BPOD port2 -> left

    # PLOT PARAMETERS

    mpl.rcParams['lines.linewidth'] = 0.7
    mpl.rcParams['lines.markersize'] = 1
    mpl.rcParams['font.size'] = 7
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False


    plt.figure(figsize=(8, 11))


    # Add session summary text
    session_summary = f'''
      S1 Sessions info
      Date: {df['date'].iloc[0]}, Animal ID: {df['subject'].iloc[0]}, Animal weight: {df['subject_weight'].iloc[0]},Box number: {df['box'].iloc[0]}, Trials: {df['trial'].iloc[-1]}, Session duration: {formatted_session_duration} min, 
      Center latency (median): {str(round(df.response_latency.iloc[1], 2))} s, Side latency (median): {str(round(df.response_latency.iloc[1], 2))} s, Missed (sides): {count_missed_trials}, Omission: {tot_omission}, Missed: {count_missed_trials}, 
      Omission (center): {tot_omission}, Water intake: {df['water_intake'].iloc[0]}ml 
      '''

    plt.figtext(0.00, 0.91, session_summary, fontsize=9)

    # PLOT 1: Outcome (first response = correct outcome)


    summary = f"First poke (side) \n"

    # Mappiamo 'correct_outcome_int' a etichette 'correct' o 'incorrect'
    df['outcome_labels'] = np.where(df['correct_outcome_int'] == 1, 'correct', 'incorrect')

    # Color palette specificata
    color_palette = {
        'no_response': 'gray',
        'right': 'purple',
        'left': 'green',
    }

    # Creazione del grafico
    axes = plt.subplot2grid((50, 50), (1, 0), rowspan=11, colspan=50)
    scatter = sns.scatterplot(x=df.trial, y=df['first_trial_response'], hue=df['outcome_labels'],
                              palette={'correct': 'black', 'incorrect': 'red'}, s=50, ax=axes)

    # Imposta il titolo della legenda
    legend = axes.legend(title="First action")

    # Impostazioni aggiuntive per il titolo e le etichette degli assi
    axes.set_title("First poke (side)")
    plt.xlabel('Trial')
    plt.ylabel('First response')

    # PLOT 3: accuracy on TRIAL

    # left/right licks and accuracy in a window of trials called "rolling average"

    # Calculate moving averages for accuracy, left poke count and right poke count

    df["accuracy"] = ((df['correct_outcome_int'] / df['trial']) * 100).astype(float)

    # Define a function to compute a rolling window statistic
    def compute_window(column, group_size):

        return column.rolling(window=5, min_periods=1).mean()

    axes = plt.subplot2grid((50, 50), (15, 0), rowspan=11, colspan=50)


    df['rolling_accuracy'] = compute_window(df["correct_outcome_int"], 5) * 100

    plt.plot(df.trial, df.rolling_accuracy, color='blueviolet', linestyle='-',
             linewidth=2, marker='o', markersize = 4)

    axes.set_yticks([0, 50, 100])

    # Add 50% line on y axes
    y_value = 50 # value

    plt.axhline(y=y_value, color='black', linestyle='--')
    plt.xlabel('Trial')
    plt.ylabel('%')
    plt.title('Trials accuracy (rolling average)')

    # PLOT 2: latency to the first correct poke

    axes = plt.subplot2grid((50, 50), (30, 0), rowspan=11, colspan=50)

    plt.plot(df.trial, df.response_latency, color='dodgerblue', linewidth=2, markersize=8)

    # Personalizzazione dei ticks sull'asse y
    axes.set_yscale('log')
    axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    plt.title('Latency to poke')
    plt.xlabel('Trial')
    plt.ylabel('Latency (s)')

    custom_yticks = [1, 10, 100]  # I valori che desideri mostrare sull'asse y
    for y in custom_yticks:
        plt.axhline(y=y, linestyle='--', color='lightgray', alpha=0.7)

    pdf_pages = PdfPages(save_path)

    # Save the plot to the PDF
    pdf_pages.savefig()

    # Close the PdfPages object to save the PDF file
    pdf_pages.close()










