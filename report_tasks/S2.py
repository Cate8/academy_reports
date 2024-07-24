import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
import matplotlib.ticker as ticker


def daily_report_S2(df, save_path, date):

    # replace all nans with 0
    df = df.replace(np.nan, 0)

    unique_boxes = df['box'].unique()
    box = unique_boxes[0]

    # BOX = 9
    # BPOD port5 ->  right
    # BPOD port3 -> (central)
    # BPOD port2 -> left

    # BOX = 12
    # BPOD port7 -> left
    # BPOD port4 -> (central)
    # BPOD port1 -> right

    if box == 9:
        df['left_poke_in'] = df['Port2In_START']
        df['left_poke_out'] = df['Port2Out_START']
        df['center_poke_in'] = df['Port3In_START']
        df['center_poke_out'] = df['Port3Out_START']
        df['right_poke_in'] = df['Port5In_START']
        df['right_poke_out'] = df['Port5Out_START']
    elif box == 12:
        df['left_poke_in'] = df['Port7In_START']
        df['left_poke_out'] = df['Port7Out_START']
        df['center_poke_in'] = df['Port4In_START']
        df['center_poke_out'] = df['Port4Out_START']
        df['right_poke_in'] = df['Port1In_START']
        df['right_poke_out'] = df['Port1Out_START']



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

    df['first_response_right'] = df['right_poke_in'].str.split(',').str[0].astype(float)
    df['first_response_left'] = df['left_poke_in'].str.split(',').str[0].astype(float)



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

    columns_of_interest = ['trial', 'STATE_side_light_END', 'left_poke_in']
    columns_of_interest1 = ['trial','STATE_side_light_END', 'center_poke_in']
    columns_of_interest2 = ['trial','STATE_side_light_END', 'right_poke_in']

        # Crea un nuovo DataFrame con solo le colonne di interesse
    exploded_port2_df = df[columns_of_interest].copy()
    exploded_port3_df = df[columns_of_interest1].copy() if 'center_poke_in' in df else 0
    exploded_port5_df = df[columns_of_interest2].copy()

        # Suddividi le colonne utilizzando la virgola come delimitatore
    exploded_port2_df['left_poke_in'] = df['left_poke_in'].str.split(',')
    exploded_port3_df['center_poke_in'] = df['center_poke_in'].astype(str).str.split(',') if 'center_poke_in' in df else 0
    exploded_port5_df['right_poke_in'] = df['right_poke_in'].str.split(',') if 'right_poke_in' in df else 0

        # Esploa le colonne con liste in righe separate
    exploded_port2_df = exploded_port2_df.explode('left_poke_in')
    exploded_port3_df = exploded_port3_df.explode('center_poke_in')
    exploded_port5_df = exploded_port5_df.explode('right_poke_in')

        #explode le colonne con liste in righe separate
    exploded_port2_df = exploded_port2_df.explode('left_poke_in')
    exploded_port3_df = exploded_port3_df.explode('center_poke_in')
    exploded_port5_df = exploded_port5_df.explode('right_poke_in')

    # replace all nans with 100
    exploded_port2_df = exploded_port2_df.replace(np.nan, 190898697687982)
    exploded_port3_df = exploded_port3_df.replace(np.nan, 190898697687982)
    exploded_port5_df = exploded_port5_df.replace(np.nan, 190898697687982)

    #  'PortIn_START' in float
    exploded_port2_df['left_poke_in'] = pd.to_numeric(exploded_port2_df['left_poke_in'], errors='coerce')
    exploded_port3_df['center_poke_in'] = pd.to_numeric(exploded_port3_df['center_poke_in'], errors='coerce')
    exploded_port5_df['right_poke_in'] = pd.to_numeric(exploded_port5_df['right_poke_in'], errors='coerce')

    # BPOD port5 ->  right
    # BPOD port3 -> (central)
    # BPOD port2 -> left

    #count total pokes in each trial

    # Numero specifico da confrontare
    fake_value = 190898697687982

    # Definire una funzione per contare i pokes
    def count_pokes(value):
        return 1 if value != fake_value else 0


    exploded_port2_df['left_poke'] = exploded_port2_df['left_poke_in'].apply(count_pokes)
    exploded_port3_df['central_poke'] = exploded_port3_df['center_poke_in'].apply(count_pokes)
    exploded_port5_df['right_poke'] = exploded_port5_df['right_poke_in'].apply(count_pokes)

    #count each poke in each trial
    df['total_poke_left'] = exploded_port2_df.groupby(exploded_port2_df.index)['left_poke'].sum()
    df['total_poke_centre'] = exploded_port3_df.groupby(exploded_port3_df.index)['central_poke'].sum()
    df['total_poke_right'] = exploded_port5_df.groupby(exploded_port5_df.index)['right_poke'].sum()

    # comparison to find poke before the correct one
    exploded_port2_df['result'] = np.where(
        exploded_port2_df['left_poke_in'] <= exploded_port2_df['STATE_side_light_END'], 1, 0)
    exploded_port3_df['result'] = np.where(
        exploded_port3_df['center_poke_in'] <= exploded_port3_df['STATE_side_light_END'], 1, 0)
    exploded_port5_df['result'] = np.where(
        exploded_port5_df['right_poke_in'] <= exploded_port5_df['STATE_side_light_END'], 1, 0)

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
             linewidth=2, marker='o', markersize=4)

    axes.set_yticks([0, 50, 100])

    # Add 50% line on y axes
    y_value = 50  # value

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
