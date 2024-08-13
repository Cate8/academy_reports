
from datetime import timedelta
from matplotlib.lines import Line2D
from scipy import stats
from academy_reports import utils
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy.special import erf
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import os


def intersession_opto(csv_path, pdf_path):
    print(csv_path)
    df = pd.read_csv(csv_path, sep=';')  # Modifica il separatore se necessario

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


    # PLOT COLORS & FONT
    label_kwargs = {'fontsize': 9}
    lines_c = 'gray'


    ##### SELECT LAST MONTH SESSIONS #####
    df['day'] = pd.to_datetime(df['date']).dt.date
    df = df.loc[df['day'] > df.day.max() - timedelta(days=20)]

    ###### RELEVANT COLUMNS ######

    #JUST OPTO SESSIONS:
    tasks_opto = ['S4_5_train_pulse', 'S4_5_single_pulse']
    df = df[df['task'].isin(tasks_opto)]

    # weights
    df['relative_weights'] = df.apply(lambda x: utils.relative_weights(x['subject'], x['subject_weight']), axis=1)

    # latencies & times
    df['center_response_time'] = df['STATE_center_light_END'] - df['STATE_center_light_START']
    df['response_time'] = df['STATE_side_light_END'] - df['STATE_side_light_START']
    df['duration_drink_delay'] = df['STATE_drink_delay_END'] - df['STATE_drink_delay_START']
    df['centre_response_latency'] = df['center_response_time']
    df['Port5In_START'] = df['right_poke_in'].astype(str)
    df['Port2In_START'] = df['left_poke_in'].astype(str)
    df['first_response_right'] = df['right_poke_in'].str.split(',').str[0].astype(float)
    df['first_response_left'] = df['left_poke_in'].str.split(',').str[0].astype(float)
    df['center_median_response_time'] = df['centre_response_latency'].median()  # median latency to first response
    df['response_latency_median'] = df['response_time'].median()  # median latency to first response
    df['probability_r'] = np.round(df['probability_r'], 1)
    # Probability calculations
    df['probability_l'] = 1 - df['probability_r']
    df['probability_r'] = df['probability_r'].astype(float)
    df['prev_iti_duration'] = df['iti_duration'].shift(1)


    # teat well the NANs and  List of conditions for knowing which one was the first choice in each trial
    df = df.replace(np.nan, 0)
    # List of conditions for teat well the NANs
    conditions = [
        (df.first_response_left == 0) & (df.first_response_right == 0),
        df.first_response_left == 0,
        df.first_response_right == 0,
        df.first_response_left <= df.first_response_right,
        df.first_response_left > df.first_response_right,
    ]
    choices = ["no_response",
               "right",
               "left",
               "left",
               "right"]
    df["first_trial_response"] = np.select(conditions, choices)

    df["correct_outcome_bool"] = df["first_trial_response"] == df["side"]
    df["correct_outcome_int"] = np.where(df["first_trial_response"] == df["side"], 1,
                                         0)  # (1 = correct choice, 0= incorrect side)

    # Probability calculations
    df['probability_l'] = 1 - df['probability_r']
    df['probability_r'] = df['probability_r'].astype(float)
    df['first_right_trial_response'] = df['first_trial_response'].apply(lambda x: 1 if x == 'right' else 0)


    ############# PAGE 1 ##############
    with PdfPages(save_path) as pdf:
        plt.figure(figsize=(11.7, 15))

        ####SUMMARY#####
        session_summary = f'''
        {df['subject'].iloc[0]}
         '''
        plt.figtext(0.4, 0.95, session_summary, fontweight='bold', verticalalignment='top', fontsize=20)


        ### PLOT 1: NUMBER OF TRIALS
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=5, colspan=23)
        grouped_df = df.groupby('session').agg({'trial': "max", 'day': "max", 'task': "max"}).reset_index()

        sns.lineplot(x='day', y='trial', style='task', markers=True, ax=axes, color='black',
                     data=grouped_df, estimator=np.average, errorbar=None)
        axes.axhline(y=100, color=lines_c, linestyle=':', linewidth=1)
        axes.set_ylabel('Nº of trials (µ)', label_kwargs)
        axes.set_ylim(0, 450)
        axes.set_xlabel('')
        axes.xaxis.set_ticklabels([])

        ### PLOT 2: N of opto trials within a session
        axes = plt.subplot2grid((50, 50), (0, 27), rowspan=5, colspan=24)

        # Raggruppa per giorno e conta il numero di trial con opto
        opto_trials = df.groupby('day')['opto_bool'].sum().reset_index()

        # Visualizza i dati
        sns.lineplot(x='day', y='opto_bool', marker='o', color='black', data=opto_trials, ax=axes)

        axes.axhline(y=100, color=lines_c, linestyle=':', linewidth=1)
        axes.set_ylabel('Opto trial (n)', label_kwargs)
        axes.set_ylim(0, 150)
        axes.set_xlabel('')
        axes.xaxis.set_ticklabels([])

        ### PLOT 4: LATENCIES: centre motor time
        axes = plt.subplot2grid((50, 50), (6, 0), rowspan=5, colspan=24)
        df_light_off = df[(df['iti_duration'] >= 6) & (df['opto_bool'] == 0)]
        filtered_df = df_light_off[df_light_off['response_time'].notnull() & df_light_off['center_response_time'].notnull()]

        sns.lineplot(x='day', y='center_response_time', data=filtered_df, estimator=np.median, errorbar=None, ax=axes,
                     color='black', markers=True, alpha=0.5)

        df_light_on = df[(df['iti_duration'] >= 6) & (df['opto_bool'] == 1)]
        sns.lineplot(x='day', y='center_response_time', data=df_light_on, estimator=np.median, errorbar=None, ax=axes,
                     color='gold', markers=True, alpha=0.5)

        axes.set_yscale('log')
        axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        custom_yticks = [0, 1, 10]
        for y in custom_yticks:
            axes.axhline(y=y, linestyle='--', color='lightgray', alpha=0.7)
        axes.tick_params(axis='x', rotation=45)
        axes.set_xlabel('Day')
        axes.set_ylabel('centre response time (me)')
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        legend_elements = [Line2D([0], [0], color='gold', lw=2, label='on', alpha=0.5),
                            Line2D([0], [0], color='black', lw=2, label='off', alpha=0.5)]
        axes.legend(handles=legend_elements, loc='upper right')

        ### PLOT 4: LATENCIES: side motor time
        axes = plt.subplot2grid((50, 50), (6, 27), rowspan=5, colspan=24)
        sns.lineplot(x='day', y='response_time', data=filtered_df, estimator=np.median, errorbar=None,
                     ax=axes,
                     color='black', markers=True, alpha=0.5)

        sns.lineplot(x='day', y='response_time', data=df_light_on, estimator=np.median, errorbar=None,
                     ax=axes,
                     color='gold', markers=True, alpha=0.5)


        axes.set_yscale('log')
        axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        custom_yticks = [0, 1, 10]
        for y in custom_yticks:
            axes.axhline(y=y, linestyle='--', color='lightgray', alpha=0.7)
        axes.tick_params(axis='x', rotation=45)
        axes.set_xlabel('Day')
        axes.set_ylabel('Motor response time (me)')
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        #5 PLOT: PC choice right vs.probability type
        axes = plt.subplot2grid((50, 50), (14, 0), rowspan=12, colspan=14)

        # Define the probit function
        def probit(x, beta, alpha):
            return 0.5 * (1 + erf((beta * x + alpha) / np.sqrt(2)))

        # Preparazione dei dati
        df_light_off = df_light_off.copy()  # Evita SettingWithCopyWarning
        df_light_off.loc[:, 'probability_l'] = 1 - df_light_off['probability_r']
        df_light_off.loc[:, 'probability_r'] = df_light_off['probability_r'].astype(float)
        df_light_off.loc[:, 'first_trial_response'] = df_light_off['first_trial_response'].apply(
            lambda x: 1 if x == 'right' else 0)

        df_light_on = df_light_on.copy()  # Evita SettingWithCopyWarning
        df_light_on.loc[:, 'probability_r'] = df_light_on['probability_r'].astype(float)
        df_light_on.loc[:, 'first_trial_response'] = df_light_on['first_trial_response'].apply(
            lambda x: 1 if x == 'right' else 0)

        # Calcolo delle frequenze delle scelte a destra per df_light_off
        probs_off = np.sort(df_light_off['probability_r'].unique())
        right_choice_freq_off = []

        for prob in probs_off:
            indx_blk = df_light_off['probability_r'] == prob
            sum_prob = np.sum(indx_blk)
            if sum_prob > 0:
                indx_blk_r = indx_blk & (df_light_off['first_trial_response'] == 1)
                sum_choice_prob = np.sum(indx_blk_r)
                right_choice_freq_off.append(sum_choice_prob / sum_prob)
            else:
                right_choice_freq_off.append(0)

        # Calcolo delle frequenze delle scelte a destra per df_light_on
        probs_on = np.sort(df_light_on['probability_r'].unique())
        right_choice_freq_on = []

        for prob in probs_on:
            indx_blk = df_light_on['probability_r'] == prob
            sum_prob = np.sum(indx_blk)
            if sum_prob > 0:
                indx_blk_r = indx_blk & (df_light_on['first_trial_response'] == 1)
                sum_choice_prob = np.sum(indx_blk_r)
                right_choice_freq_on.append(sum_choice_prob / sum_prob)
            else:
                right_choice_freq_on.append(0)

        # Adatta le curve probit
        pars_off, _ = curve_fit(probit, df_light_off['probability_r'], df_light_off['first_trial_response'], p0=[0, 1])
        pars_on, _ = curve_fit(probit, df_light_on['probability_r'], df_light_on['first_trial_response'], p0=[0, 1])

        # Impostazioni del grafico

        x = np.linspace(0, 1, 100)

        # Disegna le curve di adattamento probit
        axes.plot(x, probit(x, *pars_off), label='Off', color='black', alpha=0.5)
        axes.plot(x, probit(x, *pars_on), label='On', color='gold', alpha=0.5)

        # Disegna lo scatter plot delle frequenze calcolate per Light Off
        axes.scatter(probs_off, right_choice_freq_off, marker='o', color='black',
                    label='Off', alpha=0.5)

        # Disegna lo scatter plot delle frequenze calcolate per Light On
        axes.scatter(probs_on, right_choice_freq_on, marker='o', color='gold', label='On', alpha=0.5)

        # Impostazioni aggiuntive del grafico
        axes.set_ylim(0, 1)
        axes.axhline(0.5, color='gray', linestyle='--', alpha =0.3)
        axes.axvline(0.5, color='gray', linestyle='--', alpha =0.3)
        axes.set_xlabel('Probability Type')
        axes.set_ylabel('Right Choice Rate')


        # #6 PLOT: PC choose highest reward vs.probability type
        axes = plt.subplot2grid((50, 50), (14, 18), rowspan=12, colspan=14)

        # Copia dei DataFrame
        df_light_off = df_light_off.copy()
        df_light_on = df_light_on.copy()

        # Calcolo delle risposte corrette per df_light_off
        df_light_off['off_correct_response'] = np.where(
            (df_light_off['probability_r'] > 0.5) & (df_light_off['first_right_trial_response'] == 1), 1,
            np.where((df_light_off['probability_r'] < 0.5) & (df_light_off['first_right_trial_response'] == 0), 1, 0)
        )

        # Calcolo delle risposte corrette per df_light_on
        df_light_on['on_correct_response'] = np.where(
            (df_light_on['probability_r'] > 0.5) & (df_light_on['first_right_trial_response'] == 1), 1,
            np.where((df_light_on['probability_r'] < 0.5) & (df_light_on['first_right_trial_response'] == 0), 1, 0)
        )

        # Calcolo delle frequenze delle scelte corrette per df_light_off
        probs_off = np.sort(df_light_off['probability_r'].unique())
        off_correct_response = []

        for prob in probs_off:
            indx_blk = df_light_off['probability_r'] == prob
            sum_prob = np.sum(indx_blk)
            if sum_prob > 0:
                indx_blk_r = indx_blk & (df_light_off['off_correct_response'] == 1)
                sum_choice_prob = np.sum(indx_blk_r)
                off_correct_response.append(sum_choice_prob / sum_prob)
            else:
                off_correct_response.append(0)

        # Calcolo delle frequenze delle scelte corrette per df_light_on
        probs_on = np.sort(df_light_on['probability_r'].unique())
        on_correct_response = []

        for prob in probs_on:
            indx_blk = df_light_on['probability_r'] == prob
            sum_prob = np.sum(indx_blk)
            if sum_prob > 0:
                indx_blk_r = indx_blk & (df_light_on['on_correct_response'] == 1)
                sum_choice_prob = np.sum(indx_blk_r)
                on_correct_response.append(sum_choice_prob / sum_prob)
            else:
                on_correct_response.append(0)

        # Adattamento delle curve probit
        pars_off, _ = curve_fit(probit, probs_off, off_correct_response, p0=[0, 1])
        pars_on, _ = curve_fit(probit, probs_on, on_correct_response, p0=[0, 1])

        # Impostazioni del grafico
        x = np.linspace(0, 1, 100)


        # Disegna le curve di adattamento probit
        axes.plot(x, probit(x, *pars_off), label='Off', color='black', alpha=0.5)
        axes.plot(x, probit(x, *pars_on), label='On', color='gold', alpha=0.5)

        # Disegna lo scatter plot delle frequenze calcolate per Light Off
        axes.scatter(probs_off, off_correct_response, marker='o', color='black', label='Off', alpha=0.5)

        # Disegna lo scatter plot delle frequenze calcolate per Light On
        axes.scatter(probs_on, on_correct_response, marker='o', color='gold', label='On', alpha=0.5)

        # Impostazioni aggiuntive del grafico
        axes.set_ylim(0, 1)
        axes.axhline(0.5, color='gray', linestyle='--', alpha =0.3)
        axes.axvline(0.5, color='gray', linestyle='--', alpha =0.3)
        axes.set_xlabel('Probability Type')
        axes.set_ylabel('Correct choice Rate')

        #### 7 PLOT: CUMULATIVE TRIAL RATE
        axes = plt.subplot2grid((50, 50), (14, 36), rowspan=12, colspan=15)

        df['start_session'] = df.groupby(['subject', 'session'])['STATE_center_light_START'].transform('min')
        df['end_session'] = df.groupby(['subject', 'session'])['STATE_drink_delay_END'].transform('max')
        df['session_lenght'] = (df['end_session'] - df['start_session']) / 60
        df['current_time'] = df.groupby(['subject', 'session'])['STATE_center_light_START'].transform(
            lambda x: (x - x.iloc[0]) / 60)  # MINS

        max_timing = round(df['session_lenght'].max())
        max_timing = int(max_timing)
        sess_palette = sns.color_palette('Greens', 5)  # color per day

        for idx, day in enumerate(df.day.unique()):
            subset = df.loc[df['day'] == day]
            n_sess = len(subset.session.unique())
            try:
                hist_ = stats.cumfreq(subset.current_time, numbins=max_timing,
                                      defaultreallimits=(0, subset.current_time.max()), weights=None)
            except:
                hist_ = stats.cumfreq(subset.current_time, numbins=max_timing, defaultreallimits=(0, max_timing),
                                      weights=None)
            hist_norm = hist_.cumcount / n_sess
            bins_plt = hist_.lowerlimit + np.linspace(0, hist_.binsize * hist_.cumcount.size, hist_.cumcount.size)
            sns.lineplot(x=bins_plt, y=hist_norm, color=sess_palette[idx % len(sess_palette)], ax=axes, marker='o',
                         markersize=4)

        axes.set_ylabel('Cum. nº of trials', label_kwargs)
        axes.set_xlabel('Time (mins)', label_kwargs)

        # legend
        lines = [Line2D([0], [0], color=sess_palette[i], marker='o', markersize=7, markerfacecolor=sess_palette[i]) for
                 i in
                 range(len(sess_palette))]
        axes.legend(lines, np.arange(-5, 0, 1), title='Days', loc='lower right', bbox_to_anchor=(1, 0))

        #PLOT 8: PCs vs ITI -> 6-10 s ITI
        def probit(x, beta, alpha):
            """
            Return probit function with parameters alpha and beta.

            Parameters
            ----------
            x : float
                independent variable.
            beta : float
                sensitivity term. Sensitivity term corresponds to the slope of the psychometric curve.
            alpha : TYPE
                bias term. Bias term corresponds to the shift of the psychometric curve along the x-axis.

            Returns
            -------
            probit : float
                probit value for the given x, beta and alpha.
            """
            return 0.5 * (1 + erf((beta * x + alpha) / np.sqrt(2)))

        axes = plt.subplot2grid((50, 50), (30, 0), rowspan=12, colspan=14)

        # Copy of DataFrames
        df_light_off = df_light_off.copy()
        df_light_on = df_light_on.copy()

        # Add ITI categories
        df_light_off['prev_iti_category'] = pd.cut(df_light_off['prev_iti_duration'], 3,
                                                   labels=["6-10 sec", "11-15 sec", '16-30 sec'])

        df_light_on['prev_iti_category'] = pd.cut(df_light_on['prev_iti_duration'], 3,
                                                  labels=["6-10 sec", "11-15 sec", '16-30 sec'])

        # Filter data for the 6-10 sec ITI bin
        df_light_off_bin = df_light_off[df_light_off['prev_iti_category'] == "6-10 sec"]
        df_light_on_bin = df_light_on[df_light_on['prev_iti_category'] == "6-10 sec"]

        # Calculate correct choice rate for each condition
        def correct_choice_rate(df):
            df['correct_choice'] = np.where((df['probability_r'] > 0.5) & (df['first_right_trial_response'] == 1), 1,
                                            np.where((df['probability_r'] < 0.5) &
                                                     (df['first_right_trial_response'] == 1), 1, 0))
            return df.groupby('probability_r')['correct_choice'].mean()

        correct_choice_rate_off = correct_choice_rate(df_light_off_bin)
        correct_choice_rate_on = correct_choice_rate(df_light_on_bin)

        # Create a plot
        x = np.linspace(0, 1, 100)

        # Ensure probs_off and probs_on are defined
        probs_off = correct_choice_rate_off.index.values
        probs_on = correct_choice_rate_on.index.values

        pars_off, _ = curve_fit(probit, probs_off, correct_choice_rate_off, p0=[0, 1])
        pars_on, _ = curve_fit(probit, probs_on, correct_choice_rate_on, p0=[0, 1])

        # Disegna le curve di adattamento probit
        axes.plot(x, probit(x, *pars_off), label='Off', color='black', alpha=0.5)
        axes.plot(x, probit(x, *pars_on), label='On', color='gold', alpha=0.5)

        # Disegna lo scatter plot delle frequenze calcolate per Light Off
        axes.scatter(probs_off, correct_choice_rate_off, marker='o', color='black', label='Off', alpha=0.5)

        # Disegna lo scatter plot delle frequenze calcolate per Light On
        axes.scatter(probs_on, correct_choice_rate_on, marker='o', color='gold', label='On', alpha=0.5)

        plt.xlabel('Probability Type')
        plt.ylabel('Correct Choice Rate')
        plt.title('6-10 sec ITI')
        plt.legend()

        # Imposta i ticks personalizzati per l'asse x
        plt.xticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
                   labels=['0.1', '0.2', '0.3', '0.4', '0.6', '0.7', '0.8', '0.9'])

        # Imposta i ticks personalizzati per l'asse y
        plt.yticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
                   labels=['0.1', '0.2', '0.3', '0.4', '0.6', '0.7', '0.8', '0.9'])

        # Add legend with title
        axes.axhline(0.5, color='gray', linestyle='--', alpha =0.3)
        axes.axvline(0.5, color='gray', linestyle='--', alpha =0.3)
        plt.legend(title='ITI Duration')
        plt.ylim(0, 1)

        #PLOT 8: PCs vs ITI 11-15 sec
        # Probit function
        def probit(x, beta, alpha):
            """
            Return probit function with parameters alpha and beta.

            Parameters
            ----------
            x : float
                independent variable.
            beta : float
                sensitivity term. Sensitivity term corresponds to the slope of the psychometric curve.
            alpha : float
                bias term. Bias term corresponds to the shift of the psychometric curve along the x-axis.

            Returns
            -------
            probit : float
                probit value for the given x, beta and alpha.
            """
            return 0.5 * (1 + erf((beta * x + alpha) / np.sqrt(2)))

        # Create axes for plotting
        axes = plt.subplot2grid((50, 50), (30, 18), rowspan=12, colspan=14)

        # Copy of DataFrames
        df_light_off = df_light_off.copy()
        df_light_on = df_light_on.copy()

        # Add ITI categories
        df_light_off['prev_iti_category'] = pd.cut(df_light_off['prev_iti_duration'], [6, 10, 15, 30],
                                                   labels=["6-10 sec", "11-15 sec", '16-30 sec'])
        df_light_on['prev_iti_category'] = pd.cut(df_light_on['prev_iti_duration'], [6, 10, 15, 30],
                                                  labels=["6-10 sec", "11-15 sec", '16-30 sec'])

        # Filter data for the 11-30 sec ITI bin
        df_light_off_bin = df_light_off[df_light_off['prev_iti_category'].isin(["11-15 sec", '16-30 sec'])]
        df_light_on_bin = df_light_on[df_light_on['prev_iti_category'].isin(["11-15 sec", '16-30 sec'])]

        # Calculate correct choice rate for each condition
        def correct_choice_rate(df):
            df = df.copy()
            df['correct_choice'] = np.where((df['probability_r'] > 0.5) & (df['first_right_trial_response'] == 1), 1,
                                            np.where((df['probability_r'] < 0.5) &
                                                     (df['first_right_trial_response'] == 1), 1, 0))
            return df.groupby('probability_r')['correct_choice'].mean()

        correct_choice_rate_off = correct_choice_rate(df_light_off_bin)
        correct_choice_rate_on = correct_choice_rate(df_light_on_bin)

        # Ensure probs_off and probs_on are defined
        probs_off = correct_choice_rate_off.index.values
        probs_on = correct_choice_rate_on.index.values

        # Create a plot
        x = np.linspace(0, 1, 100)

        # Fit the probit curve
        pars_off, _ = curve_fit(probit, probs_off, correct_choice_rate_off, p0=[0, 1])
        pars_on, _ = curve_fit(probit, probs_on, correct_choice_rate_on, p0=[0, 1])

        # Draw the probit fit curves
        axes.plot(x, probit(x, *pars_off), label='Off', color='black', alpha=0.5)
        axes.plot(x, probit(x, *pars_on), label='On', color='gold', alpha=0.5)

        # Draw scatter plots of the calculated frequencies for Light Off
        axes.scatter(probs_off, correct_choice_rate_off, marker='o', color='black', label='Off', alpha=0.5)

        # Draw scatter plots of the calculated frequencies for Light On
        axes.scatter(probs_on, correct_choice_rate_on, marker='o', color='gold', label='On', alpha=0.5)

        plt.xlabel('Probability Type')
        plt.ylabel('Correct Choice Rate')
        plt.title('11-30 sec ITI')
        plt.legend()

        # Set custom ticks for x-axis
        plt.xticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
                   labels=['0.1', '0.2', '0.3', '0.4', '0.6', '0.7', '0.8', '0.9'])

        # Set custom ticks for y-axis
        plt.yticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
                   labels=['0.1', '0.2', '0.3', '0.4', '0.6', '0.7', '0.8', '0.9'])

        # Add legend with title
        plt.legend(title='ITI Duration')
        axes.axhline(0.5, color='gray', linestyle='--', alpha =0.3)
        axes.axvline(0.5, color='gray', linestyle='--', alpha =0.3)
        plt.ylim(0, 1)


        # #### penultimo PLOT : REWARD HISTORY WEIGHT
        # # Prepare df columns
        # # Converting the 'outcome' column to boolean values
        # select_columns = ['trial', 'session', 'outcome', 'side']  # Usa una lista per i nomi delle colonne
        # df_glm = df.loc[:, select_columns].copy()
        #
        # df_glm['outcome_bool'] = np.where(df_glm['outcome'] == "correct", 1, 0)
        #
        # # conditions to determine the choice of each trial:
        # # if outcome "0" & side "right", choice "left" ;
        # # if outcome "1" & side "left", choice "left" ;
        # # if outcome "0" & side "left", choice "right" ;
        # # if outcome "1" & side "right", choice "right";
        # # define conditions
        # conditions = [
        #     (df_glm['outcome_bool'] == 0) & (df_glm['side'] == 'right'),
        #     (df_glm['outcome_bool'] == 1) & (df_glm['side'] == 'left'),
        #     (df_glm['outcome_bool'] == 0) & (df_glm['side'] == 'left'),
        #     (df_glm['outcome_bool'] == 1) & (df_glm['side'] == 'right'),
        # ]
        #
        # choice = [
        #     'left',
        #     'left',
        #     'right',
        #     'right'
        # ]
        #
        # df_glm['choice'] = np.select(conditions, choice, default='other')
        #
        # # calculate correct_choice regressor L+
        # # if outcome_bool 0,  L+: incorrect (0)
        # # if outcome_bool 1, choice "right", L+: correct (1) because right,
        # # if outcome bool 1, choice "left", L+: correct (-1) because left,
        #
        # # define conditions
        # conditions = [
        #     (df_glm['outcome_bool'] == 0),
        #     (df_glm['outcome_bool'] == 1) & (df_glm['choice'] == 'right'),
        #     (df_glm['outcome_bool'] == 1) & (df_glm['choice'] == 'left'),
        # ]
        #
        # r_plus = [
        #     0,
        #     1,
        #     -1,
        # ]
        #
        # df_glm['r_plus'] = np.select(conditions, r_plus, default='other')
        # df_glm['r_plus'] = pd.to_numeric(df_glm['r_plus'], errors='coerce')
        #
        # # calculate wrong_choice regressor L- (1 correct R, -1 correct L, 0 incorrect)
        # # if outcome_bool 1,  L-: correct (0)
        # # if outcome_bool 0 & choice "right", L-: incorrect (1) because right,
        # # if outcome bool 0 & choice "left", L-: incorrect (-1) because left,
        #
        # # define conditions
        # conditions = [
        #     (df_glm['outcome_bool'] == 1),
        #     (df_glm['outcome_bool'] == 0) & (df_glm['choice'] == 'right'),
        #     (df_glm['outcome_bool'] == 0) & (df_glm['choice'] == 'left'),
        # ]
        #
        # r_minus = [
        #     0,
        #     1,
        #     -1,
        # ]
        #
        # df_glm['r_minus'] = np.select(conditions, r_minus, default='other')
        # df_glm['r_minus'] = pd.to_numeric(df_glm['r_minus'], errors='coerce')
        #
        # # Convert choice from int to num (R= 1, L=-1); define conditions
        # conditions = [
        #     (df_glm['choice'] == 'right'),
        #     (df_glm['choice'] == 'left'),
        # ]
        #
        # choice_num = [
        #     1,
        #     0,
        # ]
        #
        # df_glm['choice_num'] = np.select(conditions, choice_num, default='other')
        #
        # # Creating columns for previous trial results (both dfs)
        # for i in range(1, 21):
        #     df_glm[f'r_plus_{i}'] = df_glm.groupby('session')['r_plus'].shift(i)
        #     df_glm[f'r_minus_{i}'] = df_glm.groupby('session')['r_minus'].shift(i)
        #
        # df_glm['choice_num'] = pd.to_numeric(df_glm['choice_num'], errors='coerce')
        #
        # # "variable" and "regressors" are columnames of dataframe
        # # you can add multiple regressors by making them interact: "+" for only fitting separately,
        # # "*" for also fitting the interaction
        # # Apply glm
        # mM_logit = smf.logit(
        #     formula='choice_num ~ r_plus_1 + r_plus_2 + r_plus_3 + r_plus_4 + r_plus_5 + r_plus_6+ r_plus_7+ r_plus_8'
        #             '+ r_plus_9 + r_plus_10 + r_plus_11 + r_plus_12 + r_plus_13 + r_plus_14 + r_plus_15 + r_plus_16'
        #             '+ r_plus_17 + r_plus_18 + r_plus_19+ r_plus_20'
        #             '+ r_minus_1 + r_minus_2 + r_minus_3 + r_minus_4 + r_minus_5 + r_minus_6+ r_minus_7+ r_minus_8'
        #             '+ r_minus_9 + r_minus_10 + r_minus_11 + r_minus_12 + r_minus_13 + r_minus_14 + r_minus_15 '
        #             '+ r_minus_16 + r_minus_17 + r_minus_18 + r_minus_19+ r_minus_20',
        #     data=df_glm).fit()
        #
        # # prints the fitted GLM parameters (coefs), p-values and some other stuff
        # results = mM_logit.summary()
        # print(results)
        # # save param in df
        # m = pd.DataFrame({
        #     'coefficient': mM_logit.params,
        #     'std_err': mM_logit.bse,
        #     'z_value': mM_logit.tvalues,
        #     'p_value': mM_logit.pvalues,
        #     'conf_Interval_Low': mM_logit.conf_int()[0],
        #     'conf_Interval_High': mM_logit.conf_int()[1]
        # })
        #
        # axes = plt.subplot2grid((50, 50), (39, 0), rowspan=11, colspan=32)
        # orders = np.arange(len(m))
        #
        # # filter the DataFrame to separately the coefficients
        # r_plus = m.loc[m.index.str.contains('r_plus'), "coefficient"]
        # r_minus = m.loc[m.index.str.contains('r_minus'), "coefficient"]
        # intercept = m.loc['Intercept', "coefficient"]
        #
        # plt.plot(orders[:len(r_plus)], r_plus, label='r+', marker='o', color='indianred')
        # plt.plot(orders[:len(r_minus)], r_minus, label='r-', marker='o', color='teal')
        # plt.axhline(y=intercept, label='Intercept', color='black')
        #
        #
        # axes.set_ylabel('GLM weight', label_kwargs)
        # axes.set_xlabel('Prevous trials', label_kwargs)
        # plt.legend()
        #
        #

        #behavior around block change
        axes = plt.subplot2grid((50, 50), (45, 0), rowspan=15, colspan=20)

        df['block'] = (df['probability_r'].diff().abs() > 0).cumsum()

        # Add the new column 'highest_probability' which combines the values
        df['highest_probability'] = df.apply(
            lambda row: row['probability_r'] if row['probability_r'] > 0.5 else row['probability_l'], axis=1
        )

        # Calculate the probability of choosing the port with the highest probability
        df['high_choice'] = ((df['first_trial_response'] == 'right') & (df['probability_r'] > df['probability_l'])) | (
                (df['first_trial_response'] == 'left') & (df['probability_l'] > df['probability_r']))

        # For each block, normalize the trials around the block change
        block_positions = []
        high_choices = []
        for block in df['block'].unique()[1:]:  # Exclude the first block
            block_df = df[df['block'] == block]
            last_trial_prev_block = df[df['block'] == block - 1]['trial'].max()
            block_df['block_position'] = block_df['trial'] - last_trial_prev_block
            block_positions.extend(block_df['block_position'])
            high_choices.extend(block_df['high_choice'])

            # last 10 trials before the block change
            prev_block_df = df[df['block'] == (block - 1)]
            if len(prev_block_df) > 10:
                prev_block_df = prev_block_df.iloc[-10:]
            prev_block_positions = prev_block_df['trial'] - last_trial_prev_block
            block_positions.extend(prev_block_positions)
            high_choices.extend(prev_block_df['high_choice'])

        # Create a DataFrame with trial positions and high choices
        df_block_positions = pd.DataFrame({'block_position': block_positions, 'high_choice': high_choices})

        # Filter for the last 10 trials before and first 10 trials after the block change
        filtered_df = df_block_positions[
            ((df_block_positions['block_position'] >= -10) & (df_block_positions['block_position'] <= -1)) |
            ((df_block_positions['block_position'] >= 0) & (df_block_positions['block_position'] <= 9))
            ]

        # Calculate the mean and standard error of the mean for the high choice probability
        mean_high_choice = filtered_df.groupby('block_position')['high_choice'].mean()
        sem_high_choice = filtered_df.groupby('block_position')['high_choice'].sem()

        # Plot

        plt.plot(mean_high_choice.index, mean_high_choice, label='Average')
        plt.fill_between(mean_high_choice.index, mean_high_choice - sem_high_choice, mean_high_choice + sem_high_choice,
                         alpha=0.2)


        plt.axvline(x=0, color='black', linestyle='--', label='Block change')
        plt.xlabel('Trial Position')
        plt.ylabel('P(high port)')
        plt.title('Mouse behavior around probability block change')

    # Salva il grafico nel PDF
    with PdfPages(pdf_path) as pdf:
        pdf.savefig()
        plt.close()


folder_path = "C:\\academy_reports\\academy_reports\\sessions\\intersession"

#Recursively search for CSV files
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".csv"):
            print('csv file: ', file)
            path = os.path.join(root, file)
            print('csv path: ', path)
            save_path = path[:-3] + 'pdf'

            #if not "raw" in file and "A" in file and not os.path.exists(save_path): # requires manually delete the generated pdf in order to generate a new one
            if not "raw" in file and "A" in file: # overwrite the pdf to the previous one

                intersession_opto(path, save_path)

