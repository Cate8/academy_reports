import pandas as pd
from academy_reports import utils

from PyPDF2 import PdfReader
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.ticker as ticker
from scipy import stats
from scipy.special import erf
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import statsmodels.formula.api as smf
import os

def intersession(csv_path, pdf_path):
    df = pd.read_csv(csv_path, sep=';')  # Modifica il separatore se necessario


    # PLOT COLORS & FONT
    label_kwargs = {'fontsize': 9}
    lines_c = 'gray'

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



    ##### SELECT LAST MONTH SESSIONS #####
    # Assumendo che la colonna 'date' contenga stringhe, convertirla in formato datetime
    df['date'] = pd.to_datetime(df['date'])

    # Creare una colonna 'day' che contiene solo la parte di data (senza il tempo)
    df['day'] = df['date'].dt.date

    # Gestire eventuali valori mancanti (NaT o NaN) nella colonna 'day'
    df = df.dropna(subset=['day'])

    # Filtrare il DataFrame per ottenere solo le righe degli ultimi 20 giorni
    df = df.loc[df['day'] > df['day'].max() - timedelta(days=20)]

    ###### RELEVANT COLUMNS ######

    # weights
    weight = df.subject_weight.iloc[-1]
    df['relative_weights'] = df.apply(lambda x: utils.relative_weights(x['subject'], x['subject_weight']), axis=1)

    # latencies & times
    df['center_response_time'] = df['STATE_center_light_END'] - df['STATE_center_light_START']
    df['response_time'] = df['STATE_side_light_END'] - df['STATE_side_light_START']
    df['duration_drink_delay'] = df['STATE_drink_delay_END'] - df['STATE_drink_delay_START']
    df['centre_response_latency'] = df['center_response_time']
    df['right_poke_in'] = df['right_poke_in'].astype(str)
    df['left_poke_in'] = df['left_poke_in'].astype(str)
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
    df['first_right_trial_response'] = df['first_trial_response'].apply(lambda x: 1 if x == 'right' else 0)


    ############# PAGE 1 ##############
    with PdfPages(pdf_path) as pdf:
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
        axes.axhline(y=170, color=lines_c, linestyle=':', linewidth=1)
        axes.set_ylabel('Nº of trials (µ)', label_kwargs)
        axes.set_ylim(0, 350)
        axes.set_xlabel('')
        axes.xaxis.set_ticklabels([])
        axes.legend(title='task', fontsize=8, loc='center', bbox_to_anchor=(0.05, 1.2))

        # legend water drunk
        daily_df = df.groupby(['day', 'session']).agg({'reward_drunk': "max", 'task': "max"}).reset_index()
        reward_df = daily_df.groupby(['day']).agg({'reward_drunk': "sum"}).reset_index()
        try:
            label = 'Water Today: ' + str(reward_df.reward_drunk.iloc[-1]) + 'ul, Yesterday: ' + str(
                reward_df.reward_drunk.iloc[-2]) + 'ul, Prev: ' + str(reward_df.reward_drunk.iloc[-3]) + 'ul'
        except:
            label = 'Water Today: ' + str(reward_df.reward_drunk.iloc[-1])
        axes.text(0.12, 1.2, label, transform=axes.transAxes, fontsize=8, fontweight='bold',
                  verticalalignment='top')


        ### PLOT 2: RELATIVE WEIGHTS
        axes = plt.subplot2grid((50, 50), (0, 27), rowspan=5, colspan=24)
        sns.lineplot(x='day', y='relative_weights', style='task', markers=True, ax=axes, color='black', data=df)

        axes.axhline(y=50, color=lines_c, linestyle=':', linewidth=1)
        axes.set_ylabel('Rel. weight (%)', label_kwargs)
        axes.set_ylim(0, 200)
        axes.set_xlabel('')
        axes.xaxis.set_ticklabels([])
        axes.get_legend().remove()

        label = 'Last: ' + str(weight) + ' g'
        axes.text(0.01, 1.2, label, transform=axes.transAxes, fontsize=8, fontweight='bold', verticalalignment='top')


        ### PLOT 3:  NUMBER OF SESSIONS
        axes = plt.subplot2grid((50, 50), (6, 0), rowspan=5, colspan=24)
        # Assicurati che la colonna 'task' contenga solo stringhe
        df['task'] = df['task'].astype(str)
        sessions_df = df.groupby(['day']).agg({'session': 'nunique', 'task': "max"}).reset_index()
        sns.lineplot(x='day', y='session', style='task', markers=True, ax=axes, color='black', data=sessions_df)
        axes.axhline(y=3, color=lines_c, linestyle=':', linewidth=1)

        axes.set_ylabel('Nº of sessions', label_kwargs)
        axes.tick_params(axis='x', rotation=45)
        axes.set_xlabel('Day', label_kwargs)
        axes.get_legend().remove()

        axes.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))


        ### PLOT 4: LATENCIES
        axes = plt.subplot2grid((50, 50), (6, 27), rowspan=5, colspan=24)
        filtered_df = df[df['response_time'].notnull() & df['center_response_time'].notnull()]

        sns.lineplot(x='day', y='response_time', style ='task', data=filtered_df, estimator=np.median, errorbar=None, ax=axes,
                     color='dodgerblue', markers=True)
        sns.lineplot(x='day', y='center_response_time', style ='task', data=filtered_df, estimator=np.median, errorbar=None, ax=axes,
                     color='black', markers=True)

        axes.set_yscale('log')
        axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        custom_yticks = [0, 1, 10]
        for y in custom_yticks:
            axes.axhline(y=y, linestyle='--', color='lightgray', alpha=0.7)
        axes.tick_params(axis='x', rotation=45)
        axes.set_xlabel('Day')
        axes.set_ylabel('Me Latency (s)')
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        legend_elements = [Line2D([0], [0], color='dodgerblue', lw=2, label='Response Time'),
                           Line2D([0], [0], color='black', lw=2, label='Center Response Time')]
        axes.legend(handles=legend_elements, loc='upper right')


        #Daily PCs last five days of training: RIGHT CHOICE (BIAS)
        # Funzione per aggregare i dati per giorno
        def aggregate_data(df):
            df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
            return df

        # Funzione probit per il fitting
        def probit(x, beta, alpha):
            return 0.5 * (1 + erf((beta * x + alpha) / np.sqrt(2)))

        # Generare curve psicometriche per gli ultimi 5 giorni
        def plot_psychometric_curves(df):

            # Ottenere le ultime 5 date
            last_five_dates = df['date'].unique()[-5:]

            for i, date in enumerate(last_five_dates):
                group = df[df['date'] == date].copy()
                ax = plt.subplot2grid((50, 50), (15, i * 10), rowspan=8, colspan=8)

                group['probability_r'] = group['probability_r'].astype(float)
                group['first_trial_response'] = group['first_trial_response'].apply(lambda x: 1 if x == 'right' else 0)
                probs = np.sort(group['probability_r'].unique())
                right_choice_freq = []

                for prob in probs:
                    indx_blk = group['probability_r'] == prob
                    sum_prob = np.sum(indx_blk)
                    indx_blk_r = indx_blk & (group['first_trial_response'] == 1)
                    sum_choice_prob = np.sum(indx_blk_r)
                    right_choice_freq.append(sum_choice_prob / sum_prob)

                pars, _ = curve_fit(probit, group['probability_r'], group['first_trial_response'], p0=[0, 1])

                x = np.linspace(0, 1, 100)
                ax.plot(x, probit(x, *pars), label='Probit Fit', color='teal', linewidth=2)
                ax.scatter(probs, right_choice_freq, marker='o', color='teal', s=20)
                ax.set_ylim(0, 1)
                ax.axhline(0.5, color='gray', linestyle='--')
                ax.axvline(0.5, color='gray', linestyle='--')
                ax.set_xlabel('Probability Type')
                ax.set_ylabel('Right Choice Rate')
                ax.set_title(f'{pd.to_datetime(date).strftime("%Y-%m-%d")}')
                if i == 0:
                    ax.set_ylabel('Right Choice Rate')
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels('')
                if i == 2:
                    ax.set_xlabel('probability type')
                else:
                    ax.set_xlabel('')

        # Generare e visualizzare le curve psicometriche

        df_aggregated = aggregate_data(df)
        plot_psychometric_curves(df)

        # Daily PCs last five days of training: RIGHT CHOICE (BIAS)
        # Funzione per aggregare i dati per giorno
        def aggregate_data(df):
            df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
            return df

        # Funzione probit per il fitting
        def probit(x, beta, alpha):
            return 0.5 * (1 + erf((beta * x + alpha) / np.sqrt(2)))

        # Generare curve psicometriche per gli ultimi 5 giorni
        def plot_psychometric_curves(df):

            # Ottenere le ultime 5 date
            last_tree_dates = df['date'].unique()[-3:]

            for i, date in enumerate(last_tree_dates):
                group = df[df['date'] == date].copy()
                ax = plt.subplot2grid((50, 50), (27, i * 10), rowspan=8, colspan=8)

                group['probability_r'] = group['probability_r'].astype(float)
                group['correct_outcome_int'] = group['correct_outcome_int'].apply(lambda x: 1 if x == 1 else 0)
                probs = np.sort(group['probability_r'].unique())
                correct_choice_freq = []

                for prob in probs:
                    indx_blk = group['probability_r'] == prob
                    sum_prob = np.sum(indx_blk)
                    indx_blk_r = indx_blk & (group['correct_outcome_int'] == 1)
                    sum_choice_prob = np.sum(indx_blk_r)
                    correct_choice_freq.append(sum_choice_prob / sum_prob)

                pars, _ = curve_fit(probit, group['probability_r'], group['correct_outcome_int'], p0=[0, 1])

                x = np.linspace(0, 1, 100)
                ax.plot(x, probit(x, *pars), label='Probit Fit', color='teal', linewidth=2)
                ax.scatter(probs, correct_choice_freq, marker='o', color='teal', s=20)
                ax.set_ylim(0, 1)
                ax.axhline(0.5, color='gray', linestyle='--')
                ax.axvline(0.5, color='gray', linestyle='--')
                ax.set_xlabel('Probability Type')
                ax.set_ylabel('correct Choice Rate')
                ax.set_title(f'{pd.to_datetime(date).strftime("%Y-%m-%d")}')
                if i == 0:
                    ax.set_ylabel('correct choice Rate')
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels('')
                if i == 2:
                    ax.set_xlabel('probability type')
                else:
                    ax.set_xlabel('')

        # Generare e visualizzare le curve psicometriche

        df_aggregated = aggregate_data(df)
        plot_psychometric_curves(df)

        #PSICOMETRICA: RIGHT CHOICE REWARD VS PROBABILITY TYPE
        # Definisci la funzione probit
        axes = plt.subplot2grid((50, 50), (39, 0), rowspan=11, colspan=14)
        def probit(x, beta, alpha):
            probit = 1 / 2 * (1 + erf((beta * x + alpha) / np.sqrt(2)))
            return probit


        # Probabilità
        df['probability_l'] = 1 - df['probability_r']
        df['probability_l'] = df['probability_l'].astype(float)
        df['probability_r'] = df['probability_r'].astype(float)
        df['probability_r'] = df['probability_r'].round(1)
        df['probability_l'] = df['probability_l'].round(1)

        probs = np.sort(df['probability_r'].unique())
        right_choice_freq = []

        # Frequenze
        for prob in probs:
            indx_blk = df['probability_r'] == prob
            sum_prob = np.sum(indx_blk)
            indx_blk_r = indx_blk & (df['correct_outcome_int'])
            sum_choice_prob = np.sum(indx_blk_r)
            right_choice_freq.append(sum_choice_prob / sum_prob)

        # Fit
        pars, _ = curve_fit(probit, df['probability_r'], df['first_trial_response'] == 'right',
                            p0=[0, 1])

        # Plotting
        df_subject = df.sort_values(by='day')
        last_5_days = df_subject['day'].unique()[-20:]
        for idx, day in enumerate(last_5_days):
            # Plot curve
            x = np.linspace(0, 1, 100)
            plt.plot(x, probit(x, *pars))
            plt.scatter(probs, right_choice_freq, marker='o')

        xticks = [0.1, 0.3, 0.5, 0.7, 0.9]
        yticks = [0.1, 0.3, 0.5, 0.7, 0.9]
        axes.set_xticks(xticks)
        axes.set_yticks(xticks)
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.axvline(0.5, color='gray', linestyle='--')
        plt.xlabel('Probability type')
        plt.ylabel('Right choice Rate')
        plt.title('Fit Last 20 days')
        sns.despine()
        plt.ylim(0, 1)

        # PSICOMETRICA: RIGHT CHOICE REWARD VS PROBABILITY TYPE splitted by ITIs
        axes = plt.subplot2grid((50, 50), (39, 18), rowspan=11, colspan=14)
        # Define the probit function
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

        # Selection of columns
        colonne_corrette = ['probability_r', 'correct_outcome_int', 'first_trial_response', 'iti_duration']
        df3 = df[colonne_corrette].copy()

        # Usa .loc per evitare SettingWithCopyWarning
        df3.loc[:, 'right_choice'] = np.where(df3['first_trial_response'] == 'right', 1, 0)
        df3.loc[:, 'prob_rwd_correct_resp'] = np.where(df3['probability_r'] >= 0.5, df3['probability_r'],
                                                       1 - df3['probability_r'])
        df3.loc[:, 'prev_iti_duration'] = df3['iti_duration'].shift(1)
        df3['prev_iti_duration'].fillna(0, inplace=True)  # Sostituisce NaN con 0
        df3['prev_iti_category'] = pd.cut(df3['prev_iti_duration'], 4,
                                          labels=["1-2 sec", "2-6 sec", "7-12 sec", '12-30 sec'])

        # Assicurati che prev_iti_duration non contenga valori non numerici
        df3['prev_iti_duration'] = pd.to_numeric(df3['prev_iti_duration'], errors='coerce')

        # Aggregate the data
        grouped3 = df3.groupby(['prev_iti_category', 'probability_r'], observed=True).agg(
            fraction_of_right_responses=('right_choice', 'mean')
        ).reset_index()
        # List to keep track of ITI categories
        iti_categories = df3['prev_iti_category'].unique()
        # Generate a colormap with a gradient
        num_colors = len(iti_categories)
        colors = sns.color_palette("viridis", num_colors)
        # Create a dictionary to map ITI categories to colors
        color_map = dict(zip(iti_categories, colors))
        for category in iti_categories:
            subset = df3[df3['prev_iti_category'] == category]
            x_data = subset['probability_r']
            y_data = subset['right_choice']
            # get means
            subset = grouped3[grouped3['prev_iti_category'] == category]
            x_mean = subset['probability_r']
            y_mean = subset['fraction_of_right_responses']
            if len(x_data) > 0 and len(y_data) > 0:  # Ensure there are enough data points for fitting
                # Fit the probit function to the data
                pars, _ = curve_fit(probit, x_data, y_data, p0=[1, 0])  # Changed initial guess to p0=[1, 0]
                # Plot the fitted curve
                x_fit = np.linspace(0, 1, 100)
                y_fit = probit(x_fit, *pars)
                plt.plot(x_fit, y_fit, '-', color=color_map[category], label=f'{category}')
                plt.scatter(x_mean, y_mean, color=color_map[category])  # Scatter plot for actual data points
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.axvline(0.5, color='gray', linestyle='--')
        # Add titles and labels
        plt.xlabel('Probability Reward Right Response')
        plt.ylabel('Fraction Right Responses')
        sns.despine()

        # Imposta i ticks personalizzati per l'asse x
        plt.xticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                   labels=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
        # Imposta i ticks personalizzati per l'asse y
        plt.yticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                   labels=['0.1', '0.2', '0.3', '0.4','0.5', '0.6', '0.7', '0.8', '0.9'])
        # Add legend with title
        plt.legend(title='ITI Duration')
        plt.title('Fit Last 20 days')
        plt.ylim(0, 1)


        # #### last PLOT : CUMULATIVE TRIAL RATE
        axes = plt.subplot2grid((50, 50), (39, 36), rowspan=11, colspan=14)

        df['start_session'] = df.groupby(['subject', 'session'])['STATE_center_light_START'].transform('min')
        df['end_session'] = df.groupby(['subject', 'session'])['STATE_drink_delay_END'].transform('max')
        df['session_lenght'] = (df['end_session'] - df['start_session']) / 60
        df['current_time'] = df.groupby(['subject', 'session'])['STATE_center_light_START'].transform(
            lambda x: (x - x.iloc[0]) / 60)  # MINS

        max_timing = round(df['session_lenght'].max())
        max_timing = int(max_timing)
        sess_palette = sns.color_palette('Greens', 5)  # color per day

        last_5_days = df['day'].unique()[-5:]

        for idx, day in enumerate(last_5_days):

            print(idx, day)

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
            sns.lineplot(x=bins_plt, y=hist_norm, color = sess_palette[idx % len(sess_palette)], ax=axes, marker='o', markersize=4)

        axes.set_ylabel('Cum. nº of trials', label_kwargs)
        axes.set_xlabel('Time (mins)', label_kwargs)

        # legend
        lines = [Line2D([0], [0], color=sess_palette[i], marker='o', markersize=7, markerfacecolor=sess_palette[i]) for
                 i in
                 range(len(sess_palette))]
        axes.legend(lines, np.arange(-5, 0, 1), title='Days', loc='center', bbox_to_anchor=(0.1, 0.85))

    # Salva il grafico nel PDF
    with PdfPages(pdf_path) as pdf:
        pdf.savefig()
        plt.close()

    # try:
    #     print("************ trying to send the file: ", pdf_path)
    #     print(str(df.subject.iloc[0]))
    #     utils.slack_spam(str(df.subject.iloc[0])+'_intersession', pdf_path, "#prl_reports")
    #     print("ok")
    # except:
    #     print("could not send intersession")
    # print('Intersession completed succesfully!')



folder_path = "C:\\academy_reports\\academy_reports\\sessions\\intersession"

# Recursively search for CSV files
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".csv"):
            print('csv file: ', file)
            path = os.path.join(root, file)
            print('csv path: ', path)
            save_path = path[:-3] + 'pdf'

            # if not "raw" in file and "A" in file and not os.path.exists(save_path): # requires manually delete the generated pdf in order to generate a new one
            if not "raw" in file and "B" in file:  # overwrite the pdf to the previous one

                intersession(path, save_path)