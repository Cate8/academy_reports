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
from PyPDF2 import PdfReader

def intersession(df,  pdf_path):

    # PLOT COLORS & FONT
    label_kwargs = {'fontsize': 9}
    lines_c = 'gray'


    ##### SELECT LAST MONTH SESSIONS #####
    df['day'] = pd.to_datetime(df['date']).dt.date
    df = df.loc[df['day'] > df.day.max() - timedelta(days=20)]

    ###### RELEVANT COLUMNS ######

    # weights
    weight = df.subject_weight.iloc[-1]
    df['relative_weights'] = df.apply(lambda x: utils.relative_weights(x['subject'], x['subject_weight']), axis=1)

    # latencies & times
    df['center_response_time'] = df['STATE_center_light_END'] - df['STATE_center_light_START']
    df['response_time'] = df['STATE_side_light_END'] - df['STATE_side_light_START']
    df['duration_drink_delay'] = df['STATE_drink_delay_END'] - df['STATE_drink_delay_START']
    df['centre_response_latency'] = df['center_response_time']
    df['Port5In_START'] = df['Port5In_START'].astype(str)
    df['Port2In_START'] = df['Port2In_START'].astype(str)
    df['first_response_right'] = df['Port5In_START'].str.split(',').str[0].astype(float)
    df['first_response_left'] = df['Port2In_START'].str.split(',').str[0].astype(float)
    df['center_median_response_time'] = df['centre_response_latency'].median()  # median latency to first response
    df['response_latency_median'] = df['response_time'].median()  # median latency to first response
    df['probability_r'] = np.round(df['probability_r'], 1)

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

        ### DEFINE palette for the next two plots
        palette = sns.color_palette(sns.cubehelix_palette(gamma=.5), n_colors=10)
        probabilities = sorted(set(df['probability_r'].unique()))
        color_map = dict(zip(probabilities, palette))



        ### PLOT 5: TTYPE right choice ACCURACIES

        axes = plt.subplot2grid((50, 50), (14, 0), rowspan=11, colspan=50)

        # Assicurati che 'probability_r' sia numerico
        df['probability_r'] = pd.to_numeric(df['probability_r'], errors='coerce')

        # Filtra il dataframe
        df_filtered = df[df['probability_r'] > 0.5].copy()

        df_filtered['prob_rwd_correct_resp'] = np.where(df_filtered['probability_r'] >= 0.5, df_filtered['probability_r'],
                                                       1 - df_filtered['probability_r'])
        df_filtered['sub_behavior'] = (df_filtered['prob_rwd_correct_resp'] > 0.5) & (
                    df_filtered['first_trial_response'] == 'right')


        # Calcola la media giornaliera
        df_filtered['daily_sub_behavior_mean'] = df_filtered.groupby('day')['sub_behavior'].transform('mean') * 100

        # Primo lineplot
        #sns.lineplot(x='day', y='daily_sub_behavior_mean', data=df_filtered, estimator=np.average, errorbar=None,
                     #ax=axes, linestyle="dashed", marker=True, style='task', color='black')

        # Secondo lineplot
        grouped = df_filtered.groupby(['day', 'task', 'prob_rwd_correct_resp'])['sub_behavior'].mean().reset_index()
        grouped['sub_behavior'] *= 100
        sns.lineplot(data=grouped, x='day', y='sub_behavior', hue='prob_rwd_correct_resp', style='task', markers=True,
                     palette='viridis')

        # Configurazione degli assi e legenda
        plt.axhline(y=50, linestyle='--', color='lightgray', alpha=0.7)
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        axes.tick_params(axis='x', rotation=45)
        plt.xlabel('Day')
        plt.ylabel('Right choices (%)')
        axes.legend().remove()


        ### PLOT 6: ACCURACY

        axes = plt.subplot2grid((50, 50), (25, 0), rowspan=11, colspan=50)

        # Create a filtered DataFrame where the probability on the right is greater than 0.5
        df['probability_r'] = pd.to_numeric(df['probability_r'], errors='coerce')

        df['correct_accuracy_choice'] = ((df['probability_r'] > 0.5) & (df['first_trial_response'] == 'right')|\
                                        (df['probability_r'] < 0.5) & (df['first_trial_response'] == 'left'))


        df['daily_correct_accuracy_choice'] = df.groupby('day')['correct_accuracy_choice'].transform('mean')*100
        #sns.lineplot(x='day', y='daily_correct_accuracy_choice', data=df, estimator=np.average, errorbar=None,
                    # ax=axes, linestyle="dashed", color='black', style='task', markers=True)

        df['prob_rwd_correct_resp'] = np.where(df['probability_r'] >= 0.5, df['probability_r'],
                                                       1 - df['probability_r'])

        grouped = df.groupby(['day', 'task', 'prob_rwd_correct_resp'])['correct_accuracy_choice'].mean().reset_index()
        grouped['correct_accuracy_choice'] = grouped['correct_accuracy_choice'] * 100

        sns.lineplot(data=grouped, x='day', y='correct_accuracy_choice', hue='prob_rwd_correct_resp', style='task', markers=True,
                     palette='viridis')

        plt.axhline(y=50, linestyle='--', color='lightgray', alpha=0.7)
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        # legend
        axes.tick_params(axis='x', rotation=45)
        plt.xlabel('Day', label_kwargs)
        plt.ylabel('Accuracy (%)', label_kwargs)
        axes.legend(loc='upper left')

        #### penultimo PLOT : REWARD HISTORY WEIGHT
        # Prepare df columns
        # Converting the 'outcome' column to boolean values
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
        # for i in range(1, 6):
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
        #     formula='choice_num ~ r_plus_1 + r_plus_2 + r_plus_3 + r_plus_4 + r_plus_5'
        #             '+ r_minus_1 + r_minus_2 + r_minus_3 + r_minus_4 + r_minus_5',
        #     data=df_glm).fit()
        #
        # #prints the fitted GLM parameters (coefs), p-values and some other stuff
        # results = mM_logit.summary()
        # #print(results)
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
        # axes = plt.subplot2grid((50, 50), (39, 0), rowspan=11, colspan=14)
        #
        # orders = np.arange(len(m))
        #
        # # filter the DataFrame to separately the coefficients
        # r_plus = m.loc[m.index.str.contains('r_plus'), "coefficient"]
        # r_minus = m.loc[m.index.str.contains('r_minus'), "coefficient"]
        # intercept = m.loc['Intercept', "coefficient"]
        #
        # plt.plot(orders[:len(r_plus)], r_plus, label='r+', marker='o', color='indianred')
        # plt.plot(orders[:len(r_minus)], r_minus, label='r-', marker='o', color='teal')
        # plt.axhline(y=intercept,  linestyle='--', color='lightgray', alpha=0.7)
        #
        # xticks = [1, 2, 3, 4, 5]
        # axes.set_xticks(xticks)
        #
        # axes.set_ylabel('GLM weight', label_kwargs)
        # axes.set_xlabel('Prevous trials', label_kwargs)
        # plt.legend()

        #PSICOMETRICA: RIGHT CHOICE REWARD VS PROBABILITY TYPE
        # Definisci la funzione probit
        axes = plt.subplot2grid((50, 50), (39, 0), rowspan=11, colspan=14)
        def probit(x, beta, alpha):
            probit = 1 / 2 * (1 + erf((beta * x + alpha) / np.sqrt(2)))
            return probit

        # Ottieni gli animali e i task unici
        subjects = df['subject'].unique()
        tasks = df['task'].unique()

        # Filtro per escludere 'manual'
        subjects = subjects[subjects != 'manual']

        # Itera per ogni animale
        for subject in subjects:
            # Filtra i dati per il soggetto corrente
            df_subject = df[df['subject'] == subject]

            # Probabilità
            df_subject['probability_l'] = 1 - df_subject['probability_r']
            df_subject['probability_l'] = df_subject['probability_l'].astype(float)
            df_subject['probability_r'] = df_subject['probability_r'].astype(float)
            df_subject['probability_r'] = df_subject['probability_r'].round(1)
            df_subject['probability_l'] = df_subject['probability_l'].round(1)

            probs = np.sort(df_subject['probability_r'].unique())
            right_choice_freq = []

            # Frequenze
            for prob in probs:
                indx_blk = df_subject['probability_r'] == prob
                sum_prob = np.sum(indx_blk)
                indx_blk_r = indx_blk & (df_subject['first_trial_response'] == 'right')
                sum_choice_prob = np.sum(indx_blk_r)
                right_choice_freq.append(sum_choice_prob / sum_prob)

            # Fit
            pars, _ = curve_fit(probit, df_subject['probability_r'], df_subject['first_trial_response'] == 'right',
                                p0=[0, 1])

            # Plotting
            df_subject = df_subject.sort_values(by='day')
            last_5_days = df_subject['day'].unique()[-5:]
            for idx, day in enumerate(last_5_days):
                # Plot curve
                x = np.linspace(0, 1, 100)
                plt.plot(x, probit(x, *pars))
                plt.scatter(probs, right_choice_freq, marker='o')

            xticks = [0.1, 0.3, 0.5, 0.7, 0.9]
            axes.set_xticks(xticks)

            plt.axhline(0.5, color='gray', linestyle='--')
            plt.axvline(0.5, color='gray', linestyle='--')
            plt.xlabel('Probability type')
            plt.ylabel('Right choice Rate')
            sns.despine()

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

        #df3.loc[:, 'prev_iti_category'] = pd.cut(df3['prev_iti_duration'], 4, labels=False)

        # Aggregate the data
        grouped3 = df3.groupby(['prev_iti_category', 'probability_r']).agg(
            fraction_of_right_responses=('right_choice', 'mean')
        ).reset_index()
        # List to keep track of ITI categories
        iti_categories = grouped3['prev_iti_category'].unique()
        # Generate a colormap with a gradient
        num_colors = len(iti_categories)
        colors = sns.color_palette("viridis", num_colors)
        # Create a dictionary to map ITI categories to colors
        color_map = dict(zip(iti_categories, colors))
        for category in iti_categories:
            subset = grouped3[grouped3['prev_iti_category'] == category]
            x_data = subset['probability_r']
            y_data = subset['fraction_of_right_responses']
            if len(x_data) > 0 and len(y_data) > 0:  # Ensure there are enough data points for fitting
                # Fit the probit function to the data
                pars, _ = curve_fit(probit, x_data, y_data, p0=[1, 0])  # Changed initial guess to p0=[1, 0]
                # Plot the fitted curve
                x_fit = np.linspace(0, 1, 100)
                y_fit = probit(x_fit, *pars)
                plt.plot(x_fit, y_fit, '-', color=color_map[category], label=f'{category}')
                plt.scatter(x_data, y_data, color=color_map[category])  # Scatter plot for actual data points
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.axvline(0.5, color='gray', linestyle='--')
        # Add titles and labels
        plt.xlabel('Probability Reward Right Response')
        plt.ylabel('Fraction Right Responses')
        sns.despine()
        # Imposta i ticks personalizzati per l'asse x
        plt.xticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
                   labels=['0.1', '0.2', '0.3', '0.4', '0.6', '0.7', '0.8', '0.9'])
        # Imposta i ticks personalizzati per l'asse y
        plt.yticks(ticks=[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
                   labels=['0.1', '0.2', '0.3', '0.4', '0.6', '0.7', '0.8', '0.9'])
        # Add legend with title
        plt.legend(title='ITI Duration')
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

    try:
        print("************ trying to send the file: ", pdf_path)
        print(str(df.subject.iloc[0]))
        utils.slack_spam(str(df.subject.iloc[0])+'_intersession', pdf_path, "#prl_reports")
        print("ok")
    except:
        print("could not send intersession")
    print('Intersession completed succesfully!')