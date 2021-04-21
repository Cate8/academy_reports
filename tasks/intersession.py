import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from academy_reports import utils

# PLOT COLORS
correct_first_c = 'green'
correct_other_c = 'limegreen'
miss_c = 'black'
incorrect_c = 'red'
punish_c = 'firebrick'
water_c = 'teal'

lines_c = 'gray'
lines2_c = 'silver'

vg_c = '#393b79'
wmi_c = '#6b6ecf'
wmds_c = '#9c9ede'
wmdm_c = '#ce6dbd'
wmdl_c = '#a55194'

stim_c = 'gold'
correct_th_c = 'green'
repoke_th_c = 'orangered'

label_kwargs = {'fontsize': 9}


def intersession(df, save_path_intersesion):

    df = df[df['task'].str.contains("StageTraining")]

    if df.shape[0] > 1:
        print('Doing Intersession...')

        #####################  PARSE #####################

        ###### RELEVANT VARIABLES ######
        weight = df.subject_weight.iloc[-1]
        total_sessions = df.session.max()
        x_min = df.session.min()
        tasks_list = df.task.unique().tolist()
        try:
            df['task'].str.contains("4B")
            training_type = 't-maze'
            try:
                df['task'].str.contains("5B")
                training_type = 't-maze'
            except:
                pass
        except:
            training_type = 'classic'


        # SESSION-GROUPED DF
        grouped_df = df.groupby('session').agg(
            {'trial': max, 'task': max, 'width': max, 'correct_th': max, 'repoke_th': max, 'mask': max, 'x': 'unique'})

        # chance calculation
        grouped_df['chance_p'] = 1 / grouped_df['mask']
        grouped_df.loc[(grouped_df['mask'] == 0, 'chance_p')] = grouped_df['correct_th'].apply(
            lambda row: utils.chance_calculation(row))  # correct for mask == 0

        # binning
        ### FALTA! ####



        ######  RELEVANT COLUMNS  ######
        # add columns (when absent)
        column_list = ['STATE_Wait_for_fixation_START', 'STATE_Fixation_break_START', 'STATE_Delay_START',
                       'STATE_Doors_START', 'STATE_Correct_first_START', 'STATE_Miss_START', 'STATE_Punish_START',
                       'STATE_After_punish_START', 'STATE_Incorrect_START', 'STATE_Incorrect_END',
                       'STATE_Response_window2_START', 'STATE_Response_window2_END', 'STATE_Correct_other_START',
                       'STATE_Correct_first_reward_START', 'STATE_Correct_other_reward_START',
                       'STATE_Miss_reward_START', 'STATE_Stimulus_offset_START', 'STATE_Re_Start_task_START',
                       'fixation_time', 'stim_duration']
        for col in column_list:
            if col not in df.columns:
                df[col] = np.nan

        # ttype colors
        df['ttype_colors'] = vg_c
        df.loc[(df.trial_type == 'WM_I', 'ttype_colors')] = wmi_c

        # tresult colors
        df['treslt_colors'] = miss_c
        df.loc[(df.trial_result == 'correct_first', 'treslt_colors')] = correct_first_c
        df.loc[(df.trial_result == 'correct_other', 'treslt_colors')] = correct_other_c
        df.loc[(df.trial_result == 'incorrect', 'treslt_colors')] = incorrect_c
        df.loc[(df.trial_result == 'punish', 'treslt_colors')] = punish_c

        # separate DS DL (when required)
        try:
            if (df['pwm_ds'] > 0).any():
                df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DS')), 'trial_type'] = 'WM_Ds'
                df.loc[(df.trial_type == 'WM_Ds', 'ttype_colors')] = wmds_c
        except:
            pass
        try:
            if (df['pwm_dl'] > 0).any():
                df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DL')), 'trial_type'] = 'WM_Dl'
                df.loc[(df.trial_type == 'WM_Dl', 'ttype_colors')] = wmdl_c
        except:
            pass


        ###### USEFUL LISTS ######
        #probs
        probs = ['pvg', 'pwm_i', 'pwm_d', 'pwm_ds', 'pwm_dl']
        for prob in probs[:]:
            if prob not in df.columns:
                probs.remove(prob)

        probs, probs_c = utils.order_lists(probs, 'probs')  # order lists

        # ttypes & treslts
        ttypes = df.trial_type.unique().tolist()
        ttypes, ttypes_c = utils.order_lists(ttypes, 'ttypes')  # order lists
        treslts = df.trial_result.unique().tolist()
        treslts, treslts_c = utils.order_lists(treslts, 'treslts')  # order lists


        ###### CONVERT STRINGS TO LISTS ######
        conversion_list = ['STATE_Incorrect_START', 'STATE_Incorrect_END',
                           'STATE_Fixation_START', 'STATE_Fixation_END', 'STATE_Response_window2_START',
                           'STATE_Response_window2_END', 'STATE_Stimulus_offset_START', 'STATE_Re_Start_task_START']

        for idx, column in enumerate(conversion_list):
            try:
                df[column].str.contains(',')  # means that contains multiple values
            except:  # remove from conversion list
                conversion_list.remove(column)

        conversion_list.extend(['response_x', 'response_y'])
        df = utils.convert_strings_to_lists(df, conversion_list)


        ###### RELEVANT COLUMNS ######

        # Latencies
        # add nans to empty list, if not error
        df['STATE_Response_window2_END'] = df['STATE_Response_window2_END'].apply(
            lambda x: [np.nan] if len(x) == 0 else x)
        df['response_window_end'] = df['STATE_Response_window2_END'].apply(lambda x: x[-1])
        df['response_window_end'] = df['response_window_end'].fillna(df['STATE_Response_window_END'])
        df['reward_time'] = df['STATE_Correct_first_reward_START'].fillna(0) + df[
            'STATE_Correct_other_reward_START'].fillna(0) + df['STATE_Miss_reward_START'].fillna(0)
        df['lick_latency'] = df['reward_time'] - df['response_window_end']

        # Stimulus duration and fixation calc (when required)
        try:
            df['STATE_Fixation_START'] = df['STATE_Fixation_START'].apply(lambda x: x[-1])
            df['STATE_Fixation_END'] = df['STATE_Fixation_END'].apply(lambda x: x[-1])
        except:
            try:
                df['STATE_Fixation_START'] = df['STATE_Fixation1_START'].apply(lambda x: x[-1])
            except:
                df['STATE_Fixation_START'] = df['STATE_Fixation1_START']
            df['STATE_Fixation_END'] = df['STATE_Fixation2_END']

        try:
            df['STATE_Stimulus_offset_START'] = df['STATE_Stimulus_offset_START'].apply(lambda x: x[-1])
        except:
            pass

        df.loc[((df['stim_duration'].isnull()) & (df['trial_type'] == 'VG')), 'stim_duration'] = \
            df['response_window_end'] - df['STATE_Fixation_START']
        df.loc[((df['stim_duration'].isnull()) & (df['trial_type'] == 'WM_I')), 'stim_duration'] = \
            df['STATE_Stimulus_offset_START'] - df['STATE_Fixation_START']
        df.loc[((df['stim_duration'].isnull()) & (df['trial_type'] == 'WM_D')), 'stim_duration'] = \
            df['STATE_Stimulus_offset_START'] - df['STATE_Fixation_START']

        df.loc[df['fixation_time'].isnull(), 'fixation_time'] = df['STATE_Fixation_END'] - df['STATE_Fixation_START']

        # Relative weights
        df['relative_weights'] = df.apply(lambda x: utils.relative_weights(x['subject'], x['subject_weight']), axis=1)


        ###### CREATE REPONSES DF ######
        # check if unnest is needed
        df['unnest'] = 1
        df.loc[((df['stage'] == 1) & (df['substage'] == 1) & (df['date'] != '2020/11/18')), 'unnest'] = 0

        # needed columns before the unnest
        df['responses_time'] = df.apply(lambda row: utils.create_responses_time(row, row['unnest']), axis=1)
        df['response_result'] = df.apply(lambda row: utils.create_reponse_result(row, row['unnest']), axis=1)

        # unnest
        resp_df = utils.unnesting(df, ['response_x', 'response_y', 'responses_time', 'response_result'])

        ##### RELEVANT COLUMNS  ######

        # errors
        resp_df['error_x'] = resp_df['response_x'] - resp_df['x']

        # correct bool
        resp_df['resp_latency'] = resp_df['responses_time'] - resp_df['STATE_Response_window_START']
        resp_df['correct_bool'] = np.where(resp_df['correct_th'] / 2 >= resp_df['error_x'].abs(), 1, 0)
        resp_df.loc[(resp_df.trial_result == 'miss', 'correct_bool')] = np.nan

        # results colors
        resp_df['rreslt_colors'] = miss_c
        resp_df.loc[(resp_df.response_result == 'correct_first', 'rreslt_colors')] = correct_first_c
        resp_df.loc[(resp_df.response_result == 'correct_other', 'rreslt_colors')] = correct_other_c
        resp_df.loc[(resp_df.response_result == 'incorrect', 'rreslt_colors')] = incorrect_c
        resp_df.loc[(resp_df.response_result == 'punish', 'rreslt_colors')] = punish_c

        # SUBDATAFRAMES
        first_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='first', inplace=False).copy()
        last_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='last', inplace=False).copy()


        #####################  PLOTS #####################

        ############## PAGE 1 ##############
        with PdfPages(save_path_intersesion) as pdf:
            plt.figure(figsize=(11.7, 15))

            ### PLOT 1: NUMBER OF TRIALS
            axes = plt.subplot2grid((50, 50), (0, 0), rowspan=5, colspan=23)

            sns.lineplot(x=grouped_df.index, y=grouped_df.trial, hue=grouped_df.task, style=grouped_df.task,
                         markers=True, ax=axes, palette=['black']* len(grouped_df.task.unique()))

            axes.hlines(y=[150], xmin=x_min, xmax=total_sessions, color=lines_c, linestyle=':', linewidth=1)
            axes.set_ylabel('Nº of trials', label_kwargs)
            axes.set_xlabel('')
            axes.get_xaxis().set_ticks([])
            axes.legend(fontsize=8, loc='center', bbox_to_anchor=(0.15, 1.2))


            ### PLOT 2: RELATIVE WEIGHTS
            axes = plt.subplot2grid((50, 50), (0, 27), rowspan=5, colspan=24)

            sns.lineplot(x=df.session, y=df.relative_weights, hue=df.task, style=df.task,
                         markers=True, ax=axes, palette=['black']* len(grouped_df.task.unique()))

            axes.hlines(y=[85, 100], xmin=x_min, xmax=total_sessions, color=lines_c, linestyle=':', linewidth=1)
            axes.set_ylabel('Rel. \n Weight (%)', label_kwargs)
            axes.set_ylim(80, 120)
            axes.set_xlabel('')
            axes.get_xaxis().set_ticks([])
            axes.get_legend().remove()

            label = 'Last: ' + str(weight) + ' g'
            axes.text(1, 0.95, label, transform=axes.transAxes, fontsize=8,  fontweight='bold', verticalalignment='top')


            ### PLOT 3:  RESPONSE LATENCIES
            axes = plt.subplot2grid((50, 50), (6, 0), rowspan=5, colspan=24)
            ymax= 20
            treslt_palette = sns.set_palette(treslts_c, n_colors=len(treslts_c))
            axes.set_title('Responses', fontsize=10, fontweight='bold')
            sns.lineplot(x=first_resp_df.session, y=first_resp_df.resp_latency, hue=first_resp_df.trial_result, style=resp_df.task,
                         marker='o', ax=axes, estimator=np.median, palette = treslt_palette)

            axes.set_ylabel('Latency (sec)', label_kwargs)
            axes.set_ylim(0, ymax)
            axes.get_legend().remove()

            ## PLOT 4:  LICK LATENCIES
            ymax = 10
            axes = plt.subplot2grid((50, 50), (6, 27), rowspan=5, colspan=24)
            axes.set_title('Licks', fontsize=10, fontweight='bold')
            sns.lineplot(x=df.session, y=df.lick_latency, hue=df.trial_result, marker='o', ax=axes,
                         estimator=np.median, palette=treslt_palette)

            axes.set_ylabel('')
            axes.set_ylim(0, ymax)

            # legend
            axes.get_legend().remove()
            lines = [Line2D([0], [0], color=treslts_c[i], marker='o', markersize=7, markerfacecolor=treslts_c[i])
                     for i in range(len(treslts_c))]
            axes.legend(lines, treslts, title='Trial result', fontsize=8, loc='center', bbox_to_anchor=(1.1, 0.7))


            ### PLOT 5: TTYPE PROBABILITIES
            axes = plt.subplot2grid((50, 50), (14, 0), rowspan=3, colspan=39)

            for idx, prob in enumerate(probs):
                sns.lineplot(x=df['session'], y=df[prob], ax=axes,  color = probs_c[idx], marker='o')
                axes.hlines(y=[0, 0.5, 1], xmin=x_min, xmax=total_sessions, color=lines_c, linestyle=':', linewidth=1)

            axes.set_ylabel('Probability', label_kwargs)
            axes.set_ylim(-0.1, 1.1)
            axes.set_xlabel('')
            axes.get_xaxis().set_ticks([])
            axes.set_frame_on(False)


            # legend
            lines = [Line2D([0], [0], color=ttypes_c[i], marker='o', markersize=7, markerfacecolor=ttypes_c[i])
                for i in range(len(ttypes_c))]
            axes.legend(lines, ttypes, fontsize=8, title='Trial type', loc='center', bbox_to_anchor=(1.15, 0.7))


            ### PLOT 6: ACC TRIAL TYPE
            axes = plt.subplot2grid((50, 50), (18, 0), rowspan=9, colspan=39)

            ttype_palette = sns.set_palette(ttypes_c, n_colors=len(ttypes_c))

            sns.lineplot(x=first_resp_df.session, y=first_resp_df.correct_bool, hue=first_resp_df.trial_type,
                        marker='o', ax=axes, ci=None)

            axes.hlines(y=1, xmin=x_min, xmax=total_sessions, color=lines_c, linestyle=':', linewidth=1)
            axes.fill_between(grouped_df.index, grouped_df.chance_p, 0, facecolor=lines2_c, alpha=0.3)
            utils.axes_pcent(axes, label_kwargs)
            axes.get_legend().remove()
            axes.set_xlabel('')




            ############  LAST WEEK SESSIONS SELECTION ############
            days = 7
            first_resp_week = first_resp_df[
                (first_resp_df['session'] > total_sessions - days) & (first_resp_df['session'] <= total_sessions)]
            last_resp_week = last_resp_df[
                (last_resp_df['session'] > total_sessions - days) & (last_resp_df['session'] <= total_sessions)]
            grouped_df = grouped_df.tail(days)
            x_positions = grouped_df.x.iloc[-1]
            x_positions.sort()

            # weekly ttypes list
            ttypes = first_resp_week.trial_type.unique().tolist()
            ttypes, ttypes_c = utils.order_lists(ttypes, 'ttypes')  # order lists

            # assuming 3 hole mask to change
            l_edge = min(x_positions) - 40
            r_edge = max(x_positions) + 40

            df.rename(columns={"mask": "mask_holes"}, inplace=True) #rename because its name coincides with a function
            mask = int(df.mask_holes.iloc[-1])
            bins_resp = np.linspace(l_edge, r_edge, mask + 1)
            bins_err = np.linspace(-r_edge, r_edge, mask * 2 + 1)  # no està clar


            ### PLOT 7: ACC TRIAL TYPE
            axes = plt.subplot2grid((50, 50), (18, 40), rowspan=9, colspan=10)
            x_min = -0.5
            x_max = len(ttypes) - 0.5

            sns.pointplot(x=last_resp_week.trial_type, y=last_resp_week.correct_bool, ax=axes, s=100, ci=68,
                          color='black', linestyles=["--"])
            sns.pointplot(x=first_resp_week.trial_type, y=first_resp_week.correct_bool, ax=axes, s=100, ci=68,
                          color='black')

            axes.hlines(y=1, xmin=x_min, xmax=x_max, color=lines_c, linestyle=':', linewidth=1)
            axes.fill_between(np.linspace(x_min, x_max, 2), grouped_df.chance_p.mean(), 0, facecolor=lines2_c, alpha=0.3)
            axes.set_xlabel('Trial type', label_kwargs)
            utils.axes_pcent(axes, label_kwargs)
            axes.set_ylabel('')
            axes.yaxis.set_ticklabels([])


            ### PLOT 8:  STIMULUS DELAYS LENGHTS
            axes = plt.subplot2grid((50, 50), (28, 0), rowspan=3, colspan=39)

            sns.lineplot(x=df.session, y=df.wm_stim_dur,  marker='o', ax=axes, color=wmi_c)
            axes.hlines(y=[0, 0.5, 1], xmin=df.session.min(), xmax=total_sessions, color=lines_c, linestyle=':', linewidth=1)

            axes.set_ylabel('Duration (sec)', label_kwargs)
            axes.set_ylim(-0.1, 1.1)
            axes.set_xlabel('')
            axes.get_xaxis().set_ticks([])
            axes.set_frame_on(False)


            ### PLOT 8:  RESPONSE COUNTS SORTED BY STIMULUS POSITION
            side_colors = ['lightseagreen', 'bisque', 'orange']
            axes_loc = [33, 39, 45]
            rowspan = 5
            if mask == 5:
                side_colors.insert(0, 'darkcyan')
                side_colors.append('darkorange')
                axes_loc = [33, 36, 39, 42, 45]
                rowspan = 3
            side_palette = sns.set_palette(side_colors, n_colors=len(side_colors))  # palette creation

            first_resp_week['rt_bins'] = pd.cut(first_resp_week.response_x, bins=bins_resp, labels=x_positions,
                                              include_lowest=True)
            for idx in range(len(x_positions)):
                axes = plt.subplot2grid((50, 50), (axes_loc[idx], 0), rowspan=rowspan, colspan=36)
                subset = first_resp_week.loc[first_resp_week['x'] == x_positions[idx]]

                sns.countplot(subset.session, hue =subset.rt_bins, ax=axes, palette=side_colors)
                axes.set_ylabel('$x_{t}\ :%$' + str(x_positions[idx]))
                if idx != len(x_positions) - 1:
                    axes.xaxis.set_ticklabels([])
                    axes.set_xlabel('')
                    axes.get_legend().remove()
                else:
                    axes.legend(loc='center', bbox_to_anchor=(0.1, 0.8))


            #### PLOT 9: ACCURACY VS STIMULUS POSITION
            axes = plt.subplot2grid((50, 50), (30, 40), rowspan=9, colspan=10)
            x_min = 0
            x_max = 400  # screen size

            ttype_palette = sns.set_palette(ttypes_c, n_colors=len(ttypes_c))
            sns.lineplot(x='x', y="correct_bool", data=first_resp_week, hue='trial_type', marker='o',
                         markersize=8, err_style="bars", ci=68, ax=axes)

            axes.hlines(y=1, xmin=x_min, xmax=x_max, color=lines_c, linestyle=':', linewidth=1)
            axes.fill_between(np.linspace(x_min, x_max, 2), grouped_df.chance_p.mean(), 0, facecolor=lines2_c,
                              alpha=0.3)

            # axis
            axes.set_xlabel('')
            axes.xaxis.set_ticklabels([])
            utils.axes_pcent(axes, label_kwargs)
            axes.get_legend().remove()


            ### PLOT 10: RESPONSE COUNTS
            axes = plt.subplot2grid((50, 50), (41, 40), rowspan=9, colspan=10)

            hist, bins = np.histogram(first_resp_week.x, bins=bins_resp)
            sns.lineplot(x=x_positions, y=hist, marker='o', markersize=5, err_style="bars", color=lines2_c,
                         ax=axes)
            for ttype, ttype_df in first_resp_week.groupby('trial_type'):
                ttype_color = ttype_df.ttype_colors.iloc[0]
                hist, bins = np.histogram(ttype_df.response_x, bins=bins_resp)
                a = sns.lineplot(x=x_positions, y=hist, marker='o', markersize=8, err_style="bars",
                                 color=ttype_color)

            axes.set_xlim(x_min, x_max)
            axes.set_xlabel('$Responses\ (r_{t})\ (mm)$', label_kwargs)
            axes.set_ylabel('Counts', label_kwargs)


            # SAVING AND CLOSING PAGE
            sns.despine()
            pdf.savefig()
            plt.close()
            print('Interseesion completed succesfully!')


    else:
        print('Wait until Stgae Training for the intersession...')



