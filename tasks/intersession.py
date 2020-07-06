import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from datetime import datetime
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
wm_th_c = '#ffa600'
label_kwargs = {'fontsize': 9}

# BINNING
"""
Touchscreen active area: 1440*900 pixels --> 403.2*252 mm
Stimulus radius: 35pix (9.8mm)
x_positions: 35-1405 pix --> 9.8-393.4mm
"""
l_edge = 9.8
r_edge = 393.4
bins_resp = np.linspace(l_edge, r_edge, 11)
bins_err = np.linspace(-r_edge, r_edge, 24)



def intersession(df, save_path_intersesion):
    print('Doing Intersession...')

    ###fix incorrects
    df.loc[((df.trial_result == 'miss') & (df['response_x'].notna()), 'trial_result')] = 'incorrect'

    # CONVERT STRINGS TO LISTS
    df = utils.convert_strings_to_lists(df, ['response_x', 'response_y', 'STATE_Incorrect_START', 'STATE_Incorrect_END',
                                       'STATE_Fixation_START', 'STATE_Fixation_END', 'STATE_Fixation_break_START',
                                       'STATE_Fixation_break_END', 'STATE_Response_window2_START',
                                       'STATE_Response_window2_END'])

    # RELEVANT VARIABLES
    total_sessions = df.session.max()
    tasks_list = df.task.unique().tolist()
    weight = df.subject_weight.iloc[-1]

    # RELEVANT COLUMNS
    ### latencies
    df['resp_latency'] = np.nan
    df['lick_latency'] = np.nan
    df.loc[
        df['task'] == 'LickTeaching', ['lick_latency']] = df.STATE_Wait_for_reward_END - df.STATE_Wait_for_reward_START
    df.loc[df['task'] == 'TouchTeaching', ['lick_latency']] = (df.STATE_Correct_first_reward_START.fillna(
        0) + df.STATE_Miss_reward_START.fillna(0)) - df.STATE_Response_window_END
    df.loc[
        df['task'] == 'TouchTeaching', ['resp_latency']] = df.STATE_Response_window_END - df.STATE_Response_window_START

    df['STATE_Response_window2_END'] = df['STATE_Response_window2_END'].apply(lambda x: [np.nan] if len(x) == 0 else x)
    df['response_window_end'] = df['STATE_Response_window2_END'].apply(lambda x: x[-1])
    df['response_window_end'] = df['response_window_end'].fillna(df['STATE_Response_window_END'])
    df['reward_time'] = df['STATE_Correct_first_reward_START'].fillna(0) + df[
        'STATE_Correct_other_reward_START'].fillna(0) + df['STATE_Miss_reward_START'].fillna(0)
    df.loc[df['task'].str.contains("StageTraining"), ['lick_latency']] = df['reward_time'] - df['response_window_end']
    df.loc[df['task'].str.contains("StageTraining"), ['resp_latency']] = df['response_window_end'] - df[
        'STATE_Response_window_START']

    ### relative weights
    df['relative_weights'] = df.apply(lambda x: utils.relative_weights(x['subject'], x['subject_weight']), axis=1)


    with PdfPages(save_path_intersesion) as pdf:
        plt.figure(figsize=(11.7, 8.3))  # apaisat

        # NUMBER OF TRIALS
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=7, colspan=23)

        task_palette = sns.set_palette(['black'], n_colors=len(tasks_list))
        total_trials_s = df.groupby(['session'], sort=False)['trial', 'task'].max()
        sns.lineplot(x=total_trials_s.index, y=total_trials_s.trial, hue=total_trials_s.task, style=total_trials_s.task,
                     markers = True, ax=axes)

        axes.hlines(y=[150], xmin=1, xmax=total_sessions, color=lines_c, linestyle=':')
        axes.set_ylabel('Nº of trials', label_kwargs)
        axes.set_xlabel('')
        axes.get_xaxis().set_ticks([])
        axes.legend(fontsize=8, loc='center', bbox_to_anchor=(0.15, 1.2))

        # WEIGHT PLOT
        axes = plt.subplot2grid((50, 50), (0, 27), rowspan=7, colspan=23)

        sns.lineplot(x=df.session, y=df.relative_weights, palette=task_palette, hue=df.task, style=df.task,
                     markers=True, ax=axes)
        axes.hlines(y=[85], xmin=1, xmax=total_sessions, color=lines_c, linestyle=':')
        axes.set_ylabel('Rel. \n Weight (%)')
        axes.set_ylim(75, 95)
        axes.set_xlabel('')
        axes.get_xaxis().set_ticks([])
        axes.get_legend().remove()

        label = 'Last: ' + str(weight) + ' g'
        axes.text(0.85, 1, label, transform=axes.transAxes, fontsize=8,  fontweight='bold', verticalalignment='top')

        # RESPONSES LATENCY
        axes = plt.subplot2grid((50, 50), (8, 0), rowspan=7, colspan=23)

        sns.lineplot(x=df.session, y=df.resp_latency, hue=df.task, style=df.task, markers=True, ax=axes,
                     estimator=np.median)
        axes.set_ylabel('Resp latency \n (sec)')
        axes.set_xlim(0, total_sessions+0.5)
        axes.set_xlabel('')
        axes.get_legend().remove()

        # LICKPORT LATENCY
        axes = plt.subplot2grid((50, 50), (8, 27), rowspan=7, colspan=23)

        task_palette = sns.set_palette([water_c], n_colors=len(tasks_list))
        lick_latency_s = df.groupby(['session']).agg({'lick_latency': ['mean', 'median', 'std'], 'task': max})

        a = sns.lineplot(x=df.session, y=df.lick_latency, hue=df.task, style=df.task, ax=axes, estimator=np.median)
        sns.scatterplot(x=lick_latency_s.index, y=lick_latency_s.lick_latency.iloc[:, 1],
                        hue=lick_latency_s.task.iloc[:, 0], style=lick_latency_s.task.iloc[:, 0], s=40, ax=axes)

        handles, labels = a.get_legend_handles_labels()
        axes.legend(handles[0:4], labels[0:4], bbox_to_anchor=(0.9, 1), loc='center', fontsize=8)

        axes.set_ylabel('Lick latency \n (sec)')
        axes.set_xlabel('')
        axes.get_legend().remove()


        ############################################ STAGE TRAINING PLOTS #############################################

        df = df[df['task'].str.contains("StageTraining")]
        if df.shape[0] > 0:

            # FIX DELAY TYPE IN THE FIRST SESSIONS
            df['old'] = df.date.apply(lambda x: 'true' if x <= '2020/06/28' else 'false')
            df.loc[((df.old == 'true') & (df.trial_type == 'WM_D') & (df.delay == 0)), 'delay_type'] = 'DS'
            df.loc[((df.old == 'true') & (df.trial_type == 'WM_D') & (df.delay == 0.5)), 'delay_type'] = 'DM'
            df.loc[((df.old == 'true') & (df.trial_type == 'WM_D') & (df.delay == 1)), 'delay_type'] = 'DL'


            # FIX TTYPES
            if (df['pwm_ds'] > 0).any():
                df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DS')), 'trial_type'] = 'WM_Ds'
            if (df['pwm_dm'] > 0).any():
                df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DM')), 'trial_type'] = 'WM_Dm'
            if (df['pwm_dl'] > 0).any():
                df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DL')), 'trial_type'] = 'WM_Dl'

            # FIX DELAYS
            df.loc[(df.trial_type == 'VG'), 'delay'] = -1  # VG trials
            df.loc[(df.trial_type == 'WM_I'), 'delay'] = -0.5  # WMI trials

            ttypes = df.trial_type.unique().tolist()
            dtypes = df.delay_type.unique().tolist()
            treslt = df.trial_result.unique().tolist()

            # correct the order of the lists
            if 'WM_Ds' in ttypes:
                idx = ttypes.index('WM_Ds')
                ttypes.pop(idx)
                ttypes.insert(1, "WM_Ds")
                if 'WM_I' in ttypes:
                    idx = ttypes.index('WM_I')
                    ttypes.pop(idx)
                    ttypes.insert(1, "WM_I")

            ###colors to plot columns
            df['ttype_colors'] = vg_c
            df.loc[(df.trial_type == 'WM_I', 'ttype_colors')] = wmi_c
            df.loc[(df.trial_type == 'WM_Ds', 'ttype_colors')] = wmds_c
            df.loc[(df.trial_type == 'WM_Dm', 'ttype_colors')] = wmdm_c
            df.loc[(df.trial_type == 'WM_Dl', 'ttype_colors')] = wmdl_c
            ttype_colors = df.ttype_colors.unique().tolist()


            # CREATE REPONSES DF
            ### needed columns before the unnest
            df['responses_time'] = df.apply(lambda row: utils.create_responses_time(row), axis=1)
            df['response_result'] = df.apply(lambda row: utils.create_reponse_result(row), axis=1)
            ### unnest
            resp_df = utils.unnesting(df, ['response_x', 'response_y', 'responses_time', 'response_result'])


            # RELEVANT COLUMNS
            ### responses latency (fix this)
            resp_df['resp_latency'] = resp_df['responses_time'] - resp_df['STATE_Response_window_START']
            ### errors
            resp_df['error_x'] = resp_df['response_x'] - resp_df['x']
            ### correct bool
            resp_df['correct_bool'] = np.where(resp_df['correct_th'] / 2 >= resp_df['error_x'].abs(), 1, 0)
            resp_df.loc[(resp_df.trial_result == 'miss', 'correct_bool')] = np.nan

            # SUBDATAFRAMES
            first_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='first', inplace=False).copy()
            last_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='last', inplace=False).copy()

            # SPLIT SESSION IN TWO HALFS
            first_resp_df['half_sess'] = first_resp_df['trial'].groupby(first_resp_df['session']).transform(
                np.median).astype(np.int64)
            last_resp_df['half_sess'] = last_resp_df['trial'].groupby(last_resp_df['session']).transform(
                np.median).astype(np.int64)

            first_1_half = first_resp_df.groupby('session').apply(lambda x: x[x['trial'] < x['half_sess']])
            first_2_half = first_resp_df.groupby('session').apply(lambda x: x[x['trial'] > x['half_sess']])
            last_1_half = last_resp_df.groupby('session').apply(lambda x: x[x['trial'] < x['half_sess']])
            last_2_half = last_resp_df.groupby('session').apply(lambda x: x[x['trial'] > x['half_sess']])

            # RELEVANT VARIABLES
            vg_correct_th = df.vg_correct_th.iloc[0]
            wm_correct_th = df.wm_correct_th.iloc[0]
            vg_chance_p = utils.chance_calculation(vg_correct_th)
            wm_chance_p = utils.chance_calculation(wm_correct_th)
            x_min = df.session.iloc[0]


            # PLOTS

            # ACCURACY TRIAL INDEX
            axes = plt.subplot2grid((50, 50), (17, 0), rowspan=13, colspan=24)

            ### first poke
            task_palette = sns.set_palette([correct_first_c], n_colors=len(tasks_list))
            sns.lineplot(x=first_1_half.session, y=first_1_half.correct_bool, hue=first_1_half.subject,
                         style=first_1_half.subject, markers=True, ci=None, ax=axes)
            sns.lineplot(x=first_2_half.session, y=first_2_half.correct_bool, hue=first_2_half.subject,
                         style=first_2_half.subject, markers=True, ci=None, dashes=[(2, 2), (2, 2)], ax=axes)

            ### last poke
            task_palette = sns.set_palette([correct_other_c], n_colors=len(tasks_list))
            sns.lineplot(x=last_1_half.session, y=last_1_half.correct_bool,  hue=last_1_half.subject,
                         style=last_1_half.subject, markers=True, ci=None, ax=axes)
            sns.lineplot(x=last_2_half.session, y=last_2_half.correct_bool, hue=last_2_half.subject,
                         style=last_2_half.subject, markers=True,  ci=None, dashes=[(2, 2), (2, 2)], ax=axes)

            axes.hlines(y=[0.5, 1], xmin=4, xmax=total_sessions, color=lines_c, linestyle=':')
            axes.fill_between(df.session, vg_chance_p, 0, facecolor=lines_c, alpha=0.3)
            axes.fill_between(df.session, wm_chance_p, 0, facecolor=lines2_c, alpha=0.4)

            axes.set_ylabel('Accuracy (%)', label_kwargs)
            axes.set_ylim(0, 1.1)
            axes.set_yticks(np.arange(0, 1.1, 0.1))
            axes.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
            axes.get_legend().remove()

            # STD TRIAL INDEX
            axes = plt.subplot2grid((50, 50), (17, 27), rowspan=13, colspan=23)

            ### first poke
            task_palette = sns.set_palette([correct_first_c], n_colors=len(tasks_list))
            sns.lineplot(x=first_1_half.session, y=first_1_half.error_x, hue=first_1_half.subject,
                         style=first_1_half.subject, markers=True, ci=None, estimator=np.std, ax=axes)
            sns.lineplot(x=first_2_half.session, y=first_2_half.error_x, hue=first_2_half.subject,
                         style=first_2_half.subject, markers=True, ci=None, estimator=np.std,
                         dashes=[(2, 2), (2, 2)],  ax=axes)

            ### last poke
            task_palette = sns.set_palette([correct_other_c], n_colors=len(tasks_list))
            sns.lineplot(x=last_1_half.session, y=last_1_half.error_x, hue=last_1_half.subject,
                         style=last_1_half.subject, markers=True, ci=None, estimator=np.std, ax=axes)
            sns.lineplot(x=last_2_half.session, y=last_2_half.error_x, hue=last_2_half.subject,
                         style=last_2_half.subject, markers=True, ci=None, estimator=np.std,
                         dashes=[(2, 2), (2, 2)], ax=axes)

            axes.set_ylabel('STD (mm)', label_kwargs)
            colors = [correct_first_c, correct_other_c, miss_c, miss_c]
            labels = ['First poke', 'Last poke', 'First half', 'Last half']
            linestyle = ['-', '-', '-', 'dotted']
            marker = ['o', 'o', ',', ',']
            lines = [Line2D([0], [0], linestyle=linestyle[i], color=colors[i], marker=marker[i], markersize=6,
                            markerfacecolor=colors[i]) for i in range(len(colors))]
            axes.legend(lines, labels, fontsize=8, loc='center', bbox_to_anchor=(1, 0.9))


            # SELECT LAST WEEK SESSIONS
            first_resp_week = first_resp_df[
                (first_resp_df['session'] > total_sessions - 6) & (first_resp_df['session'] <= total_sessions)]
            last_resp_week = last_resp_df[
                (last_resp_df['session'] > total_sessions - 6) & (last_resp_df['session'] <= total_sessions)]

            first_1_half_week = first_1_half[
                (first_1_half['session'] > total_sessions - 6) & (first_1_half['session'] <= total_sessions)]
            first_2_half_week = first_2_half[
                (first_2_half['session'] > total_sessions - 6) & (first_2_half['session'] <= total_sessions)]
            last_1_half_week = last_1_half[
                (last_1_half['session'] > total_sessions - 6) & (last_1_half['session'] <= total_sessions)]
            last_2_half_week = last_2_half[
                (last_2_half['session'] > total_sessions - 6) & (last_2_half['session'] <= total_sessions)]

            list_ = [first_1_half_week, first_2_half_week, last_1_half_week, last_2_half_week]


            # ACC TRIAL TYPE
            axes = plt.subplot2grid((50, 50), (34, 0), rowspan=16, colspan=15)
            colors = [correct_first_c, correct_first_c, correct_other_c, correct_other_c]
            linestyles = ['-', '--', '-', '--']

            week_ttypes = first_resp_week.trial_type.unique()
            x_max = len(week_ttypes) - 1
            chance_p = [vg_chance_p]
            for i in range(x_max):
                chance_p.append(wm_chance_p)

            for idx, item in enumerate(list_):
                sns.pointplot(x=item.delay, y=item.correct_bool, ax=axes, color=colors[idx], linestyles=linestyles[idx])
                axes.hlines(y=[0.5, 1], xmin=0, xmax=x_max, color=lines_c, linestyle=':')
                axes.fill_between(week_ttypes, chance_p, 0, facecolor=lines2_c, alpha=0.4)

                axes.set_xlabel('Delay (sec)', label_kwargs)
                axes.set_ylabel('Accuracy (%)', label_kwargs)
                axes.set_ylim(0, 1.1)
                axes.set_yticks(np.arange(0, 1.1, 0.1))
                axes.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])

            # STD TRIAL TYPE
            axes = plt.subplot2grid((50, 50), (34, 18), rowspan=16, colspan=15)

            for idx, item in enumerate(list_):
                sns.pointplot(x=item.delay, y=item.error_x, ax=axes, estimator=np.std, color=colors[idx],
                              linestyles=linestyles[idx])
                axes.set_xlabel('Delay (sec)')
                axes.set_ylabel('STD (mm)')

            sns.despine()

        # SAVING AND CLOSING PAGE 1
        pdf.savefig()
        plt.close()

        if df.shape[0] > 0:
            plt.figure(figsize=(11.7, 8.3))

            # PROBABILITIES PLOT
            df.loc[df['task'] != 'StageTraining_2B_V4', ['pwm_ds', 'pwm_dm', 'pwm_dl']] = df[
                ['pwm_ds', 'pwm_dm', 'pwm_dl']].multiply(df["pwm_d"], axis=0)

            probs = df.groupby("session", as_index=True)[['pvg', 'pwm_i', 'pwm_d', 'pwm_ds', 'pwm_dm', 'pwm_dl']].mean()
            probs_list = [probs.pvg, probs.pwm_i, probs.pwm_ds, probs.pwm_dm, probs.pwm_dl ]
            probs_labels = ['VG', 'WM_I', 'WM_Ds', 'WM_Dm', 'WM_Dl']
            probs_colors = [vg_c, wmi_c, wmds_c, wmdm_c, wmdl_c]

            axes = plt.subplot2grid((50, 50), (0, 0), rowspan=10, colspan=23)
            for idx, prob in enumerate(probs_list):
                sns.lineplot(x=probs.index, y=prob, ax=axes, color=probs_colors[idx])
                sns.scatterplot(x=probs.index, y=prob, ax=axes, color=probs_colors[idx], label=probs_labels[idx])

            axes.hlines(y=[0.5, 1], xmin=x_min, xmax=total_sessions, color=lines_c, linestyle=':')
            axes.set_xlabel('')
            axes.get_xaxis().set_ticks([])
            axes.set_ylim(-0.1, 1.1)
            axes.set_ylabel('Probability \n appearance', label_kwargs)
            axes.legend(fontsize=8, title='Trial type', loc='center', bbox_to_anchor=(0.12, 0.9))

            # ACCURACY TRIAL INDEX TRIAL TYPE
            axes = plt.subplot2grid((50, 50), (11, 0), rowspan=10, colspan=24)
            ttype_palette = sns.set_palette(ttype_colors, n_colors=len(ttype_colors))

            sns.lineplot(x=first_resp_df.session, y=first_resp_df.correct_bool, hue=first_resp_df.trial_type,
                         style=first_resp_df.trial_type, markers=len(ttypes)*['o'], ax=axes, ci=None)
            axes.hlines(y=[0.5, 1], xmin=4, xmax=total_sessions, color=lines_c, linestyle=':')
            axes.fill_between(df.session, vg_chance_p, 0, facecolor=lines_c, alpha=0.3)
            axes.fill_between(df.session, wm_chance_p, 0, facecolor=lines2_c, alpha=0.4)

            axes.set_ylabel('Accuracy (%)', label_kwargs)
            axes.set_ylim(0, 1.1)
            axes.set_yticks(np.arange(0, 1.1, 0.1))
            axes.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
            axes.get_legend().remove()

            #ERRORS TRIAL TYPE
            axes = plt.subplot2grid((50, 50), (0, 27), rowspan=10, colspan=23)

            sns.lineplot(x=first_resp_df.session, y=first_resp_df.error_x, hue=first_resp_df.trial_type,
                         style=first_resp_df.trial_type, markers=len(ttypes)*['o'], ax=axes)
            axes.hlines(y=[0], xmin=x_min, xmax=total_sessions, color=lines_c, linestyle=':')
            axes.set_ylabel('$Errors\ (r_{t}-x_{t})\ (mm)%$', label_kwargs)
            axes.set_xlabel('')
            axes.get_xaxis().set_ticks([])
            axes.get_legend().remove()

            # STD TRIAL INDEX TRIAL TYPE
            axes = plt.subplot2grid((50, 50), (11, 27), rowspan=10, colspan=23)
            sns.lineplot(x=first_resp_df.session, y=first_resp_df.error_x, hue=first_resp_df.trial_type,
                         style=first_resp_df.trial_type, markers=len(ttypes) * ['o'], ax=axes, estimator=np.std)
            axes.set_ylabel('STD (mm)', label_kwargs)
            axes.get_legend().remove()

            # REPONSES HIST
            ### order the week ttrial types list
            week_ttypes = first_resp_week.trial_type.unique().tolist()
            if 'WM_Ds' in week_ttypes:
                idx = week_ttypes.index('WM_Ds')
                week_ttypes.pop(idx)
                week_ttypes.insert(1, "WM_Ds")
                if 'WM_I' in week_ttypes:
                    idx = week_ttypes.index('WM_I')
                    week_ttypes.pop(idx)
                    week_ttypes.insert(1, "WM_I")

            axes_loc = [0, 11, 21, 31, 41]
            for idx, ttype in enumerate(week_ttypes):
                subset = first_resp_week.loc[first_resp_week['trial_type'] == ttype]
                axes = plt.subplot2grid((50, 50), (27, axes_loc[idx]), rowspan=9, colspan=9)
                axes.set_title(ttype, fontsize=11, fontweight='bold')
                color = subset.ttype_colors.unique()

                sns.distplot(subset.response_x, kde=False, bins=bins_resp, color=color, ax=axes,
                             hist_kws={'alpha': 0.9})
                sns.distplot(subset.x, kde=False, bins=bins_resp, color=lines2_c, ax=axes,
                             hist_kws={'alpha': 0.4})
                axes.set_xlabel('$Responses\ (r_{t})\ (mm)%$', label_kwargs)
                axes.set_ylabel('')
                if ttype == 'VG':
                    axes.set_ylabel('Nº of touches', label_kwargs)

            # ERRORS HIST
            for idx, ttype in enumerate(week_ttypes):
                subset = first_resp_week.loc[first_resp_week['trial_type'] == ttype]
                axes = plt.subplot2grid((50, 50), (41, axes_loc[idx]), rowspan=9, colspan=9)
                color = subset.ttype_colors.unique()
                correct_th = subset.correct_th.mean()

                sns.distplot(subset.error_x, kde=False, bins=bins_err, color=color, ax=axes,
                             hist_kws={'alpha': 0.9})
                axes.axvline(x=0, color=stim_c, linestyle=':', linewidth=1.5)
                axes.axvline(x=-correct_th, color=wm_th_c, linestyle=':', linewidth=1.5)
                axes.axvline(x=correct_th, color=wm_th_c, linestyle=':', linewidth=1.5)

                axes.set_xlabel('$Errors\ (r_{t}-x_{t})\ (mm)%$', label_kwargs)
                axes.set_ylabel('')
                if ttype == 'VG':
                    axes.set_ylabel('Nº of touches', label_kwargs)

            sns.despine()

            # SAVING AND CLOSING PAGE 1
            pdf.savefig()
            plt.close()




        print('intersession completed successfully')

