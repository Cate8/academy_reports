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
        plt.figure(figsize=(8.3, 11.7))  # A4 vertical

        # NUMBER OF TRIALS
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=7, colspan=50)

        task_palette = sns.set_palette(['black'], n_colors=len(tasks_list))
        total_trials_s = df.groupby(['session'], sort=False)['trial', 'task'].max()
        sns.lineplot(x=total_trials_s.index, y=total_trials_s.trial, hue=total_trials_s.task, style=total_trials_s.task,
                     markers = True, ax=axes)

        axes.hlines(y=[150], xmin=1, xmax=total_sessions, color=lines_c, linestyle=':')
        axes.set_ylabel('Nº of trials', label_kwargs)
        axes.get_xaxis().set_ticks([])
        axes.set_xlabel('')
        axes.legend(fontsize=8, loc='center', bbox_to_anchor=(0.15, 1.2))

        # WEIGHT PLOT
        axes = plt.subplot2grid((50, 50), (8, 0), rowspan=4, colspan=50)

        sns.lineplot(x=df.session, y=df.relative_weights, palette=task_palette, hue=df.task, style=df.task,
                     markers=True, ax=axes)
        axes.hlines(y=[85], xmin=1, xmax=total_sessions, color=lines_c, linestyle=':')
        axes.set_ylabel('Rel. \n Weight (%)')
        axes.set_ylim(75, 95)
        axes.get_legend().remove()

        label = 'Last weight: ' + str(weight) + ' g'
        axes.text(0.85, 0.9, label, transform=axes.transAxes, fontsize=8, verticalalignment='top',
                  bbox=dict(facecolor='white', alpha=0.5))

        # RESPONSES LATENCY
        axes = plt.subplot2grid((50, 50), (15, 0), rowspan=9, colspan=24)

        sns.lineplot(x=df.session, y=df.resp_latency, hue=df.task, style=df.task, markers=True, ax=axes,
                     estimator=np.median)
        axes.set_ylabel('Resp latency (sec)')
        axes.set_xlim(0, total_sessions+0.5)
        axes.set_xlabel('')
        axes.get_legend().remove()

        # LICKPORT LATENCY
        axes = plt.subplot2grid((50, 50), (15, 26), rowspan=9, colspan=24)

        task_palette = sns.set_palette([water_c], n_colors=len(tasks_list))
        lick_latency_s = df.groupby(['session']).agg({'lick_latency': ['mean', 'median', 'std'], 'task': max})

        a = sns.lineplot(x=df.session, y=df.lick_latency, hue=df.task, style=df.task, ax=axes, estimator=np.median)
        sns.scatterplot(x=lick_latency_s.index, y=lick_latency_s.lick_latency.iloc[:, 1],
                        hue=lick_latency_s.task.iloc[:, 0], style=lick_latency_s.task.iloc[:, 0], s=40, ax=axes)

        handles, labels = a.get_legend_handles_labels()
        axes.legend(handles[0:4], labels[0:4], bbox_to_anchor=(0.9, 1), loc='center', fontsize=8)

        axes.set_ylabel('Lick latency (sec)')
        axes.get_legend().remove()


        ############################################ STAGE TRAINING PLOTS #############################################

        df = df[df['task'].str.contains("StageTraining")]
        if df.shape[0] > 0:

            ###trial type
            delays = df.delay.unique()
            if df.loc[df.pwm_ds > 0, :].shape[0] > 0:
                df.loc[((df.trial_type == 'WM_D') & (df.delay == delays[0])), 'trial_type'] = 'WM_Ds'

            if df.loc[df.pwm_dm > 0, :].shape[0] > 0 and len(delays) > 1:
                df.loc[((df.trial_type == 'WM_D') & (df.delay == delays[1])), 'trial_type'] = 'WM_Dm'

            if df.loc[df.pwm_dl > 0, :].shape[0] and len(delays) > 2:
                df.loc[((df.trial_type == 'WM_D') & (df.delay == delays[2])), 'trial_type'] = 'WM_Dl'

            df.loc[(df.trial_type == 'VG'), 'delay'] = -1  # VG trials
            df.loc[(df.trial_type == 'WM_I'), 'delay'] = -0.5  # WMI trials

            ttypes = df.trial_type.unique().tolist()

            # correct the order of the list
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

            # RELEVANT VARIABLES
            vg_chance_p = utils.chance_calculation(df.vg_correct_th.iloc[0]) #calcular la mitjana!
            wm_chance_p = utils.chance_calculation(df.wm_correct_th.iloc[0])
            x_min = df.session.iloc[0]


            # PLOTS

            # ACCURACY TRIAL INDEX RESPONSE NUMBER
            axes = plt.subplot2grid((50, 50), (27, 0), rowspan=9, colspan=50)

            task_palette = sns.set_palette([correct_first_c], n_colors=len(tasks_list))
            sns.lineplot(x=first_resp_df.session, y=first_resp_df.correct_bool, hue=first_resp_df.subject,
                         style=first_resp_df.subject, markers=True, ax=axes, ci=None)
            task_palette = sns.set_palette([correct_other_c], n_colors=len(tasks_list))
            sns.lineplot(x=last_resp_df.session, y=last_resp_df.correct_bool, hue=last_resp_df.subject,
                         style=last_resp_df.subject, markers=True, ax=axes, ci=None)

            axes.hlines(y=[0.5, 1], xmin=4, xmax=total_sessions, color=lines_c, linestyle=':')
            axes.fill_between(df.session, vg_chance_p, 0, facecolor=lines_c, alpha=0.3)
            axes.fill_between(df.session, wm_chance_p, 0, facecolor=lines2_c, alpha=0.4)

            axes.set_xlabel('')
            axes.set_ylabel('Accuracy (%)', label_kwargs)
            axes.set_ylim(0, 1.1)
            axes.set_yticks(np.arange(0, 1.1, 0.1))
            axes.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
            colors = [correct_first_c, correct_other_c]
            labels = ['First poke', 'Last poke']
            lines = [Line2D([0], [0], color=c, marker='o', markersize=6, markerfacecolor=c) for c in colors]
            axes.legend(lines, labels, fontsize=8, loc='center', bbox_to_anchor=(0.15, 0.9))
            sns.despine()


        # SAVING AND CLOSING PAGE 1
        pdf.savefig()
        plt.close()

        if df.shape[0] > 0:
            plt.figure(figsize=(8.3, 11.7))

            # PROBABILITIES PLOT
            probs_list = [df.pvg, df.pwm_i, df.pwm_ds * df.pwm_d, df.pwm_dm * df.pwm_d, df.pwm_dl * df.pwm_d]
            ttypes = ['VG', 'WM_I', 'WM_Ds', 'WM_Dm', 'WM_Dl']
            colors = [vg_c, wmi_c, wmds_c, wmdm_c, wmdl_c]

            axes = plt.subplot2grid((50, 50), (0, 0), rowspan=8, colspan=50)

            for idx, prob in enumerate(probs_list):
                sns.lineplot(x=df.session, y=prob, ax=axes, color=colors[idx])
                sns.scatterplot(x=df.session, y=prob, ax=axes, color=colors[idx], label=ttypes[idx])
            axes.hlines(y=[0.5, 1], xmin=x_min, xmax=total_sessions, color=lines_c, linestyle=':')

            axes.set_xlabel('')
            axes.set_ylim(-0.1, 1.1)
            axes.set_ylabel('Probability \n appearance', label_kwargs)
            axes.legend(fontsize=8, title='Trial type', loc='center', bbox_to_anchor=(0.12, 0.9))

            # ACCURACY TRIAL INDEX TRIAL TYPE
            axes = plt.subplot2grid((50, 50), (9, 0), rowspan=9, colspan=50)
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
            axes.set_xlabel('')


            #ERRORS TRIAL TYPE
            axes = plt.subplot2grid((50, 50), (20, 0), rowspan=9, colspan=50)

            sns.lineplot(x=first_resp_df.session, y=first_resp_df.error_x, hue=first_resp_df.trial_type,
                         style=first_resp_df.trial_type, markers=len(ttypes)*['o'], ax=axes)
            axes.hlines(y=[0], xmin=x_min, xmax=total_sessions, color=lines_c, linestyle=':')
            axes.get_legend().remove()
            axes.set_xlabel('$Errors\ (r_{t}-x_{t})\ (mm)%$', label_kwargs)

            sns.despine()

            # SAVING AND CLOSING PAGE 1
            pdf.savefig()
            plt.close()




        print('intersession completed successfully')

