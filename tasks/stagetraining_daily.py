import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from academy_reports import utils
from matplotlib.lines import Line2D



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
wm_th_c = '#ffa600' #'#fc6900'

label_kwargs = {'fontsize': 9}

# BINNING
"""
Touchscreen active area: 1440*900 pixels --> 403.2*252 mm
Stimulus radius: 35pix (9.8mm)
x_positions: 35-1405 pix --> 9.8-393.4mm
"""
l_edge = 9.8
r_edge = 393.4
bins_resp = np.linspace(l_edge, r_edge, 6)
bins_err = np.linspace(-r_edge, r_edge, 12)


def stagetraining_daily (df, save_path, date):

    # RELEVANT COLUMNS
    ###correct for incorrects
    df.loc[((df.trial_result == 'miss') & (df.response_x != '[]'), 'trial_result')] = 'incorrect'

    ###colors to plot columns
    df['tresp_colors'] = miss_c
    df.loc[(df.trial_result == 'correct_first', 'tresp_colors')] = correct_first_c
    df.loc[(df.trial_result == 'correct_other', 'tresp_colors')] = correct_other_c
    df.loc[(df.trial_result == 'incorrect', 'tresp_colors')] = incorrect_c
    df.loc[(df.trial_result == 'punish', 'tresp_colors')] = punish_c

    df['ttype_colors'] = vg_c
    df.loc[(df.trial_type == 'WM_I', 'ttype_colors')] = wmi_c
    df.loc[(df.trial_type == 'WM_D', 'ttype_colors')] = wmds_c

    tresp_colors = df.tresp_colors.unique().tolist()
    ttype_colors = df.ttype_colors.unique().tolist()
    ttype_palette = sns.set_palette(ttype_colors, n_colors=len(ttype_colors))

    #latencies
    df['lick_time_start'] = df['STATE_Response_window2_END']
    df['lick_time_start'] = df['lick_time_start'].fillna(df['STATE_Response_window_END'])
    df['lick_time_stop'] = df['STATE_Correct_first_reward_START'].fillna(0) + df[
        'STATE_Correct_other_reward_START'].fillna(0) + df['STATE_Miss_reward_START'].fillna(0)
    df['lick_latency'] = df['lick_time_stop'] - df['lick_time_start']
    df['resp_latency'] = df['lick_time_start'] - df['STATE_Response_window_START']


    # REMOVE LIST BRACKETS & UNNEST THE REPONSES
    if date[0:8] < '20200610':
        df['response_x'] = df['response_x'].apply(lambda x: x.replace('[', '').replace(']', ''))
        df['response_y'] = df['response_y'].apply(lambda x: x.replace('[', '').replace(']', ''))
    df = utils.convert_strings_to_lists(df, ['response_x', 'response_y'])
    # 'STATE_Incorrect_END', 'STATE_Fixation_START', 'STATE_Fixation_END','STATE_Fixation_break_START', 'STATE_Fixation_break_END'])
    resp_df = utils.unnesting(df, ['response_x', 'response_y'])
    resp_df['response_x'] = pd.to_numeric(resp_df.response_x, errors='coerce')
    resp_df['response_y'] = pd.to_numeric(resp_df.response_y, errors='coerce')

    # RELEVANT VARIABLES
    subject = df.subject.iloc[0]
    weight = df.subject_weight.iloc[0]
    total_trials = int(df.trial.iloc[-1])
    valid_trials = (df.loc[df['trial_result'] != 'miss']).shape[0]
    missed_trials = total_trials - valid_trials
    reward_drunk =  int(df.reward_drunk.iloc[-1])

    total_acc_first_poke = df[df.trial_result == 'correct_first'].shape[0] / valid_trials
    total_acc_last_poke = df[(df.trial_result == 'correct_first') | (df.trial_result == 'correct_other')].shape[0] / valid_trials

    vg_df = df[df.trial_type == 'VG']
    wmi_df = df[df.trial_type == 'WM_I']
    wmd_df = df[df.trial_type == 'WM_D']

    total_acc_vg = vg_df[vg_df.trial_result == 'correct_first'].shape[0] / \
                   vg_df.loc[vg_df['trial_result'] != 'miss'].shape[0]
    if wmi_df.shape[0] > 0:
        total_acc_wmi = wmi_df[wmi_df.trial_result == 'correct_first'].shape[0] / \
                        wmi_df.loc[wmi_df['trial_result'] != 'miss'].shape[0]
        acc_wm = '  /  Acc WM Intro: ' + str(int(total_acc_wmi * 100)) + "%"
    if wmd_df.shape[0] > 0:
        total_acc_wmd = wmd_df[wmd_df.trial_result == 'correct_first'].shape[0] / \
                        wmd_df.loc[wmi_df['trial_result'] != 'miss'].shape[0]
        acc_wmd = '  /  Acc WM Delay: ' + str(int(total_acc_wmd * 100)) + "%"
        if wmi_df.shape[0] > 0:
            acc_wm = acc_wm + acc_wmd
        else:
            acc_wm = acc_wmd

    # CHANCE CALCULATION
    screen_size = 1440 * 0.28
    vg_correct_th = df.vg_correct_th.iloc[0]
    wm_correct_th = df.wm_correct_th.iloc[0]
    vg_chance_p = 1 / (screen_size / vg_correct_th)
    wm_chance_p = 1 / (screen_size / wm_correct_th)

    # PAGE 1:
    with PdfPages(save_path) as pdf:

        plt.figure(figsize=(8.3, 11.7))  # A4 vertical

        # HEADER
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=7, colspan=50)
        s1 = ('Subject: ' + str(subject) +
              '  /  Task: ' + str(df.task.iloc[0]) +
              '  /  Date: ' + str(date) +
              '  /  Box: ' + str(df.box.iloc[0]) +
              '  /  Weight: ' + str(weight) + " g" +
              '  /  Relative weight: ' + str(round(utils.relative_weights(subject, weight), 2)) + "%" +'\n')

        s2 = ('Prob VG: ' + str((df.pvg.iloc[0])*100) +
              '  /  Prob WM Intro: ' + str((df.pwm_i.iloc[0])*100) +
              '  /  Prob WM Delay: ' + str((df.pwm_d.iloc[0])*100) +
              '                                '  +
              ' Delay s: ' + str((df.pwm_i.iloc[0]) * 100) +
              '  /  Delay m: ' + str((df.pwm_d.iloc[0]) * 100) +
              '  /  Delay l: ' + str((df.pwm_d.iloc[0]) * 100) + '\n')

        s3 = ('Total trials: ' + str(int(total_trials)) +
              '  /  Valid trials: ' + str(valid_trials) +
              '  /  Missed trials: ' + str(missed_trials) +
              '  /  Reward drunk: ' + str(reward_drunk) + " ul" + '\n')

        s4 = ('Acc first poke: ' + str(int(total_acc_first_poke * 100)) + '%' +
              '  /  Acc last poke: ' + str(int(total_acc_last_poke * 100)) + "%" +
              '  /  Acc VG: ' + str(int(total_acc_vg * 100)) + "%" + acc_wm + '\n')

        axes.text(0.1, 0.9, s1+s2+s3+s4, fontsize=8, transform=plt.gcf().transFigure)

        # ACCURACY PLOT
        # we use the same axes than header

        # RAFA! create a column (correct_bool) with: 1 correct, 0 incorrects/punish; nan miss
        resp_df['error_x'] = resp_df['response_x'] - resp_df['x']
        resp_df['correct_bool'] = np.where(resp_df['correct_th'] / 2 >= resp_df['error_x'].abs(), 1, 0)
        resp_df.loc[(resp_df.trial_result == 'miss', 'correct_bool')] = np.nan

        first_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='first', inplace=False).copy()
        last_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='last', inplace=False).copy()
        first_resp_df['acc'] = utils.compute_window(first_resp_df.correct_bool, 20)
        last_resp_df['acc'] = utils.compute_window(last_resp_df.correct_bool, 20)

        sns.scatterplot(x=first_resp_df.trial, y=first_resp_df.acc, s=30, ax=axes, color=correct_first_c)
        sns.lineplot(x=first_resp_df.trial, y=first_resp_df.acc, ax=axes, color= correct_first_c)
        sns.scatterplot(x=last_resp_df.trial, y=last_resp_df.acc, s=30, ax=axes, color=correct_other_c)
        sns.lineplot(x=last_resp_df.trial, y=last_resp_df.acc, ax=axes, color=correct_other_c)
        axes.hlines(y=[0.5, 1], xmin=0, xmax=total_trials, color=lines_c, linestyle=':')
        axes.fill_between(df.trial, vg_chance_p, 0, facecolor=lines_c, alpha=0.2)
        axes.fill_between(df.trial, wm_chance_p, 0, facecolor=lines2_c, alpha=0.2)

        axes.set_xlabel('Trials', label_kwargs)
        axes.set_ylabel('Accuracy (%)', label_kwargs)
        axes.set_xlim([1, total_trials + 1])
        axes.set_ylim(0, 1.1)
        axes.set_yticks(np.arange(0, 1.1, 0.1))
        axes.set_yticklabels(['0', '', '', '', '', '50', '', '', '', '', '100'])
        sns.despine()

        colors = [correct_first_c, correct_other_c]
        labels = ['First poke', 'Last poke']
        lines = [Line2D([0], [0], color=c, marker='o', markersize=6, markerfacecolor=c) for c in colors]
        axes.legend(lines, labels, fontsize=8, loc='center', bbox_to_anchor=(0.9, 1.1))

        # RESPONSE LATENCIES
        axes = plt.subplot2grid((50, 50), (10, 0), rowspan=15, colspan=23)

        a = sns.boxplot(x='trial_result', y='resp_latency', hue='trial_type', data=df, color='white', showfliers=False,
                        ax=axes)
        sns.stripplot(x="trial_result", y="resp_latency", hue='trial_type', palette=ttype_palette, data=df, dodge=True,
                      ax=axes)

        axes.set_xticklabels(['Correct\nFirst', 'Correct\nOther', 'Incorrect', 'Miss'], fontsize=9)
        axes.set_ylabel("Response latency (sec)", label_kwargs)
        axes.set_xlabel("")
        axes.get_legend().remove()
        sns.despine()

        # LICKPORT LATENCIES
        axes = plt.subplot2grid((50, 50), (10, 27), rowspan=15, colspan=23)

        a = sns.boxplot(x='trial_result', y='lick_latency', hue='trial_type', data=df, color='white', showfliers=False,
                        ax=axes)
        sns.stripplot(x="trial_result", y="lick_latency", hue='trial_type', palette=ttype_palette, data=df, dodge=True,
                      ax=axes)

        handles, labels = a.get_legend_handles_labels()
        axes.legend(handles[2:4], labels[2:4], bbox_to_anchor=(0.9, 1), loc='center', fontsize=8)

        axes.set_xticklabels(['Correct\nFirst', 'Correct\nOther', 'Incorrect', 'Miss'], fontsize=9)
        axes.set_ylabel("Lickport latency (sec)", label_kwargs)
        axes.set_xlabel("")
        sns.despine()

        # RESPONSES HISTOGRAMS
        axes1 = plt.subplot2grid((50, 50), (29, 0), rowspan=9, colspan=14)
        axes2 = plt.subplot2grid((50, 50), (29, 17), rowspan=9, colspan=14)
        axes3 = plt.subplot2grid((50, 50), (29, 36), rowspan=9, colspan=14)

        axes = [axes1, axes2, axes3]
        axes_idx = 0

        for ttype, ttype_df in first_resp_df.groupby('trial_type'):
            axes[axes_idx].set_title(ttype, fontsize=11, fontweight='bold')
            sns.distplot(ttype_df.response_x, kde=False, bins=bins_resp, color=ttype_colors[axes_idx], ax=axes[axes_idx],
                         hist_kws={'alpha': 0.9})
            sns.distplot(ttype_df.x, kde=False, bins=bins_resp, color=lines2_c, ax=axes[axes_idx],
                         hist_kws={'alpha': 0.4})
            axes[axes_idx].set_xlabel('$Responses\ (r_{t})\ (mm)%$', label_kwargs)
            axes[axes_idx].set_ylabel('Nº of touches', label_kwargs)
            axes_idx = + 1
            sns.despine()

        # ERRORS HISTOGRAMS
        axes1 = plt.subplot2grid((50, 50), (41, 0), rowspan=9, colspan=14)
        axes2 = plt.subplot2grid((50, 50), (41, 18), rowspan=9, colspan=14)
        axes3 = plt.subplot2grid((50, 50), (41, 36), rowspan=9, colspan=14)

        axes = [axes1, axes2, axes3]
        axes_idx = 0

        for ttype, ttype_df in first_resp_df.groupby('trial_type'):
            sns.distplot(ttype_df.error_x, kde=False, bins=bins_err, color=ttype_colors[axes_idx], ax=axes[axes_idx],
                         hist_kws={'alpha': 0.9})
            axes[axes_idx].axvline(x=0, color=stim_c, linestyle=':', linewidth=1.5)
            correct_th = ttype_df['correct_th'].iloc[0]
            axes[axes_idx].axvline(x=-correct_th, color=wm_th_c, linestyle=':', linewidth=1.5)
            axes[axes_idx].axvline(x=correct_th, color=wm_th_c, linestyle=':', linewidth=1.5)
            axes[axes_idx].set_xlabel('$Errors\ (r_{t}-x_{t})\ (mm)%$', label_kwargs)
            axes[axes_idx].set_ylabel('Nº of touches', label_kwargs)
            axes_idx = + 1
            sns.despine()

        # SAVING AND CLOSING PAGE
        pdf.savefig()
        plt.close()

        # PAGE 2:
        plt.figure(figsize=(11.7, 11.7))  # A4 vertical

        # RESPONSES LATENCY TRIAL INDEX
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=8, colspan=50)
        sns.scatterplot(x=df.trial, y=df.resp_latency, color=wmdl_c, s=30, ax=axes)
        sns.lineplot(x=df.trial, y=df.resp_latency, color=wmdl_c, ax=axes)
        axes.hlines(y=5, xmin=0, xmax=total_trials, color=lines_c, linestyle=':')
        axes.set_ylabel('Response latency (sec)', label_kwargs)
        axes.set_ylim(0, 60)
        axes.set_xlim(0, total_trials + 1)
        axes.set_xlabel('')

        resp_latency_mean = df.resp_latency.median()
        lick_latency_mean = df.lick_latency.median()

        label = 'Mean: ' + str(round(resp_latency_mean, 1)) + ' sec'
        axes.text(0.85, 0.9, label, transform=axes.transAxes, fontsize=9, verticalalignment='top',
                  bbox=dict(facecolor='white', alpha=0.5))

        # Lick latency plot
        axes = plt.subplot2grid((50, 50), (10, 0), rowspan=8, colspan=50)
        sns.scatterplot(x=df.trial, y=df.lick_latency, color=water_c, s=30, ax=axes)
        sns.lineplot(x=df.trial, y=df.lick_latency, color=water_c, ax=axes)
        axes.hlines(y=5, xmin=0, xmax=total_trials, color=lines_c, linestyle=':')
        axes.set_ylabel('Lick latency (sec)', label_kwargs)
        axes.set_ylim(0, 60)
        axes.set_ylabel('Trials')
        axes.set_xlim(0, total_trials + 1)

        label = 'Mean: ' + str(round(lick_latency_mean, 1)) + ' sec'
        axes.text(0.85, 0.9, label, transform=axes.transAxes, fontsize=8, verticalalignment='top',
                  bbox=dict(facecolor='white', alpha=0.5))

        # RESPONSES TRIAL INDEX
        tresp_palette = sns.set_palette(tresp_colors, n_colors=len(tresp_colors))
        axes = plt.subplot2grid((50, 50), (20, 0), rowspan=14, colspan=50)
        sns.scatterplot(x=resp_df.trial, y=resp_df.response_x, hue=resp_df.trial_result, palette=tresp_palette, s=30, ax=axes)
        axes.hlines(y=[200], xmin=0, xmax=total_trials + 1, color='gray', linestyle=':')
        axes.set_ylabel('$Responses\ (r_{t})\ (mm)%$', label_kwargs)
        axes.set_xlim(-1, total_trials + 1)
        sns.despine()

        # ERRORS TRIAL INDEX
        axes = plt.subplot2grid((50, 50), (36, 0), rowspan=14, colspan=50)
        sns.scatterplot(x=resp_df.trial, y=resp_df.error_x, hue=resp_df.trial_result, palette=tresp_palette, s=30, ax=axes)
        axes.hlines(y=[0], xmin=0, xmax=total_trials + 1, color='gray', linestyle=':')
        axes.hlines(y=[0.5, 1], xmin=0, xmax=total_trials + 1, color='gray', linestyle=':')
        axes.set_ylabel('$Errors\ (r_{t}-x_{t})\ (mm)%$')
        axes.set_xlim(-1, total_trials + 1)
        sns.despine()

        # SAVING AND CLOSING PAGE
        pdf.savefig()
        plt.close()

        # PAGE 3:
        plt.figure(figsize=(11.7, 11.7))  # A4 vertical

        states_list = ['STATE_Correct_first_START', 'STATE_Correct_other_START', 'lick_time_stop', 'STATE_Miss_START',
                       'STATE_Punish_START']
        for idx, state in enumerate(states_list):
            resp_df[state] = resp_df[state] - resp_df['STATE_Response_window_START']

        # RASTER PLOT
        x_min = -3
        x_max = 40
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=42, colspan=25)
        sns.scatterplot(x=resp_df.lick_time_stop, y=resp_df.trial, color=water_c, s=20, ax=axes)
        sns.scatterplot(x=resp_df.STATE_Correct_first_START, y=resp_df.trial, style=resp_df.trial_type, color=correct_first_c, s=20, ax=axes)
        sns.scatterplot(x=resp_df.STATE_Correct_other_START, y=resp_df.trial, style=resp_df.trial_type, color=correct_other_c, s=20, ax=axes)
        sns.scatterplot(x=resp_df.STATE_Miss_START, y=resp_df.trial, style=resp_df.trial_type, color=miss_c, s=20, ax=axes)
        sns.scatterplot(x=resp_df.STATE_Punish_START, y=resp_df.trial, style=resp_df.trial_type, color=punish_c, s=20, ax=axes)

        df['stim_duration_align'] = df['stim_duration'] - df['fixation_time']
        axes.barh(list(df.trial), width=100, color=lines2_c, left=-2, height=0.5, alpha=0.2, zorder=0)  # horizontal bars each trial
        axes.barh(list(df.trial), width=df.stim_duration_align, color=lines_c, left=0, height=0.7, alpha=0.2, zorder=0)  # horizontal bars signal stim duration
        axes.axvline(x=0, color=lines_c, linewidth=1.5, zorder=10)

        axes.set_ylim(-1, total_trials + 1)
        axes.set_xlabel('')
        axes.xaxis.set_ticklabels([])
        axes.set_ylabel('Trials', label_kwargs)
        axes.set_xlim(x_min, x_max)
        axes.get_legend().remove()

        #  HISTOGRAM OF LATENCIES
        axes = plt.subplot2grid((50, 50), (43, 0), rowspan=7, colspan=25)
        size_bins = 1
        bins = np.arange(0, x_max, size_bins)
        sns.distplot(resp_df.lick_time_stop, kde=False, bins=bins, color=water_c, ax=axes,  hist_kws={'alpha': 0.8, 'histtype':'step', 'linewidth':2})
        sns.distplot(resp_df.STATE_Correct_first_START,  kde=False, bins=bins, color=correct_first_c, ax=axes,  hist_kws={'alpha': 0.8, 'histtype':'step', 'linewidth':2})
        # sns.distplot(resp_df.STATE_Correct_other_START, bins=bins, color=correct_other_c, ax=axes,  hist_kws={'alpha': 0.8, 'histtype':'step', 'linewidth':2})

        axes.axvline(x=0, color=lines_c, linewidth=1.5, zorder=10)

        axes.set_xlim(x_min, x_max)
        axes.set_xlabel('Latency (sec)', label_kwargs)
        axes.set_ylabel('Number of pokes', label_kwargs)

        # add correct stimulus lenghts in the raster
        # add incorrects

        # ERRORS TRIAL INDEX
        axes = plt.subplot2grid((50, 50), (0, 25), rowspan=42, colspan=25)

        sns.scatterplot(x=resp_df.error_x, y=resp_df.trial, hue=resp_df.trial_result, style=resp_df.trial_type,
                        palette=tresp_palette, s=20, ax=axes, zorder=20)
        axes.barh(list(df.trial), width=800, color=lines2_c, left=-400, height=0.7, alpha=0.2, zorder=0)
        axes.vlines(x=[-wm_correct_th, -vg_correct_th, vg_correct_th, wm_correct_th], ymin=0, ymax=total_trials + 1,
                    linewidth=1.5, color=wm_th_c, zorder=10)
        axes.axvline(x=0, color=stim_c, linewidth=1.5, zorder=10)

        axes.set_xlabel('')
        axes.set_ylabel('')
        axes.yaxis.set_ticklabels([])
        axes.xaxis.set_ticklabels([])
        axes.set_ylim(-1, total_trials + 1)
        axes.legend(loc='center', bbox_to_anchor=(0.9, 0.9))
        sns.despine()


        # ERRORS HISTOGRAMS
        axes = plt.subplot2grid((50, 50), (43, 25), rowspan=7, colspan=25)
        sns.distplot(resp_df.error_x, kde=False, bins=bins_err, ax=axes, hist_kws={'alpha': 0.9, 'histtype':'step', 'linewidth':2})

        axes.axvline(x=0, color=stim_c, linewidth=1.5)
        correct_th = ttype_df['correct_th'].iloc[0]
        axes.axvline(x=-correct_th, color=wm_th_c, linewidth=1.5)
        axes.axvline(x=correct_th, color=wm_th_c,  linewidth=1.5)

        axes.set_xlabel('$Errors\ (r_{t}-x_{t})\ (mm)%$', label_kwargs)
        axes.set_ylabel('')
        axes.yaxis.set_ticklabels([])
        axes.set_xlim(-400, 450)
        sns.despine()

        # fix colors, incorrects pls!




        # SAVING AND CLOSING PAGE
        pdf.savefig()
        plt.close()






        # TRIALS VS DISTANCE
        # axes = plt.subplot2grid((50, 50), (14, 0), rowspan=25, colspan=50)
        #
        # correct_responses_wm_first_df = resp_df[resp_df.wm_bool & (resp_df.incorrect_attempts == 0) &
        #                                         (resp_df.trial_result == 1)]
        # correct_responses_wm_other_df = resp_df[resp_df.wm_bool & (resp_df.incorrect_attempts == 0) &
        #                                         (resp_df.trial_result > 1)]
        # correct_responses_wm_none_df = resp_df[resp_df.wm_bool & (resp_df.incorrect_attempts > 0)]
        #
        # correct_responses_vg_first_df = resp_df[(resp_df.wm_bool == False) & (resp_df.incorrect_attempts == 0) &
        #                                         (resp_df.trial_result == 1)]
        # correct_responses_vg_other_df = resp_df[(resp_df.wm_bool == False) & (resp_df.incorrect_attempts == 0) &
        #                                         (resp_df.trial_result > 1)]
        # correct_responses_vg_none_df = resp_df[(resp_df.wm_bool == False) & (resp_df.incorrect_attempts > 0)]
        #
        # axes.scatter(correct_responses_wm_first_df.x_errors, correct_responses_wm_first_df.trial, s=3,
        #              marker='^', c=first_correct_c)
        # axes.scatter(correct_responses_vg_first_df.x_errors, correct_responses_vg_first_df.trial, s=3,
        #              marker='o', c=first_correct_c)
        #
        # axes.scatter(correct_responses_wm_other_df.x_errors, correct_responses_wm_other_df.trial, s=3,
        #              marker='^', c=correct_c)
        # axes.scatter(correct_responses_vg_other_df.x_errors, correct_responses_vg_other_df.trial, s=3,
        #              marker='o', c=correct_c)
        #
        # axes.scatter(correct_responses_wm_none_df.x_errors, correct_responses_wm_none_df.trial, s=3,
        #              marker='^', c=correct_responses_wm_none_df.incorrect_attempts, cmap='OrRd_r')
        # axes.scatter(correct_responses_vg_none_df.x_errors, correct_responses_vg_none_df.trial, s=3,
        #              marker='o', c=correct_responses_vg_none_df.incorrect_attempts, cmap='OrRd_r')
        #
        # axes.hlines(y=df.trial, xmin=-200, xmax=200, color=lines_c, linewidth=0.2, linestyle=':', zorder=0)
        # axes.vlines(x=[-params.vg_th_diameter / 2, params.vg_th_diameter / 2], ymin=0, ymax=total_trials + 1,
        #             color=vg_th_diam_c, zorder=0)
        # axes.vlines(x=[-params.wm_th_diameter / 2, params.wm_th_diameter / 2], ymin=0, ymax=total_trials + 1,
        #             color=wm_th_diam_c, zorder=0)
        # axes.vlines(x=[-params.stim_diameter / 2, params.stim_diameter / 2], ymin=0, ymax=total_trials + 1,
        #             color=stim_diam_c, zorder=0)
        #
        # axes.set_xlim(-200, 200)
        # axes.set_xlabel('$Error\ (r_{t}\ -\ x_{t})\ (mm)$', label_kwargs)
        # axes.set_ylabel('Trials', label_kwargs)
        #
        # colors = [first_correct_c, incorrect_c, stim_diam_c, vg_th_diam_c, vg_th_diam_c, other_c, other_c]
        # widths = [0, 0, 1, 1, 1, 0, 0]
        # markers = ['s', 's', ',', ',', ',', 'o', '^']
        # labels = ['Correct', 'Incorrect', 'Stimulus', 'VG threshold', 'WM threshold', 'Visually guided', "WM guided"]
        # lines = [Line2D([0], [0], color=colors[i], marker=markers[i], markersize=6,
        #                 markerfacecolor=colors[i], linewidth=widths[i]) for i in range(len(colors))]
        # axes.legend(lines, labels, fontsize=6)



        # # REPOKE COUNTS PER TRIAL
        # axes = plt.subplot2grid((50, 50), (11, 26), rowspan=10, colspan=24)
        # sns.countplot(x='response_x_index', hue='trial_type', data=resp_df, palette=ttype_palette)
        # axes.set_xlabel('Number of responses')
        # axes.get_legend().remove()
        # axes.set_xlim(-1, 15)
        # sns.despine()
