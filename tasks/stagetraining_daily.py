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
lines_c_list = [lines_c, lines2_c]

vg_c = '#393b79'
wmi_c = '#6b6ecf'
wmds_c = '#9c9ede'
wmdm_c = '#ce6dbd'
wmdl_c = '#a55194'

stim_c = 'gold'
correct_th_c = 'green'
repoke_th_c = 'orangered'

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

    # FIXATION TIME WHEN IS EMPTY
    df['fixation_time'] = df['fixation_time'].fillna(1)

    # FIX DELAY TYPE IN THE FIRST SESSIONS
    if 'delay_type' not in df.columns:
        df['delay_type'] = np.nan
        df.loc[((df.trial_type == 'WM_D') & (df.delay == 0)), 'delay_type'] = 'DS'
        df.loc[((df.trial_type == 'WM_D') & (df.delay == 0.5)), 'delay_type'] = 'DM'
        df.loc[((df.trial_type == 'WM_D') & (df.delay == 1)), 'delay_type'] = 'DL'

    # FIX TTYPES
    if (df['pwm_ds'] > 0).any():
        df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DS')), 'trial_type'] = 'WM_Ds'
    if (df['pwm_dm'] > 0).any():
        df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DM')), 'trial_type'] = 'WM_Dm'
    if (df['pwm_dl'] > 0).any():
        df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DL')), 'trial_type'] = 'WM_Dl'

    # FIX DELAYS
    df['delay_o'] = df['delay'] #keep original delay values
    df.loc[(df.trial_type == 'VG'), 'delay'] = -1      # VG trials
    df.loc[(df.trial_type == 'WM_I'), 'delay'] = -0.5  # WMI trials

    ttypes = df.trial_type.unique().tolist()
    # dtypes = df.delay_type.unique().tolist()
    treslt = df.trial_result.unique().tolist()

    #correct the order of the lists
    if 'WM_Ds' in ttypes:
        idx = ttypes.index('WM_Ds')
        ttypes.pop(idx)
        ttypes.insert(1, "WM_Ds")
        if 'WM_I' in ttypes:
            if 'VG' in ttypes:
                idx = ttypes.index('WM_I')
                ttypes.pop(idx)
                ttypes.insert(1, "WM_I")

    # RELEVANT COLUMNS
    ###colors to plot columns
    df['ttype_colors'] = vg_c
    df.loc[(df.trial_type == 'WM_I', 'ttype_colors')] = wmi_c
    df.loc[(df.trial_type == 'WM_Ds', 'ttype_colors')] = wmds_c
    df.loc[(df.trial_type == 'WM_Dm', 'ttype_colors')] = wmdm_c
    df.loc[(df.trial_type == 'WM_Dl', 'ttype_colors')] = wmdl_c
    ttype_colors = df.ttype_colors.unique().tolist()

    df['treslt_colors'] = miss_c
    df.loc[(df.trial_result == 'correct_first', 'treslt_colors')] = correct_first_c
    df.loc[(df.trial_result == 'correct_other', 'treslt_colors')] = correct_other_c
    df.loc[(df.trial_result == 'incorrect', 'treslt_colors')] = incorrect_c
    df.loc[(df.trial_result == 'punish', 'treslt_colors')] = punish_c
    treslt_colors = df.treslt_colors.unique().tolist()

    # CONVERT STRINGS TO LISTS
    df = utils.convert_strings_to_lists(df, ['response_x', 'response_y', 'STATE_Incorrect_START', 'STATE_Incorrect_END',
                                             'STATE_Fixation_START', 'STATE_Fixation_END', 'STATE_Fixation_break_START',
                                             'STATE_Fixation_break_END', 'STATE_Response_window2_START',
                                             'STATE_Response_window2_END'])

    # CALCULATE LATENCIES
    #add nans to empty list, if not error
    df['STATE_Response_window2_END'] = df['STATE_Response_window2_END'].apply(lambda x: [np.nan] if len(x) == 0 else x)
    df['response_window_end'] = df['STATE_Response_window2_END'].apply(lambda x: x[-1])
    df['response_window_end'] = df['response_window_end'].fillna(df['STATE_Response_window_END'])
    df['reward_time'] = df['STATE_Correct_first_reward_START'].fillna(0) + df[
        'STATE_Correct_other_reward_START'].fillna(0) + df['STATE_Miss_reward_START'].fillna(0)
    df['lick_latency'] = df['reward_time'] - df['response_window_end']

    # CREATE REPONSES DF
    ### needed columns before the unnest
    df['responses_time'] = df.apply(lambda row: utils.create_responses_time(row), axis=1)
    df['response_result'] = df.apply(lambda row: utils.create_reponse_result(row), axis=1)
    ### unnest
    resp_df = utils.unnesting(df, ['response_x', 'response_y', 'responses_time', 'response_result'])

    # RELEVANT COLUMNS
    resp_df['resp_latency'] = resp_df['responses_time'] - resp_df['STATE_Response_window_START']
    resp_df['error_x'] = resp_df['response_x'] - resp_df['x']
    resp_df['abs_error_x'] = resp_df['error_x'].abs()
    resp_df['correct_bool'] = np.where(resp_df['correct_th'] / 2 >= resp_df['error_x'].abs(), 1, 0)
    resp_df.loc[(resp_df.trial_result == 'miss', 'correct_bool')] = np.nan
    # Correct_bool column: 1 correct; 0 incorrects/punish; nan miss

    ###results colors
    resp_df['rreslt_colors'] = miss_c
    resp_df.loc[(resp_df.response_result == 'correct_first', 'rreslt_colors')] = correct_first_c
    resp_df.loc[(resp_df.response_result == 'correct_other', 'rreslt_colors')] = correct_other_c
    resp_df.loc[(resp_df.response_result == 'incorrect', 'rreslt_colors')] = incorrect_c
    resp_df.loc[(resp_df.response_result == 'punish', 'rreslt_colors')] = punish_c
    rreslt_colors = resp_df.rreslt_colors.unique().tolist()

    # SUBDATAFRAMES
    first_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='first', inplace=False).copy()
    last_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='last', inplace=False).copy()

    # RELEVANT VARIABLES
    subject = df.subject.iloc[0]
    weight = df.subject_weight.iloc[0]
    total_trials = int(df.trial.iloc[-1])
    valid_trials = (df.loc[df['trial_result'] != 'miss']).shape[0]
    missed_trials = total_trials - valid_trials
    reward_drunk =  int(df.reward_drunk.iloc[-1])
    task = df.task.iloc[0]
    try:
        stage = df.stage.iloc[0]
    except:
        stage = np.nan
    try:
        substage = df.substage.iloc[0]
    except:
        substage = np.nan

    # THRESHOLDS & CHANCE
    stim_width = df.width.iloc[0]/2
    vg_correct_th = df.correct_th.unique()[0]/2
    vg_repoke_th = df.repoke_th.unique()[0]/2
    vg_chance_p = utils.chance_calculation(vg_correct_th)
    threshold_list = [vg_correct_th]
    chance_list = [vg_chance_p]
    lines_list = [stim_width, vg_correct_th, vg_repoke_th]
    lines_list_colors = [stim_c, correct_th_c, repoke_th_c]

    if len(df.correct_th.unique()) > 1:
        wm_correct_th = df.correct_th.unique()[1]/2
        wm_chance_p = utils.chance_calculation(wm_correct_th)
        threshold_list.append(wm_correct_th)
        chance_list.append(wm_chance_p)
        lines_list.append(wm_correct_th)
        lines_list_colors.append(correct_th_c)
    if len(df.repoke_th.unique()) > 1:
        wm_repoke_th = df.repoke_th.unique()[1]/2
        lines_list.append(wm_repoke_th)
        lines_list_colors.append(repoke_th_c)

    #total accuracies
    total_acc_first_poke = first_resp_df.correct_bool.mean()
    total_acc_last_poke = last_resp_df.correct_bool.mean()

    total_acc_dict = {}
    for ttype in ttypes:
        ttype_df =  first_resp_df.loc[first_resp_df['trial_type'] == ttype]
        single_acc = ttype_df.correct_bool.mean() if ttype_df.shape[0] != 0 else 0.0
        total_acc_dict[ttype] = single_acc

    total_acc_ttype = ''
    for key, value in total_acc_dict.items():
        total_acc_ttype = total_acc_ttype + '  /  Acc ' + str(key) + ': ' + str(int(value * 100)) + "%"


    # PAGE 1:
    with PdfPages(save_path) as pdf:

        plt.figure(figsize=(8.3, 11.7))  # A4 vertical

        # HEADER
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=10, colspan=50)
        s1 = ('Subject: ' + str(subject) +
              '  /  Task: ' + str(df.task.iloc[0]) +
              '  /  Date: ' + str(date) +
              '  /  Box: ' + str(df.box.iloc[0]) +
              '  /  Stage: ' + str(int(stage)) +
              '  /  Substage: ' + str(int(substage)) + '\n')

        s3 = ('Prob VG: ' + str(round(df.pvg.mean() * 100, 2))  +
              '  /  Prob WMI: ' + str(round(df.pwm_i.mean() * 100, 2)) +
              '  /  Prob WMDs: ' + str(round(df.pwm_ds.mean() * 100, 2)) +
              '  /  Prob WMDm: ' + str(round(df.pwm_dm.mean() * 100, 2)) +
              '  /  Prob WMDl: ' + str(round(df.pwm_dl.mean() * 100, 2)) + '\n')

        s4 = ('Delay s: ' + str((df.ds.iloc[0])) +
              '  /  Delay m: ' + str((df.dm.iloc[0])) +
              '  /  Delay l: ' + str((df.dl.iloc[0])) + '\n')

        s2 = ('Total trials: ' + str(int(total_trials)) +
              '  /  Valids: ' + str(valid_trials) +
              '  /  Missed: ' + str(missed_trials) +
              '  /  Reward drunk: ' + str(reward_drunk) + " ul" +
              '  /  Weight: ' + str(weight) + " g" +
              '  /  Rel. weight: ' + str(round(utils.relative_weights(subject, weight), 2)) + "%" + '\n')

        s5 = ('Acc first poke: ' + str(int(total_acc_first_poke * 100)) + '%' + 
              '  /  Acc last poke: ' + str(int(total_acc_last_poke * 100)) + "%" + '\n')

        s6 = (total_acc_ttype +  '\n')

        axes.text(0.1, 0.9, s1+s2+s3+s4+s5+s6, fontsize=8, transform=plt.gcf().transFigure)

        # ACCURACY VS TRIAL INDEX PLOT
        # we use the same axes than header
        first_resp_df['acc'] = utils.compute_window(first_resp_df.correct_bool, 20)
        last_resp_df['acc'] = utils.compute_window(last_resp_df.correct_bool, 20)

        sns.scatterplot(x=first_resp_df.trial, y=first_resp_df.acc, s=20, ax=axes, color=correct_first_c)
        sns.lineplot(x=first_resp_df.trial, y=first_resp_df.acc, ax=axes, color= correct_first_c)
        sns.scatterplot(x=last_resp_df.trial, y=last_resp_df.acc, s=20, ax=axes, color=correct_other_c)
        sns.lineplot(x=last_resp_df.trial, y=last_resp_df.acc, ax=axes, color=correct_other_c)
        axes.hlines(y=[0.5, 1], xmin=0, xmax=total_trials, color=lines_c, linestyle=':')
        for idx, i in enumerate(chance_list):
            axes.fill_between(df.trial, chance_list[idx], 0, facecolor=lines_c_list[idx], alpha=0.3)
        df.loc[(df.trial_type == 'WM_I', 'ttype_colors')] = wmi_c

        axes.set_xlabel('Trial')
        axes.set_xlim([1, total_trials + 1])
        utils.axes_pcent(axes, label_kwargs)

        colors = [correct_first_c, correct_other_c]
        labels = ['First poke', 'Last poke']
        lines = [Line2D([0], [0], color=c, marker='o', markersize=6, markerfacecolor=c) for c in colors]
        axes.legend(lines, labels, fontsize=8, loc='center', bbox_to_anchor=(0.9, 1.1))

        # ACCURACY TRIAL TYPE
        axes = plt.subplot2grid((50, 50), (15, 0), rowspan=12, colspan=21)

        sns.pointplot(x=first_resp_df.trial_type, y=first_resp_df.correct_bool, s=20, ax=axes,
                      color=correct_first_c, order=ttypes)
        sns.pointplot(x=last_resp_df.trial_type, y=last_resp_df.correct_bool, s=20, ax=axes,
                      color=correct_other_c, order=ttypes)
        axes.hlines(y=[0.5, 1], xmin=0, xmax=len(ttypes) - 1, color=lines_c, linestyle=':')
        chance_list2 = chance_list.copy()
        print(df.trial_type.unique())

        for idx, i in enumerate(df.trial_type.unique()):
            if idx >= 1:
                if len(chance_list) > 1 and idx > 1:
                    chance_list2.append(chance_list[1])
                elif len(chance_list) == 1:
                    chance_list2.append(chance_list[0])
        print(chance_list)
        print(ttypes)
        axes.fill_between(ttypes, chance_list2, 0, facecolor=lines_c, alpha=0.3)
        axes.set_xlabel('')
        utils.axes_pcent(axes, label_kwargs)

        # STD TRIAL TYPE
        axes = plt.subplot2grid((50, 50), (15, 26), rowspan=12, colspan=25)

        sns.pointplot(x=first_resp_df.trial_type, y=first_resp_df.error_x, s=20, ax=axes,
                      color=correct_first_c, order=ttypes, estimator=np.std)
        sns.pointplot(x=last_resp_df.trial_type, y=last_resp_df.error_x, s=20, ax=axes,
                      color=correct_other_c, order=ttypes, estimator=np.std)
        axes.hlines(y=[stim_width], xmin=0, xmax=len(ttypes) - 1, color=stim_c, linestyle=':')

        for th in threshold_list:
            axes.hlines(y=th, xmin=0, xmax=len(ttypes) - 1, color=correct_first_c, linestyle=':')
        axes.hlines(y=[vg_repoke_th], xmin=0, xmax=len(ttypes) - 1, color=repoke_th_c, linestyle=':')
        axes.fill_between(ttypes, stim_width, 0, facecolor=stim_c, alpha=0.2)  # chance
        axes.fill_between(ttypes, 160, 155, facecolor=lines_c, alpha=0.3) #chance

        axes.set_xlabel('')
        axes.set_ylabel('STD (mm)', label_kwargs)


        # RESPONSE LATENCIES   # we look to all the reponses time
        axes = plt.subplot2grid((50, 50), (31, 0), rowspan=17, colspan=22)
        ttype_palette = sns.set_palette(ttype_colors, n_colors=len(ttype_colors))

        y_max = 12
        order = sorted(treslt)
        sns.boxplot(x='response_result', y='resp_latency', hue='trial_type', data=resp_df, color='white',
                        showfliers=False, ax=axes, order=order)
        sns.stripplot(x="response_result", y="resp_latency", hue='trial_type', data=resp_df, dodge=True, ax=axes,
                      order=order)
        axes.set_xticklabels(order, fontsize=9, rotation=35)
        axes.set_ylabel("Response latency (sec)", label_kwargs)
        axes.set_xlabel("")
        axes.set_ylim(0, y_max)
        axes.get_legend().remove()

        # LICKPORT LATENCIES   # we look only the trial time
        axes = plt.subplot2grid((50, 50), (31, 27), rowspan=17, colspan=23)

        sns.boxplot(x='trial_result', y='lick_latency', hue='trial_type', data=df, color='white', showfliers=False,
                        ax=axes, order=order)
        sns.stripplot(x="trial_result", y="lick_latency", hue='trial_type', data=df, dodge=True,
                      ax=axes, order=order)

        axes.set_xticklabels(order, fontsize=9, rotation=35)
        axes.set_ylabel("Lickport latency (sec)", label_kwargs)
        axes.set_xlabel("")
        axes.set_ylim(0, y_max)
        axes.get_legend().remove()
        sns.despine()


        # SAVING AND CLOSING PAGE
        pdf.savefig()
        plt.close()


        # PAGE 2:
        plt.figure(figsize=(11.7, 11.7))  # apaisat

        #align to the reponse window onset
        states_list = ['STATE_Correct_first_START', 'STATE_Correct_other_START','STATE_Miss_START',
                       'STATE_Punish_START', 'reward_time', 'responses_time']
        for idx, state in enumerate(states_list):
            resp_df[state] = resp_df[state] - resp_df['STATE_Response_window_START']

        # horizontal bars needed for raster plot
        resp_df['stim_duration_align'] = resp_df['stim_duration'] - resp_df['fixation_time']
        # cut stim duration in VG when reponse is correct
        resp_df.loc[
            ((resp_df.trial_type == 'VG') & (resp_df.response_result == 'correct_first')), 'stim_duration_align'] = \
            resp_df['responses_time']
        resp_df.loc[
            ((resp_df.trial_type == 'VG') & (resp_df.response_result == 'correct_other')), 'stim_duration_align'] = \
            resp_df['responses_time']
        resp_df.loc[((resp_df.trial_type == 'VG') & (resp_df.response_result == 'punish')), 'stim_duration_align'] = \
            resp_df['responses_time']

        h_bars = resp_df.drop_duplicates(subset=['trial', 'session'], keep='last', inplace=False).copy()
        h_bars['origin'] = h_bars['fixation_time'] + h_bars['delay_o']
        origin = h_bars.origin.tolist()
        origin = [-i for i in origin]

        # misses
        miss_df = df.loc[df['trial_result'] == 'miss', :]
        origin_m = miss_df.fixation_time.tolist()
        origin_m = [-i for i in origin_m]

        # RASTER PLOT
        x_min = -3
        x_max = 10
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=42, colspan=25)
        rreslt_palette = sns.set_palette(rreslt_colors, n_colors=len(rreslt_colors))

        sns.scatterplot(x=resp_df.reward_time, y=resp_df.trial, color=water_c, s=20, ax=axes, label='water')
        sns.scatterplot(x=resp_df.responses_time, y=resp_df.trial, hue=resp_df.response_result,
                        style=resp_df.trial_type, s=20, ax=axes)

        # horizontal lines
        axes.barh(list(df.trial), width=100, color=lines2_c, left=-2, height=0.7, alpha=0.4,
                  zorder=0)  # horizontal bars each trial
        axes.barh(list(h_bars.trial), width=h_bars.stim_duration_align, color=lines_c, left=0, height=0.7, alpha=0.3,
                  zorder=0)  # horizontal bars signal stim duration after RW onset
        axes.barh(list(h_bars.trial), width=h_bars.fixation_time, color=lines_c, left=origin, height=0.7,
                  alpha=0.3, zorder=0)  # horizontal bars signal stim duration before RW onset
        axes.axvline(x=0, color=lines_c, linewidth=1.5, zorder=10)
        ### misses
        axes.barh(list(miss_df.trial), width=miss_df.stim_duration, color=lines_c, left=-1, height=0.7, alpha=0.2,
                  zorder=0)  # horizontal bars signal stim duration after RW onset
        axes.barh(list(miss_df.trial), width=miss_df.stim_duration, color=lines_c, left=origin_m, height=0.7,
                  alpha=0.2, zorder=0)  # horizontal bars signal stim duration before RW onset

        axes.set_ylim(-1, total_trials + 1)
        axes.set_xlabel('')
        axes.xaxis.set_ticklabels([])
        axes.set_ylabel('Trials', label_kwargs)
        axes.set_xlim(x_min, x_max)
        axes.get_legend().remove()

        # HISTOGRAM OF LATENCIES
        axes = plt.subplot2grid((50, 50), (43, 0), rowspan=7, colspan=25)
        size_bins = 0.4
        bins = np.arange(0, x_max, size_bins)
        sns.distplot(resp_df.reward_time, kde=False, bins=bins, color=water_c, ax=axes,
                     hist_kws={'alpha': 0.8, 'histtype': 'step', 'linewidth': 2})
        for respres, respres_df in resp_df.groupby('response_result'):
            colors = respres_df.rreslt_colors.iloc[0]
            sns.distplot(respres_df.responses_time, kde=False, bins=bins, color=colors, ax=axes,
                         hist_kws={'alpha': 0.8, 'histtype': 'step', 'linewidth': 2})

        axes.axvline(x=0, color=lines_c, linewidth=1.5, zorder=10)
        axes.set_xlim(x_min, x_max)
        axes.set_xlabel('Latency (sec)', label_kwargs)
        axes.set_ylabel('Number of pokes', label_kwargs)

        # ERRORS TRIAL INDEX
        axes = plt.subplot2grid((50, 50), (0, 25), rowspan=42, colspan=25)
        x_min = -200
        x_max = 200
        sns.scatterplot(x=resp_df.error_x, y=resp_df.trial, hue=resp_df.response_result, style=resp_df.trial_type,
                        s=20, ax=axes, zorder=20)
        axes.barh(list(df.trial), width=800, color=lines2_c, left=-400, height=0.7, alpha=0.4, zorder=0)
        axes.axvspan(-stim_width, stim_width, color=stim_c,  alpha=0.1)

        #vertical lines
        neg_lines_list = [-x for x in lines_list]
        all_lines = lines_list + neg_lines_list
        all_colors = lines_list_colors + lines_list_colors

        for idx, line in enumerate(all_lines):
            axes.axvline(x=line, color=all_colors[idx], linestyle=':', linewidth=1)
        axes.axvspan(-stim_width, stim_width, color=stim_c,  alpha=0.1)

        axes.set_xlabel('')
        axes.set_ylabel('')
        axes.yaxis.set_ticklabels([])
        axes.xaxis.set_ticklabels([])
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(-1, total_trials + 1)
        axes.legend(loc='center', bbox_to_anchor=(1, 1)).set_zorder(10)
        sns.despine()

        # ERRORS HISTOGRAM
        axes = plt.subplot2grid((50, 50), (43, 25), rowspan=7, colspan=25)
        bins = np.linspace(-r_edge, r_edge, 41)
        sns.distplot(resp_df.error_x, kde=False, bins=bins, color=lines_c, ax=axes,
                     hist_kws={'alpha': 0.9, 'histtype': 'step', 'linewidth': 2})

        #vertical lines
        for idx, line in enumerate(all_lines):
            axes.axvline(x=line, color=all_colors[idx], linestyle=':', linewidth=1)
        axes.axvspan(-stim_width, stim_width, color=stim_c, alpha=0.1)

        axes.set_xlabel('$Errors\ (r_{t}-x_{t})\ (mm)%$', label_kwargs)
        axes.set_ylabel('')
        axes.yaxis.set_ticklabels([])
        axes.set_xlim(x_min, x_max)
        sns.despine()

        # lines legend
        labels = ['Stimulus', 'Correct threshold', 'Repoke threshold']
        colors = [stim_c, correct_th_c, repoke_th_c]
        lines = [Line2D([0], [0], linestyle='dotted', color=colors[i], marker=',', markersize=6,
                        markerfacecolor=colors[i]) for i in range(len(colors))]
        axes.legend(lines, labels, fontsize=8, loc='center', bbox_to_anchor=(1, 6.25))


        # SAVING AND CLOSING PAGE
        pdf.savefig()
        plt.close()


        # PAGE 3
        plt.figure(figsize=(11.7, 11.7))

        # ACCURACY TRIAL INDEX & TRIAL TYPE
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=8, colspan=50)

        ttype_palette = sns.set_palette(ttype_colors, n_colors=len(ttype_colors))
        for ttype, ttype_df in first_resp_df.groupby('trial_type'):
            ttype_color = ttype_df.ttype_colors.iloc[0]
            ttype_df['acc'] = utils.compute_window(ttype_df.correct_bool, 20)
            a = sns.scatterplot(x=ttype_df.trial, y=ttype_df.acc, s=20, ax=axes, color=ttype_color, label=ttype)
            sns.lineplot(x=ttype_df.trial, y=ttype_df.acc, ax=axes, color=ttype_color, label=ttype)

        axes.hlines(y=[0.5, 1], xmin=0, xmax=total_trials, color=lines_c, linestyle=':')
        for idx, i in enumerate(chance_list):
            axes.fill_between(df.trial, chance_list[idx], 0, facecolor=lines_c_list[idx], alpha=0.3)

        axes.set_xlabel('')
        axes.set_xlim([1, total_trials + 1])
        utils.axes_pcent(axes, label_kwargs)

        colors = ttype_colors
        labels = df.trial_type.unique()
        lines = [Line2D([0], [0], color=c, marker='o', markersize=7, markerfacecolor=c) for c in colors]
        axes.legend(lines, labels, fontsize=8, title="Trial type", bbox_to_anchor=(1.05, 0.9), loc='center')

        # STD TRIAL INDEX & TRIAL TYPE
        axes = plt.subplot2grid((50, 50), (10, 0), rowspan=8, colspan=50)

        for ttype, ttype_df in first_resp_df.groupby('trial_type'):
            ttype_color = ttype_df.ttype_colors.iloc[0]
            ttype_df['err'] = utils.compute_window(ttype_df.abs_error_x, 20)
            a = sns.scatterplot(x=ttype_df.trial, y=ttype_df.err, s=20, ax=axes, color=ttype_color, label=ttype)
            sns.lineplot(x=ttype_df.trial, y=ttype_df.err, ax=axes, color=ttype_color, label=ttype)

        axes.hlines(y=155, xmin=0, xmax=total_trials, color=lines_c, linestyle=':')
        axes.set_xlabel('')
        axes.set_ylabel('Abs Error (mm)')
        axes.set_xlim([1, total_trials + 1])
        axes.get_legend().remove()

        # TRIAL TYPE PROBABILITY PROGRESSIONS // STIMULUS LENGHT PLOT // DELAY LENGHT PLOT
        axes = plt.subplot2grid((50, 50), (20, 0), rowspan=4, colspan=50)

        # DELAY LENGHT PLOT
        if task == "StageTraining_2B_V6" and stage == 3 and substage == 2 or task == "StageTraining_MA_V3" and stage == 3\
                or task == "StageTraining_2B_V6" and stage == 4 or task == "StageTraining_2B_V7" and stage == 3:
            dtypes = df.delay_type.unique()
            dtype_colors = []
            for i in dtypes:
                if i == 'DS':
                    dtype_colors.append(wmds_c)
                elif i == 'DM':
                    dtype_colors.append(wmdm_c)
                elif i == 'DL':
                    dtype_colors.append(wmdl_c)

            dtypes_palette = sns.set_palette(dtype_colors, n_colors=len(dtype_colors))
            sns.lineplot(x=df.trial, y=df.delay, hue=df.delay_type, style=df.delay_type, markers=True, ax=axes)
            dl_df = df.loc[df['delay_type']== 'DL']
            if dl_df.shape[0] > 1:
                label = 'Max: ' + str(round(dl_df.delay.max(), 2)) + ' s\n' + \
                        'Min: ' + str(round(dl_df.delay.min(), 2)) + ' s'
                axes.text(1, 0.9, label, transform=axes.transAxes, fontsize=8, fontweight='bold',
                          verticalalignment='top')
            axes.set_ylabel('Delay (sec)', label_kwargs)
            axes.get_legend().remove()


        # STIMULUS DURATION PLOT
        elif task == "StageTraining_2B_V5" or task == "StageTraining_2B_V6" or task == "StageTraining_2B_V7" or task == "StageTraining_MA_V3" and stage == 2:
            for ttype, ttype_df in df.groupby('trial_type'):
                if ttype == 'WM_I':
                    ttype_color = ttype_df.ttype_colors.iloc[0]
                    ttype_df['stim_respwin'] = ttype_df['stim_duration'] - ttype_df['fixation_time']
                    sns.lineplot(x=ttype_df.trial, y=ttype_df.stim_respwin, style=ttype_df.trial_type, markers=True,
                                 ax=axes, color=ttype_color)
                    axes.hlines(y=[0.2, 0.4], xmin=min(ttype_df.trial), xmax=max(ttype_df.trial), color=lines_c,
                                linestyle=':')
                    y_max = ttype_df.stim_respwin.max() + 0.2
                    axes.get_legend().remove()
                    label = 'Max: ' + str(round(ttype_df.stim_respwin.max(), 3)) + ' s\n' + \
                            'Min: ' + str(round(ttype_df.stim_respwin.min(), 3)) + ' s'
                    axes.text(1, 0.9, label, transform=axes.transAxes, fontsize=8, fontweight='bold',
                              verticalalignment='top')
                elif ttype == 'WM_Ds':
                    ttype_color = ttype_df.ttype_colors.iloc[0]
                    sns.lineplot(x=ttype_df.trial, y=ttype_df.delay, style=ttype_df.trial_type, markers=True,
                                 ax=axes, color=ttype_color)
                    axes.get_legend().remove()

            axes.set_ylim([-0.05, y_max])
            axes.set_ylabel('Stim duration \n (sec)', label_kwargs)
            axes.set_xlabel('Trials', label_kwargs)
            axes.set_xlim([1, total_trials + 1])
            axes.set_ylim()

        # PROBS PLOT
        else:
            if task == "StageTraining_2B_V4":
                probs_list = [df.pvg, df.pwm_i, df.pwm_ds, df.pwm_dm, df.pwm_dl]
                df['pwm_d'] = df.pwm_ds + df.pwm_dm + df.pwm_dl
            else:
                probs_list = [df.pvg, df.pwm_i, df.pwm_ds * df.pwm_d, df.pwm_dm * df.pwm_d, df.pwm_dl * df.pwm_d]

            probs_labels = ['VG', 'WM_I', 'WM_Ds', 'WM_Dm', 'WM_Dl']
            probs_colors = [vg_c, wmi_c, wmds_c, wmdm_c, wmdl_c]

            for idx, prob in enumerate(probs_list):
                sns.lineplot(x=df.trial, y=prob, ax=axes, color=probs_colors[idx])
                sns.scatterplot(x=df.trial, y=prob, ax=axes, color=probs_colors[idx], label=probs_labels[idx], s=15)

            axes.set_xlabel('Trials', label_kwargs)
            axes.set_ylabel('Prob \n appearance', label_kwargs)
            axes.set_ylim(-0.1, 1.1)
            axes.set_xlim([1, total_trials + 1])
            axes.get_legend().remove()


        # RESPONSES HISTOGRAMS
        axes_loc = [0, 11, 21, 31, 41]
        for axes_idx, ttype in enumerate(ttypes):
            subset = first_resp_df.loc[first_resp_df['trial_type'] == ttype]
            axes = plt.subplot2grid((50, 50), (32, axes_loc[axes_idx]), rowspan=7, colspan=9)
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
            axes_idx += 1


        #ERRORS HISTOGRAMS
        for axes_idx, ttype in enumerate(ttypes):
            subset = first_resp_df.loc[first_resp_df['trial_type'] == ttype]
            axes = plt.subplot2grid((50, 50), (42, axes_loc[axes_idx]), rowspan=7, colspan=9)
            color = subset.ttype_colors.unique()
            correct_th = (subset.correct_th.unique())/2

            sns.distplot(subset.error_x, kde=False, bins=bins_err, color=color, ax=axes,
                         hist_kws={'alpha': 0.9})
            # vertical lines
            for idx, line in enumerate(all_lines):
                axes.axvline(x=line, color=all_colors[idx], linestyle=':', linewidth=1)
            axes.axvspan(-stim_width, stim_width, color=stim_c, alpha=0.1)

            axes.set_xlabel('$Errors\ (r_{t}-x_{t})\ (mm)%$', label_kwargs)
            axes.set_ylabel('')
            if ttype == 'VG':
                axes.set_ylabel('Nº of touches', label_kwargs)
        sns.despine()


        # SAVING AND CLOSING PAGE
        pdf.savefig()
        plt.close()

        # PAGE 4
        plt.figure(figsize=(11.7, 11.7))


        # ACCURACY STIMULUS POSITION & TRIAL TYPE
        #first poke
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=9, colspan=14)
        ttype_palette = sns.set_palette(ttype_colors, n_colors=len(ttype_colors))

        first_resp_df['xt_bins'] = pd.cut(first_resp_df.x, bins=bins_resp, labels=False, include_lowest=True)
        x_ax_ticks = list(np.linspace(-0.5, 4.5, 5))

        sns.pointplot(x='xt_bins', y="correct_bool", data=first_resp_df, hue='trial_type', s=3)
        axes.hlines(y=[0.5, 1], xmin=min(x_ax_ticks), xmax=max(x_ax_ticks), color=lines_c, linestyle=':')
        for idx, i in enumerate(chance_list):
            axes.fill_between(x_ax_ticks, chance_list[idx], 0, facecolor=lines_c_list[idx], alpha=0.3)

        axes.set_xticks(x_ax_ticks)
        axes.set_xticklabels(['0', '100', '200', '300', '400'])
        axes.set_xlabel('$Stimulus position\ (x_{t})\ (mm)%$', label_kwargs)
        utils.axes_pcent(axes, label_kwargs)
        axes.set_title('First poke', fontsize=11, fontweight='bold')
        axes.get_legend().remove()

        #last poke
        axes = plt.subplot2grid((50, 50), (0, 15), rowspan=9, colspan=14)
        last_resp_df['xt_bins'] = pd.cut(last_resp_df.x, bins=bins_resp, labels=False, include_lowest=True)

        sns.pointplot(x='xt_bins', y="correct_bool", data=last_resp_df, hue='trial_type', s=3)
        axes.hlines(y=[0.5, 1], xmin=min(x_ax_ticks), xmax=max(x_ax_ticks), color=lines_c, linestyle=':')
        for idx, i in enumerate(chance_list):
            axes.fill_between(x_ax_ticks, chance_list[idx], 0, facecolor=lines_c_list[idx], alpha=0.3)

        axes.set_xticks(x_ax_ticks)
        axes.set_xticklabels(['0', '100', '200', '300', '400'])
        axes.set_xlabel('$Stimulus position\ (x_{t})\ (mm)%$', label_kwargs)
        utils.axes_pcent(axes, label_kwargs)
        axes.set_ylabel('')
        axes.yaxis.set_ticklabels([])
        axes.set_title('Last poke', fontsize=11, fontweight='bold')
        axes.get_legend().remove()

        # ERROR VS STIMULUS POSITION
        axes = plt.subplot2grid((50, 50), (0, 33), rowspan=9, colspan=17)

        sns.pointplot(x='xt_bins', y="error_x", data=first_resp_df, hue='trial_type', s=3, ax=axes)
        axes.hlines(y=[-stim_width, stim_width], xmin=min(x_ax_ticks), xmax=max(x_ax_ticks), color=stim_c, linestyle=':')
        axes.fill_between(x_ax_ticks, stim_width, -stim_width, facecolor=stim_c, alpha=0.1)

        axes.set_xticks(x_ax_ticks)
        axes.set_xticklabels(['0', '100', '200', '300', '400'])
        axes.set_title('First poke', fontsize=11, fontweight='bold')
        axes.set_xlabel('$Stimulus position\ (x_{t})\ (mm)%$', label_kwargs)
        axes.set_ylabel('$Error\ (r_{t}\ -\ x_{t})\ (mm)$', label_kwargs)
        axes.get_legend().remove()

        sns.despine()

        # SAVING AND CLOSING PAGE
        pdf.savefig()
        plt.close()

        print('New daily report completed successfully')



        ############################################# OLD USEFUL PLOTS #############################################
        # # RESPONSES TRIAL INDEX
        # axes = plt.subplot2grid((50, 50), (20, 0), rowspan=14, colspan=50)
        # rreslt_palette = sns.set_palette(rreslt_colors, n_colors=len(rreslt_colors))
        #
        # sns.scatterplot(x=resp_df.trial, y=resp_df.response_x, hue=resp_df.trial_result, s=30, ax=axes)
        # axes.hlines(y=[200], xmin=0, xmax=total_trials + 1, color='gray', linestyle=':')
        # axes.set_ylabel('$Responses\ (r_{t})\ (mm)%$', label_kwargs)
        # axes.set_xlim(-1, total_trials + 1)
        # axes.get_legend().remove()
        #
        # # ERRORS TRIAL INDEX
        # axes = plt.subplot2grid((50, 50), (36, 0), rowspan=14, colspan=50)
        # sns.scatterplot(x=resp_df.trial, y=resp_df.error_x, hue=resp_df.trial_result, s=30, ax=axes)
        # axes.hlines(y=[0], xmin=0, xmax=total_trials + 1, color='gray', linestyle=':')
        # axes.hlines(y=[0.5, 1], xmin=0, xmax=total_trials + 1, color='gray', linestyle=':')
        # axes.set_ylabel('$Errors\ (r_{t}-x_{t})\ (mm)%$')
        # axes.set_xlim(-1, total_trials + 1)
        # axes.get_legend().remove()
        # sns.despine()

        # # RESPONSE LATENCY VS TRIAL INDEX PLOT
        # axes = plt.subplot2grid((50, 50), (9, 0), rowspan=8, colspan=50)
        #
        # treslt_palette = sns.set_palette(treslt_colors, n_colors=len(treslt_colors))
        # sns.scatterplot(x=df.trial, y=df.resp_latency, hue=df.trial_result, s=20, ax=axes, zorder=10)
        # sns.lineplot(x=df.trial, y=df.resp_latency, color=lines_c, ax=axes, zorder=0)
        # axes.hlines(y=5, xmin=0, xmax=total_trials, color=lines_c, linestyle=':')
        # axes.set_ylabel('Response latency (sec)', label_kwargs)
        # axes.set_ylim(0, 30)
        # axes.set_xlim(0, total_trials + 1)
        # axes.set_xlabel('')
        # axes.legend(fontsize=8, loc='center', bbox_to_anchor=(1, 1))
        #
        # resp_latency_mean = df.resp_latency.median()
        # lick_latency_mean = df.lick_latency.median()
        #
        # label = 'Median: ' + str(round(resp_latency_mean, 1)) + ' sec'
        # axes.text(0.5, 0.9, label, transform=axes.transAxes, fontsize=8, verticalalignment='top',
        #           bbox=dict(facecolor='white', alpha=0.5))
        #
        # # LICK LATENCY VS TRIAL INDEX PLOT
        # axes = plt.subplot2grid((50, 50), (18, 0), rowspan=8, colspan=50)
        # sns.scatterplot(x=df.trial, y=df.lick_latency, hue=df.trial_result, s=20, ax=axes, zorder=10)
        # sns.lineplot(x=df.trial, y=df.lick_latency, color=lines_c, ax=axes, zorder=0)
        # axes.hlines(y=5, xmin=0, xmax=total_trials, color=lines_c, linestyle=':')
        # axes.set_ylabel('Lick latency (sec)', label_kwargs)
        # axes.set_ylim(0, 30)
        # axes.set_xlabel('Trials')
        # axes.set_xlim(0, total_trials + 1)
        # axes.get_legend().remove()
        #
        # label = 'Median: ' + str(round(lick_latency_mean, 1)) + ' sec'
        # axes.text(0.5, 0.9, label, transform=axes.transAxes, fontsize=8, verticalalignment='top',
        #           bbox=dict(facecolor='white', alpha=0.5))

        # handles, labels = a.get_legend_handles_labels()
        # axes.legend(handles[2:4], labels[2:4], bbox_to_anchor=(0.9, 1), loc='center', fontsize=8)