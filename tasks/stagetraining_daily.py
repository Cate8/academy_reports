import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from academy_reports import utils
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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


def stagetraining_daily (df, save_path, date):

    ##################### PARSE #####################

    ###### RELEVANT VARIABLES ######
    subject = df.subject.iloc[0]
    weight = df.subject_weight.iloc[0]
    total_trials = int(df.trial.iloc[-1])
    valid_trials = (df.loc[df['trial_result'] != 'miss']).shape[0]
    missed_trials = total_trials - valid_trials
    reward_drunk = int(df.reward_drunk.iloc[-1])
    task = df.task.iloc[0]
    stage = df.stage.iloc[0]
    substage = df.substage.iloc[0]
    try:
        df.rename(columns={"mask": "mask_holes"}, inplace=True)  # rename because its name coincides with a function
        mask = int(df.mask_holes.iloc[0])
    except:
        mask = 3


    # THRESHOLDS,  CHANCE & SCREEN BINNING
    stim_width = df.width.iloc[0] / 2
    try:
        correct_th = df.correct_th.iloc[20] / 2
        repoke_th = df.repoke_th.iloc[20] / 2
    except:
        correct_th = df.correct_th.iloc[0] / 2
        repoke_th = df.repoke_th.iloc[0] / 2
    threshold_lines = [stim_width, correct_th,repoke_th]
    threshold_lines_c = [stim_c, correct_th_c, repoke_th_c]

    if mask != 0:
        # chance
        chance_p = 1 / mask
        chance_lines = []
        for i in range(1, mask + 1, 1):
            chance_lines.append(i * chance_p)
        chance_lines = [chance_lines[0], chance_lines[int(mask / 2)], chance_lines[-1]]
        #binning
        x_positions = df.x.unique().tolist()
        x_positions.sort()
        l_edge = int(min(x_positions) - correct_th)
        r_edge = int(max(x_positions) + correct_th)
        bins_resp = np.linspace(l_edge, r_edge, mask + 1)
        bins_err = np.linspace(-r_edge, r_edge, mask*2 + 1) # no està clar

    else:
        # chance
        chance_p = utils.chance_calculation(correct_th)
        chance_lines = [chance_p, 0.5, 1]
        # binning
        l_edge = 0 + stim_width
        r_edge = 403.2 - stim_width # 403.2mm screen size
        bins_resp = np.linspace(l_edge, r_edge, 6)
        bins_err = np.linspace(-r_edge, r_edge, 12)

    ###### CHECK IF REPOKING IS ALLOWED ######
    if repoke_th > correct_th:
        repoking_bool = True
    else:
        repoking_bool = False


    ######  RELEVANT COLUMNS  ######

    # add columns (when absent)
    column_list = ['STATE_Wait_for_fixation_START', 'STATE_Fixation_break_START', 'STATE_Delay_START',
                   'STATE_Doors_START', 'STATE_Correct_first_START', 'STATE_Miss_START', 'STATE_Punish_START',
                   'STATE_After_punish_START', 'STATE_Incorrect_START', 'STATE_Incorrect_END',
                   'STATE_Response_window2_START',  'STATE_Response_window2_END', 'STATE_Correct_other_START',
                   'STATE_Correct_first_reward_START', 'STATE_Correct_other_reward_START', 'STATE_Miss_reward_START',
                   'STATE_Stimulus_offset_START', 'STATE_Re_Start_task_START']
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
        if (df['trial_type'] == 'WM_D').any():
            df.loc[(df.trial_type == 'WM_D', 'ttype_colors')] = wmds_c
    except:
        pass
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



    ###### CONVERT STRINGS TO LISTS ######
    conversion_list = ['STATE_Incorrect_START', 'STATE_Incorrect_END',
                       'STATE_Fixation_START', 'STATE_Fixation_END', 'STATE_Response_window2_START',
                       'STATE_Response_window2_END']


    for idx, column in enumerate(conversion_list):
        try:
            df[column].str.contains(',')  # means that contains multiple values
        except:  # remove from conversion list
            if column != 'STATE_Incorrect_START': # to remove
                conversion_list.remove(column)

    conversion_list.extend(['response_x', 'response_y'])
    df = utils.convert_strings_to_lists(df, conversion_list)


    ######  COLUMNS OPERATIONS ######
    # CALCULATE LATENIES
    #add nans to empty list, if not error
    df['STATE_Response_window2_END'] = df['STATE_Response_window2_END'].apply(lambda x: [np.nan] if len(x) == 0 else x)
    df['response_window_end'] = df['STATE_Response_window2_END'].apply(lambda x: x[-1])
    df['response_window_end'] = df['response_window_end'].fillna(df['STATE_Response_window_END'])
    df['reward_time'] = df['STATE_Correct_first_reward_START'].fillna(0) + df[
        'STATE_Correct_other_reward_START'].fillna(0) + df['STATE_Miss_reward_START'].fillna(0)
    df['lick_latency'] = df['reward_time'] - df['response_window_end']

    #DEFINE TASK TYPE
    task_type = 'classic'
    if '4B' in task or '5B' in task or '6B' in task:
        task_type = 't-maze'

    # CALCULATE STIMULUS DURATION & FIXATION (when required)
    if task_type == 't-maze':
        df['fixation_time'] = df['STATE_Fixation3_END'] - df['STATE_Fixation1_START']

        df.loc[df['trial_type'] == 'VG', 'stim_duration'] = df['response_window_end'] - df['STATE_Fixation1_START']
        df.loc[df['trial_type'] == 'WM_I', 'stim_duration'] = df['STATE_Fixation3_END'] - df[
            'STATE_Fixation1_START']  + df['rw_stim_dur']
        try:
            df.loc[((df['trial_type'] == 'WM_D') & (df['delay_type'] == 'DS')), 'stim_duration'] = df['STATE_Fixation2_END'] - df[
                'STATE_Fixation1_START'] + df['wm_stim_dur']
        except:
            pass
        try:
            df.loc[((df['trial_type'] == 'WM_D') & (df['delay_type'] == 'DL')), 'stim_duration'] = df['STATE_Fixation1_END'] - df[
            'STATE_Fixation1_START']
        except:
            pass


    ###### CREATE RESPONSES DF ######

    # needed columns before the unnest
    df['unnest'] = 1
    df['responses_time'] = df.apply(lambda row: utils.create_responses_time(row, row['unnest']), axis=1)
    df['response_result'] = df.apply(lambda row: utils.create_reponse_result(row, row['unnest']), axis=1)

    ### unnest
    resp_df = utils.unnesting(df, ['response_x', 'response_y', 'responses_time', 'response_result'])


    ######  RELEVANT COLUMNS  ######
    resp_df['resp_latency'] = resp_df['responses_time'] - resp_df['STATE_Response_window_START']
    resp_df['error_x'] = resp_df['response_x'] - resp_df['x']
    resp_df['abs_error_x'] = resp_df['error_x'].abs()
    resp_df['correct_bool'] = np.where(resp_df['correct_th'] / 2 >= resp_df['error_x'].abs(), 1, 0)
    resp_df.loc[(resp_df.trial_result == 'miss', 'correct_bool')] = np.nan
    # Correct_bool column: 1 correct; 0 incorrects/punish; nan miss

    # rresult colors
    resp_df['rreslt_colors'] = miss_c
    resp_df.loc[(resp_df.response_result == 'correct_first', 'rreslt_colors')] = correct_first_c
    resp_df.loc[(resp_df.response_result == 'correct_other', 'rreslt_colors')] = correct_other_c
    resp_df.loc[(resp_df.response_result == 'incorrect', 'rreslt_colors')] = incorrect_c
    resp_df.loc[(resp_df.response_result == 'punish', 'rreslt_colors')] = punish_c

    ###### USEFUL LISTS ######
    ttypes = df.trial_type.unique().tolist()
    ttypes, ttypes_c = utils.order_lists(ttypes, 'ttypes')  # order lists
    treslts = df.trial_result.unique().tolist()
    treslts, treslts_c = utils.order_lists(treslts, 'treslts')  # order lists
    rreslts_c = resp_df.rreslt_colors.unique().tolist()


    ######  CREATE SUBDATAFRAMES  ######
    first_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='first', inplace=False).copy()
    if repoking_bool == True:
        last_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='last', inplace=False).copy()



    ######  TOTAL ACCURACIES CALCULATION  ######
    total_acc_first_poke = int(first_resp_df.correct_bool.mean() * 100)
    if repoking_bool == True:
        total_acc_last_poke = int(last_resp_df.correct_bool.mean() * 100)

    total_acc_dict = {}
    for ttype in ttypes:
        ttype_df =  first_resp_df.loc[first_resp_df['trial_type'] == ttype]
        single_acc = ttype_df.correct_bool.mean() if ttype_df.shape[0] != 0 else 0.0
        total_acc_dict[ttype] = single_acc

    total_acc_ttype = ''
    for key, value in total_acc_dict.items():
        total_acc_ttype = total_acc_ttype + '  /  Acc ' + str(key) + ': ' + str(int(value * 100)) + "%"

    ##################### PLOT #####################

    ############## PAGE 1 ##############
    with PdfPages(save_path) as pdf:

        plt.figure(figsize=(11.7, 11.7))  # apaisat

        #### HEADER
        s1 = ('Subject: ' + str(subject) +
              '  /  Task: ' + str(df.task.iloc[0]) +
              '  /  Date: ' + str(date) +
              '  /  Box: ' + str(df.box.iloc[0]) +
              '  /  Stage: ' + str(int(stage)) +
              '  /  Substage: ' + str(int(substage)) + '\n')

        s2 = ('Total trials: ' + str(int(total_trials)) +
              '  /  Valids: ' + str(valid_trials) +
              '  /  Missed: ' + str(missed_trials) +
              '  /  Weight: ' + str(weight) + " g" +
              '  /  Rel. weight: ' + str(round(utils.relative_weights(subject, weight), 2)) + "%" +
              '  /  Reward drunk: ' + str(reward_drunk) + " ul" + '\n')

        s3 = ('Acc global: ' + str(total_acc_first_poke) + '%' + total_acc_ttype + '\n')


        ### PLOT 0:
        # STIMULUS DURATION VS TRIAL INDEX
        if stage == 2 or stage == 3 and task_type == 't-maze':
            axes = plt.subplot2grid((50, 50), (0, 0), rowspan=4, colspan=39)

            color = wmi_c
            hlines = [0.8, 0.6, 0.4, 0.2, 0]
            var='wm_stim_dur'
            if stage == 2 and task_type == 't-maze':
                var = 'rw_stim_dur'
                hlines = [0.4, 0.3, 0.2, 0.1, 0]
                color=wmds_c

            sns.lineplot(x=first_resp_df.trial, y=first_resp_df[var], marker='o', ax=axes, color=color)
            axes.hlines(y=hlines, xmin=min(first_resp_df.trial), xmax=max(first_resp_df.trial), color=lines2_c, linestyle=':', linewidth=1)
            axes.set_ylabel('Stim lenght (s)')
            axes.set_xlabel('')
            axes.xaxis.set_ticklabels([])
            axes.set_xlim(1, total_trials + 1)

            try:
                subs= first_resp_df.loc[first_resp_df['trial']>25]
                ymin= int(min(var))
                ymax= int(max(subs[var]))
            except:
                ymin = 0
                ymax = 0.8
            axes.set_ylim(ymin - 0.2, ymax + 0.2)

           #legend
            label = 'Max: ' + str(round(ymax, 3))+ ' s\n' + \
                    'Min: ' + str(round(ymin, 3))+ ' s'
            axes.text(0.9, 1.3, label, transform=axes.transAxes, fontsize=8, verticalalignment='top',
                      bbox=dict(facecolor='white', edgecolor=lines2_c, alpha=0.5))

            axes.text(0.1, 0.9, s1 + s2 + s3, fontsize=8, transform=plt.gcf().transFigure) #header
            axes = plt.subplot2grid((50, 50), (5, 0), rowspan=7, colspan=39) #axes for the next plot

        # PROBS VS TRIAL INDEX
        elif stage == 3 and task_type == 'classic':
            axes = plt.subplot2grid((50, 50), (0, 0), rowspan=4, colspan=39)
            # probs calculation
            probs = ['pvg', 'pwm_i', 'pwm_d', 'pwm_ds', 'pwm_dl']
            for prob in probs[:]:
                if prob not in df.columns:
                    probs.remove(prob)
            probs, probs_c = utils.order_lists(probs, 'probs')  # order lists

            for idx, prob in enumerate(probs):
                sns.lineplot(x=df['trial'], y=df[prob], ax=axes, color=probs_c[idx])
                axes.hlines(y=[0, 0.5, 1], xmin=0, xmax=total_trials, color=lines_c, linestyle=':', linewidth=1)
            axes.set_ylabel('Probability', label_kwargs)
            axes.set_ylim(-0.1, 1.1)
            axes.set_xlabel('')
            axes.get_xaxis().set_ticks([])
            axes.set_frame_on(False)

            #label
            try:
                prob_max = round(df.pwm_ds.iloc[21], 3)
                prob_min = round(df.pwm_ds.iloc[-1], 3)
            except:
                try:
                    prob_max = round(df.pwm_d.iloc[21], 3)
                    prob_min = round(df.pwm_d.iloc[-1], 3)
                except:
                    print('session very short!')
                    try:
                        prob_max = round(df.pwm_ds.iloc[-1], 3)
                        prob_min = prob_max
                    except:
                        prob_max = round(df.pwm_d.iloc[-1], 3)
                        prob_min = prob_max

            label = 'Init: ' + str(prob_max) + '\n' + \
                    'Final: ' + str(prob_min)
            axes.text(0.9, 1.3, label, transform=axes.transAxes, fontsize=8, verticalalignment='top',
                      bbox=dict(facecolor='white', edgecolor=lines2_c, alpha=0.5))

            axes.text(0.1, 0.9, s1 + s2 + s3, fontsize=8, transform=plt.gcf().transFigure)  # header
            axes = plt.subplot2grid((50, 50), (5, 0), rowspan=7, colspan=39)  # axes for the next plot

        # DELAY LENGTH PLOT
        elif stage == 4 and task_type == 'classic':
            axes = plt.subplot2grid((50, 50), (0, 0), rowspan=4, colspan=39)
            # DELAY LENGTH PLOT
            subset = df.loc[df['trial_type'] == 'WM_Dl']
            sns.lineplot(x=subset.trial, y=subset.dl, marker= 'o', markersize=5, ax=axes, color=wmdl_c)
            lines = [1.4, 1.2, 1, 0.8, 0.6, 0.4, 0.2, 0]
            axes.hlines(y=lines, xmin=min(df.trial), xmax=max(df.trial), color=lines_c, linestyle=':', linewidth=1)

            y_min = subset.dl.min()
            y_max = subset.dl.max()

            axes.set_xlim(1, total_trials + 1)
            axes.set_ylim(y_min - 0.2, y_max + 0.2)
            axes.set_ylabel('Delay len \n (sec)')
            axes.set_xlabel('')
            axes.xaxis.set_ticklabels([])

            label = 'Max: ' + str(round(y_max, 3)) + ' s\n' + \
                    'Min: ' + str(round(y_min, 3)) + ' s'
            axes.text(0.9, 1.3, label, transform=axes.transAxes, fontsize=8, verticalalignment='top',
                      bbox=dict(facecolor='white', edgecolor=lines2_c, alpha=0.5))

            axes.text(0.1, 0.9, s1 + s2 + s3, fontsize=8, transform=plt.gcf().transFigure)  # header
            axes = plt.subplot2grid((50, 50), (5, 0), rowspan=7, colspan=39)  # axes for the next plot



        else:
            axes = plt.subplot2grid((50, 50), (0, 0), rowspan=12, colspan=39)
            axes.text(0.1, 0.9, s1 + s2 + s3, fontsize=8, transform=plt.gcf().transFigure) # header


        #### PLOT 1: ACCURACY VS TRIAL INDEX
        # we use the previous defined axes

        ttype_palette = sns.set_palette(ttypes_c, n_colors=len(ttypes_c)) # set the palette
        labels = list(ttypes)
        colors = list(ttypes_c)

        ### trial type accuracies######
        for ttype, ttype_df in first_resp_df.groupby('trial_type'):
            ttype_color = ttype_df.ttype_colors.iloc[0]
            ttype_df['acc'] = utils.compute_window(ttype_df.correct_bool, 20)
            sns.lineplot(x=ttype_df.trial, y=ttype_df.acc, ax=axes, color=ttype_color, marker='o', markersize=5)

        if repoking_bool == True: # separate in first and last poke
            for ttype, ttype_df in last_resp_df.groupby('trial_type'):
                ttype_color = ttype_df.ttype_colors.iloc[0]
                ttype_df['acc'] = utils.compute_window(ttype_df.correct_bool, 20)
                sns.lineplot(x=ttype_df.trial, y=ttype_df.acc, ax=axes, color=ttype_color, marker='o', markersize=5) #style=ttype_df.subject, dashes=[(2, 2), (2, 2)]
            linestyle = len(colors) * ['-']
            labels.extend(['First poke', 'Last poke'])
            colors.extend(['black', 'black'])
            linestyle.extend(['-', 'dotted'])
            label = 'First acc: ' + str(total_acc_first_poke) + '%' + '\n' + 'Last acc: ' + str(
                total_acc_last_poke) + '%'
            axes.text(0.9, 1.1, label, transform=axes.transAxes, fontsize=8, verticalalignment='top',
                      bbox=dict(facecolor='white', edgecolor=lines2_c, alpha=0.5))

        else: # global accuracy
            first_resp_df['acc'] = utils.compute_window(first_resp_df.correct_bool, 20)
            sns.lineplot(x=first_resp_df.trial, y=first_resp_df.acc, ax=axes, color='black', marker='o', markersize=5)
            labels.append('All')
            colors.append('black')
            linestyle = len(colors) * ['-']

        # axis
        axes.hlines(y=chance_lines, xmin=0, xmax=total_trials, color=lines_c, linestyle=':', linewidth=1)
        axes.fill_between(df.trial, chance_lines[0], 0, facecolor=lines2_c, alpha=0.3)
        axes.set_xlabel('Trial', label_kwargs)
        axes.set_xlim([1, total_trials + 1])
        utils.axes_pcent(axes, label_kwargs)
        try:
            axes.get_legend().remove()
        except:
            pass


        #### PLOT 2: ACCURACY VS TRIAL TYPE
        axes = plt.subplot2grid((50, 50), (0, 40), rowspan=12, colspan=10)
        x_min = -0.5
        x_max = len(ttypes) -0.5

        sns.pointplot(x=first_resp_df.trial_type, y=first_resp_df.correct_bool, order=ttypes,
                      ax=axes, s= 100, ci=68, color='black')
        if repoking_bool == True:  # add last poke
            sns.pointplot(x=last_resp_df.trial_type, y=last_resp_df.correct_bool, ax=axes, ci=68, color='black',
                          linestyles=["--"])

        # axis
        axes.hlines(y=chance_lines, xmin=x_min, xmax=x_max, color=lines_c, linestyle=':', linewidth=1)
        axes.fill_between(np.linspace(x_min, x_max, 2), chance_p, 0, facecolor=lines2_c, alpha=0.3)
        axes.set_xlabel('Trial type', label_kwargs)
        utils.axes_pcent(axes, label_kwargs)
        axes.set_ylabel('')
        axes.yaxis.set_ticklabels([])

        # legend
        lines = [Line2D([0], [0], color=colors[i], marker='o', markersize=7, markerfacecolor=colors[i], linestyle=linestyle[i]) for i in
                 range(len(colors))]
        axes.legend(lines, labels, title='Trial type', fontsize=8, loc='center', bbox_to_anchor=(1, 1.1))


        #### PLOT 3: ACCURACY VS STIMULUS POSITION
        axes = plt.subplot2grid((50, 50), (15, 0), rowspan=11, colspan=15)
        x_min = 0
        x_max = 400 #screen size

        sns.lineplot(x='x', y="correct_bool", data=first_resp_df, hue='trial_type', marker='o', markersize=8,
                     err_style="bars", ci=68, ax=axes)

        axes.hlines(y=chance_lines, xmin=x_min, xmax=x_max, color=lines_c, linestyle=':', linewidth=1)
        axes.fill_between(np.arange(x_min, x_max, 1), chance_p, 0, facecolor=lines2_c, alpha=0.3)
        if len(first_resp_df.trial_type.unique()) > 1:
            axes.get_legend().remove()

        # axis
        axes.set_xlabel('')
        axes.xaxis.set_ticklabels([])
        utils.axes_pcent(axes, label_kwargs)


        ### PLOT 4: ERRORS VS STIMULUS POSITION
        axes = plt.subplot2grid((50, 50), (26, 0), rowspan=11, colspan=15)

        sns.lineplot(x='x', y="error_x", data=first_resp_df, hue='trial_type', marker='o', markersize=8,
                     err_style="bars", ci=68, ax=axes)

        axes.hlines(y=0, xmin=x_min, xmax=x_max, color=lines_c, linestyle=':', linewidth=1)
        axes.fill_between(np.arange(x_min, x_max, 1), correct_th / 2, -correct_th / 2, facecolor='yellow', alpha=0.25)
        axes.set_xlabel('$Stimulus \ position\ (x_{t})\ (mm)%$', label_kwargs)
        axes.set_ylabel('Error (mm)', label_kwargs)
        if len(first_resp_df.trial_type.unique()) > 1:
            axes.get_legend().remove()


        ### PLOT 5: RESPONSE COUNTS
        axes = plt.subplot2grid((50, 50), (39, 0), rowspan=11, colspan=15)

        hist, bins = np.histogram(first_resp_df.x, bins=bins_resp)

        #sns.lineplot(x=x_positions, y=hist, marker='o', markersize=5, err_style="bars", color=lines2_c, ax=axes)
        for ttype, ttype_df in first_resp_df.groupby('trial_type'):
            ttype_color = ttype_df.ttype_colors.iloc[0]
            hist, bins = np.histogram(ttype_df.response_x, bins=bins_resp)
            a = sns.lineplot(x=x_positions, y=hist, marker='o', markersize=8, err_style="bars", color=ttype_color)

        axes.set_xlim(x_min, x_max)
        axes.set_xlabel('$Responses\ (r_{t})\ (mm)$', label_kwargs)
        axes.set_ylabel('Counts', label_kwargs)


        ### PLOT 6: RESPONSE COUNTS SORTED BY STIMULUS POSITION
        side_colors = ['lightseagreen', 'bisque', 'orange']
        axes_loc = [17, 28, 39]
        colspan = 10
        if mask == 5:
            side_colors.insert(0, 'darkcyan')
            side_colors.append('darkorange')
            axes_loc = [16, 23, 30, 37, 44]
            colspan = 7

        side_palette = sns.set_palette(side_colors, n_colors=len(side_colors))  # palette creation
        first_resp_df['rt_bins'] = pd.cut(first_resp_df.response_x, bins=bins_resp, labels=x_positions,
                                          include_lowest=True)
        for idx in range(len(x_positions)):
            axes = plt.subplot2grid((50, 50), (16, axes_loc[idx]), rowspan=12, colspan=colspan)
            subset = first_resp_df.loc[first_resp_df['x'] == x_positions[idx]]
            axes.set_title('$x_{t}\ :%$' + str(x_positions[idx]), fontsize=13, fontweight='bold')
            sns.countplot(subset.rt_bins, ax=axes, palette=side_colors)
            axes.set_xlabel('')
            if idx != 0:
                axes.set_ylabel('')
                axes.yaxis.set_ticklabels([])
                if idx == int((len(x_positions) / 2)):
                    axes.set_xlabel('$Responses\ (r_{t})\ (mm)$')

        # legend
        labels = list(x_positions)
        lines = [Patch(facecolor=c, edgecolor=c) for c in side_colors]
        axes.legend(lines, labels, fontsize=8, title= 'Rt', loc='center', bbox_to_anchor=(1.1, 1))

        ### PLOT 7: RESPONSE LATENCIES   # we look to all the reponses time
        axes = plt.subplot2grid((50, 50), (31, 18), rowspan=25, colspan=15)

        if stage == 1:
            y_max = 30
        else:
            y_max = 10
        sns.boxplot(x='x', y='resp_latency', hue='rt_bins', data=first_resp_df, color='white', linewidth=0.5,
                        showfliers=False, ax=axes)
        resp_df['rt_bins'] = pd.cut(resp_df.response_x, bins=bins_resp, labels=x_positions,
                                          include_lowest=True)
        sns.stripplot(x='x', y='resp_latency', hue='rt_bins', data=resp_df, dodge=True, ax=axes)
        axes.set_ylabel("Response latency (sec)", label_kwargs)
        axes.set_xlabel('$Stimulus \ position\ (x_{t})\ (mm)%$', label_kwargs)
        axes.set_ylim(0, y_max)
        axes.get_legend().remove()

        ### PLOT 8: LICKPORT LATENCIES   # we look only the trial time
        axes = plt.subplot2grid((50, 50), (31, 38), rowspan=25, colspan=13)

        y_max = 20
        sns.boxplot(x='trial_result', y='lick_latency', data=df, color='white', linewidth=0.5, showfliers=False,
                    ax=axes, order=treslts)
        sns.stripplot(x="trial_result", y="lick_latency", color=water_c, data=df, dodge=True,
                      ax=axes, order=treslts)

        axes.set_xticklabels(treslts, fontsize=9, rotation=35)
        axes.set_ylabel("Lickport latency (sec)", label_kwargs)
        axes.set_xlabel("")
        axes.set_ylim(0, y_max)
        sns.despine()


        # SAVING AND CLOSING PAGE
        sns.despine()
        pdf.savefig()
        plt.close()

        ############## PAGE 2 ##############

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
        try:
            h_bars['origin'] = h_bars['fixation_time'] + h_bars['delay']
        except:
            h_bars['origin'] = h_bars['fixation_time']
        origin = h_bars.origin.tolist()
        origin = [-i for i in origin]

        # misses
        miss_df = df.loc[df['trial_result'] == 'miss', :]
        origin_m = miss_df.fixation_time.tolist()
        origin_m = [-i for i in origin_m]

        # RASTER PLOT
        x_min = -2
        if stage == 1:
            x_max = 30
        else:
            x_max = 10

        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=42, colspan=25)
        treslt_palette = sns.set_palette(rreslts_c, n_colors=len(rreslts_c))

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
        axes = plt.subplot2grid((50, 50), (0, 26), rowspan=42, colspan=25)
        if mask != 0:
            x_min = -400
            x_max = 400

        else:
            x_min = -200
            x_max = 200
        sns.scatterplot(x=resp_df.error_x, y=resp_df.trial, hue=resp_df.response_result, style=resp_df.trial_type,
                        s=20, ax=axes, zorder=20)
        axes.barh(list(df.trial), width=x_max*4, color=lines2_c, left=x_min*2, height=0.7, alpha=0.4, zorder=0)

        #vertical lines
        axes.axvspan(-stim_width, stim_width, color=stim_c, alpha=0.2)
        for idx, line in enumerate(threshold_lines):
            axes.axvline(x=line, color=threshold_lines_c[idx], linestyle=':', linewidth=1)
            axes.axvline(x=-line, color=threshold_lines_c[idx], linestyle=':', linewidth=1)

        axes.set_xlabel('')
        axes.set_ylabel('')
        axes.yaxis.set_ticklabels([])
        axes.xaxis.set_ticklabels([])
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(-1, total_trials + 1)
        axes.legend(loc='center', bbox_to_anchor=(1, 1)).set_zorder(10)
        sns.despine()

        # ERRORS HISTOGRAM
        axes = plt.subplot2grid((50, 50), (43, 26), rowspan=7, colspan=25)
        bins = np.linspace(-r_edge, r_edge, mask * 2 + 6)
        sns.distplot(first_resp_df.error_x, kde=False, bins=bins, color=lines_c, ax=axes,
                     hist_kws={'alpha': 0.9, 'histtype': 'step', 'linewidth': 2})

        # vertical lines
        axes.axvspan(-stim_width, stim_width, color=stim_c, alpha=0.2)
        for idx, line in enumerate(threshold_lines):
            axes.axvline(x=line, color=threshold_lines_c[idx], linestyle=':', linewidth=1)
            axes.axvline(x=-line, color=threshold_lines_c[idx], linestyle=':', linewidth=1)

        axes.set_xlabel('$Errors\ (r_{t}-x_{t})\ (mm)%$', label_kwargs)
        axes.set_ylabel('')
        axes.yaxis.set_ticklabels([])
        axes.set_xlim(x_min, x_max)
        sns.despine()


        # SAVING AND CLOSING PAGE
        sns.despine()
        pdf.savefig()
        plt.close()

        print('New daily report completed successfully')