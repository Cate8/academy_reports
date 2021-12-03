import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from academy_reports import utils
from datetime import timedelta, datetime

# PLOT COLORS
correct_first_c = 'green'
correct_other_c = 'limegreen'
miss_c = 'black'
incorrect_c = 'red'
punish_c = 'firebrick'
lines_c = 'gray'
lines2_c = 'silver'
vg_c = '#393b79'
wmi_c = '#6b6ecf'
wmds_c = '#9c9ede'
wmdm_c = '#ce6dbd'
wmdl_c = '#a55194'
correct_th_c = 'green'
repoke_th_c = 'orangered'
label_kwargs = {'fontsize': 9}


def intersession(df, save_path_intersesion):
    df = df[df['task'].str.contains("StageTraining")]

    if df.shape[0] > 1:
        print('Doing Intersession...')

        #####################  PARSE #####################

        ##### SELECT LAST MONTH SESSIONS #####
        df['day'] = pd.to_datetime(df['date']).dt.date
        df= df.loc[df['day'] > df.day.max() - timedelta(days=20)]

        #######  RELEVANT COLUMNS  ######
        # add columns (when absent)
        column_list = ['STATE_Correct_first_START', 'STATE_Miss_START', 'STATE_Punish_START',
                       'STATE_Correct_other_START',
                       'STATE_Incorrect_START', 'STATE_Incorrect_END', 'STATE_Response_window2_START',
                       'STATE_Response_window2_END',
                       'STATE_Correct_first_reward_START', 'STATE_Correct_other_reward_START',
                       'STATE_Miss_reward_START']
        for col in column_list:
            if col not in df.columns:
                df[col] = np.nan
       

        ###### CONVERT STRINGS TO LISTS ######
        conversion_list = ['response_x', 'STATE_Response_window2_START','STATE_Response_window2_END', 'STATE_Incorrect_START']

        for idx, column in enumerate(conversion_list):
            try:
                df[column].str.contains(',')  # means that contains multiple values
            except:  # remove from conversion list
                conversion_list.remove(column)

        df = utils.convert_strings_to_lists(df, conversion_list)
        

        ###### RELEVANT COLUMNS ######

        # get first & last items
        df['response_first'] = df['response_x'].str[0]
        df['response_last'] = df['response_x'].str[-1]
        df['incorrect_first']= df['STATE_Incorrect_START'].str[0]
        df['rw_end_last'] = df['STATE_Response_window2_END'].str[-1]

        # weights
        weight = df.subject_weight.iloc[-1]
        df['relative_weights'] = df.apply(lambda x: utils.relative_weights(x['subject'], x['subject_weight']), axis=1)

        # task version
        df['version']='classic'
        df.loc[((df['task'].str.contains("4B")) | (df['task'].str.contains("5B")) | (df['task'].str.contains("6B"))), 'version'] = 't-maze'
        version= df.version.unique()

        # chance & mask
        mask= df['mask'].unique()
        mask=mask[0]
        df['chance'] = np.where(df['mask'] == 3, 1/3, 1/5)
        chance= df.chance.unique()

        # categorize stimulus positions and respsonses
        if mask == 3:
            bins = pd.IntervalIndex.from_tuples([(25, 105), (160, 240), (295, 375)])
            labels = [-1, 0, 1]
        elif mask == 5:
            bins = pd.IntervalIndex.from_tuples([(0, 82), (82, 163), (163, 242), (242, 322), (322, 401)])
            labels = [-2, -1, 0, 1, 2]
        df['r_c'] = pd.cut(df['response_first'], bins).map(dict(zip(bins, labels)))
        df['x_c'] = pd.cut(df['x'], bins).map(dict(zip(bins, labels)))

        ## previous stim, resp and outcome
        df['prev_x_c'] = df['x_c'].shift(1)
        df['prev_r_c'] = df['r_c'].shift(1)
        df['prev_result'] = df['trial_result'].shift(1)
        df['rep_bool'] = df.apply(lambda x: 1 if x.loc['r_c'] == x.loc['prev_r_c'] else 0, axis=1)

        # latencies & times
        df['rw_end'] = df['rw_end_last'].fillna(df['STATE_Response_window_END'])

        df['responses_time'] = df.apply(lambda row: utils.create_responses_time(row), axis=1)
        df['responses_time_first'] = df['responses_time'].str[0]
        df['responses_time_last'] = df['responses_time'].str[-1]

        df['reward_time'] = df['STATE_Correct_first_reward_START'].fillna(0) + df[
            'STATE_Correct_other_reward_START'].fillna(0) + df['STATE_Miss_reward_START'].fillna(0)

        df['resp_latency'] = df['rw_end'] - df['STATE_Response_window_START']
        df['lick_latency'] = df['reward_time'] - df['rw_end']


        # separate delays (when required)
        try:
            if (df['pwm_ds'] > 0).any():
                df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DS')), 'trial_type'] = 'WM_Ds'
                df.loc[(df.trial_type == 'WM_Ds', 'ttype_colors')] = wmds_c
        except:
            pass
        try:
            if (df['pwm_dm'] > 0).any():
                df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DM')), 'trial_type'] = 'WM_Dm'
                df.loc[(df.trial_type == 'WM_Dm', 'ttype_colors')] = wmdm_c
        except:
            pass
        try:
            if (df['pwm_dl'] > 0).any():
                df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DL')), 'trial_type'] = 'WM_Dl'
                df.loc[(df.trial_type == 'WM_Dl', 'ttype_colors')] = wmdl_c
        except:
            pass

        # Abs delay
        df['delay_abs'] = 0  # FOR VG
        if version== 't-maze':
            df.loc[df['trial_type'] == 'WM_I', 'delay_abs'] = df['rw_end'] - df['STATE_Fixation3_END'] - df[
                'rw_stim_dur']  # FOR WMI
            df.loc[df['trial_type'] == 'WM_Ds', 'delay_abs'] = df['rw_end'] - df['STATE_Fixation2_END'] - df[
                'wm_stim_dur']  # FOR WMDS
            df.loc[df['trial_type'] == 'WM_Dl', 'delay_abs'] = df['rw_end'] - df['STATE_Fixation1_END']  # FOR WMDL
        elif version == 'classic':
            df.loc[df['trial_type'] == 'WM_I', 'delay_abs'] = df['rw_end'] - df['STATE_Response_window_START'] - df[
                'wm_stim_dur']  # FOR WMI
            df.loc[((df['trial_type'] != 'VG') & (df['trial_type'] != 'WM_I')), 'delay_abs'] = \
                df['rw_end'] - df['STATE_Response_window_START'] + df['delay'] # FOR ALL DELAYS


        # error
        df['error_first'] = df['response_first'] - df['x']
        df['error_last'] = df['response_last'] - df['x']

        # correct bool
        df['correct_bool'] = np.where(df['trial_result'] == 'correct_first', 1, 0)
        df.loc[(df.trial_result == 'miss', 'correct_bool')] = np.nan
        df['correct_bool_last'] = df['correct_bool']
        df.loc[df['trial_result'] == 'correct_other', 'correct_bool_last'] = 1

        #day of  a date
        df['date_day']= df['date'].apply(lambda x: x[-5:])


        #####################  PLOTS #####################

        ############## PAGE 1 ##############
        with PdfPages(save_path_intersesion) as pdf:
            plt.figure(figsize=(11.7, 15))

            ### PLOT 1: NUMBER OF TRIALS
            axes = plt.subplot2grid((50, 50), (0, 0), rowspan=5, colspan=23)
            grouped_df = df.groupby('session').agg({'trial': max, 'day': max, 'stage': max}).reset_index()

            sns.lineplot(x='day', y='trial', style='stage', markers=True, ax=axes, color='black',
                         data=grouped_df, estimator=sum, ci=None)
            axes.axhline(y=100,color=lines_c, linestyle=':', linewidth=1)
            axes.set_ylabel('Nº of trials', label_kwargs)
            axes.set_xlabel('')
            axes.xaxis.set_ticklabels([])
            axes.legend(title='Stage', fontsize=8, loc='center', bbox_to_anchor=(0.15, 1.2))


            ### PLOT 2: RELATIVE WEIGHTS
            axes = plt.subplot2grid((50, 50), (0, 27), rowspan=5, colspan=24)
            sns.lineplot(x='day', y='relative_weights', style='stage', markers=True, ax=axes, color='black', data=df)

            axes.axhline(y=100, color=lines_c, linestyle=':', linewidth=1)
            axes.set_ylabel('Rel. weight (%)', label_kwargs)
            axes.set_ylim(70, 150)
            axes.set_xlabel('')
            axes.xaxis.set_ticklabels([])

            # axes.get_xaxis().set_ticks([])
            axes.get_legend().remove()

            label = 'Last: ' + str(weight) + ' g'
            axes.text(1, 0.95, label, transform=axes.transAxes, fontsize=8,  fontweight='bold', verticalalignment='top')


            ### PLOT 3:  RESPONSE LATENCIES
            axes = plt.subplot2grid((50, 50), (6, 0), rowspan=5, colspan=24)

            treslt_palette = [correct_first_c, correct_other_c, punish_c]
            treslt_order= ['correct_first', 'correct_other', 'punish']

            subset= df.loc[((df['trial_result']== 'correct_first') | (df['trial_result']== 'correct_other') | (df['trial_result']== 'punish'))]
            ymax= subset.resp_latency.median()+5
            sns.lineplot(x='day', y='resp_latency', hue='trial_result', hue_order=treslt_order, style='stage', data=subset,
                         markers=True, ax=axes, estimator=np.median, palette = treslt_palette)

            axes.set_ylabel('Response latency(s)', label_kwargs)
            axes.set_xlabel('')
            axes.set_ylim(0, ymax)
            axes.get_legend().remove()
            xticks = axes.get_xticks()
            xtime = df.date_day.unique()
            div = int(len(xtime) / len(xticks))
            if div == 0:
                div= 1
            xtime2 = []
            for i in range(0, len(xtime), div):
                xtime2.append(xtime[i])
            axes.set_xticklabels(xtime2, rotation=40)

            ### PLOT 4:  LICK LATENCIES
            axes = plt.subplot2grid((50, 50), (6, 27), rowspan=5, colspan=24)
            ymax = subset.lick_latency.median()+10
            sns.lineplot(x='day', y='lick_latency', hue='trial_result', hue_order=treslt_order, style='stage', data=subset,
                         markers=True, ax=axes, estimator=np.median, palette = treslt_palette, ci=68)

            axes.set_ylabel('Lick latency(s)', label_kwargs)
            axes.set_ylim(0, ymax)
            axes.set_xlabel('')
            axes.set_xticklabels(xtime2, rotation=40)
            # legend
            axes.get_legend().remove()
            lines = [Line2D([0], [0], color=treslt_palette[i], marker='o', markersize=7, markerfacecolor=treslt_palette[i])
                     for i in range(len(treslt_palette))]
            axes.legend(lines, treslt_order, title='Trial result', fontsize=8, loc='center', bbox_to_anchor=(1.1, 0.7))


            ### PLOT 5: TTYPE ACCURACIES
            axes = plt.subplot2grid((50, 50), (14, 0), rowspan=8, colspan=50)
            ttype_palette = [vg_c, wmi_c, wmds_c, wmdm_c, wmdl_c]
            ttype_order= ['VG', 'WM_I', 'WM_Ds', 'WM_Dm', 'WM_Dl']

            index_list=[]
            for ttype in ttype_order:
                if ttype not in df.trial_type.unique():
                    index_list.append(ttype_order.index(ttype))
                    
            for i in sorted(index_list, reverse=True):
                del ttype_order[i]
                del ttype_palette[i]

            sns.lineplot(x='day', y='correct_bool', hue='trial_type', hue_order=ttype_order, palette=ttype_palette, 
                    style='stage', markers=True, ax=axes, data=df, ci=68)

            axes.axhline(y=chance, color=lines_c, linestyle=':', linewidth=1)
            axes.fill_between(df.day, df.chance, 0, facecolor=lines2_c, alpha=0.3)
            utils.axes_pcent(axes, label_kwargs)

            axes.get_legend().remove()
            axes.set_xlabel('')
            axes.set_xticklabels(xtime)

            # legend
            lines = [Line2D([0], [0], color=ttype_palette[i], marker='o', markersize=7, markerfacecolor=ttype_palette[i])
                for i in range(len(ttype_palette))]
            axes.legend(lines, ttype_order , fontsize=8, title='Trial type', loc='center', bbox_to_anchor=(1.05, 0.75))

            axes = plt.subplot2grid((50, 50), (22, 0), rowspan=5, colspan=50)


            ## PLOT 6:  RESPONSE COUNTS SORTED BY STIMULUS POSITION
            side_colors = ['lightseagreen', 'bisque', 'orange']
            axes_loc = [22, 27, 32]
            rowspan = 5
            if mask == 5:
                side_colors.insert(0, 'darkcyan')
                side_colors.append('darkorange')
                axes_loc = [22, 25, 28, 31, 34]
                rowspan = 3
            side_palette = sns.set_palette(side_colors, n_colors=len(side_colors))  # palette creation

            for idx, x in enumerate(labels):
                axes = plt.subplot2grid((50, 50), (axes_loc[idx], 0), rowspan=rowspan, colspan=50)
                subset = df.loc[df['x_c'] == x]

                sns.countplot(subset.day, hue =subset.r_c, ax=axes, palette=side_colors)
                axes.set_xlabel('Time (Month/Day)')
                axes.set_ylabel('$x_{t}\ :%$' + str(x))
                #legend
                if idx != len(labels)-1:
                    axes.xaxis.set_ticklabels([])
                    axes.set_xlabel('')
                    axes.get_legend().remove()
                else:
                    axes.legend(loc='center', bbox_to_anchor=(1.05, 1.5), title= '$r_{t}\ :%$')
                    axes.set_xticklabels(xtime)

#
            ## PLOT 7: ACC TRIAL TYPE
            axes = plt.subplot2grid((50, 50), (39, 0), rowspan=11, colspan=11)
            x_min = -0.5
            x_max = len(ttype_order) - 0.5

            sns.pointplot(x='trial_type', y='correct_bool_last', order=ttype_order, ax=axes, s=100, ci=68,
                          color='black', linestyles=["--"], data=df)
            sns.pointplot(x='trial_type', y='correct_bool', order=ttype_order, ax=axes, s=100, ci=68,
                          color='black', data=df)

            axes.hlines(y=[chance], xmin=x_min, xmax=x_max, color=lines_c, linestyle=':', linewidth=1)
            axes.fill_between(np.linspace(x_min, x_max, 2), chance, 0, facecolor=lines2_c, alpha=0.3)
            axes.set_xlabel('Trial type', label_kwargs)
            axes.set_xticklabels(ttype_order, rotation=40)

            utils.axes_pcent(axes, label_kwargs)
            axes.set_ylabel('Accuracy (%)', label_kwargs)

            #### PLOT 8: ACCURACY VS ABS DELAY
            axes = plt.subplot2grid((50, 50), (39, 11), rowspan=11, colspan=15)
            x_min = 0
            x_max = 4
            delay_bins = np.arange(x_min, x_max, 0.3)
            bins_plt = delay_bins[:-1] + (delay_bins[1] - delay_bins[0]) / 2

            df['delay_abs_bins'] = pd.cut(df.delay_abs, bins=delay_bins, labels=bins_plt, include_lowest=True)
            sns.lineplot(x='delay_abs_bins', y='correct_bool', hue='trial_type', hue_order=ttype_order, data=df, marker='o',
                         markersize=8, ax=axes, palette=ttype_palette, zorder=10, err_style='bars')

            axes.hlines(y=[chance], xmin=x_min, xmax=x_max, color=lines_c, linestyle=':', linewidth=1)
            axes.fill_between(np.linspace(x_min, x_max, 2), chance, 0, facecolor=lines2_c, alpha=0.3)
            axes.set_xlabel('Delay absolute (sec)', label_kwargs)
            utils.axes_pcent(axes, label_kwargs)
            axes.yaxis.set_ticklabels([])
            axes.set_ylabel('')
            axes.get_legend().remove()

            #### PLOT 9: ACCURACY VS STIMULUS POSITION
            axes = plt.subplot2grid((50, 50), (39, 26), rowspan=11, colspan=11)
            x_min = df.x_c.min()
            x_max = df.x_c.max()

            sns.lineplot(x='x_c', y="correct_bool", data=df, hue='trial_type', marker='o',
                         markersize=8, err_style="bars", ci=68, ax=axes, palette=ttype_palette)

            axes.hlines(y=[chance], xmin=x_min, xmax=x_max, color=lines_c, linestyle=':', linewidth=1)
            axes.fill_between(np.linspace(x_min, x_max, 2), chance, 0, facecolor=lines2_c, alpha=0.3)

            # axis
            axes.set_xlabel('Stimulus position')
            utils.axes_pcent(axes, label_kwargs)
            axes.yaxis.set_ticklabels([])
            axes.set_ylabel('')
            axes.get_legend().remove()


#           #### PLOT 10: REPEATING BIAS
            axes = plt.subplot2grid((50, 50), (39, 41), rowspan=11, colspan=9)


            # Boxplot
            stats_= utils.stats_function(df, ['day', 'prev_result'])
            treslt_labels=['C_f', 'C_o', 'P']
            if df.stage.max != 1:
                treslt_palette = [correct_first_c, punish_c]
                treslt_order = ['correct_first', 'punish']
                treslt_labels = ['Correct', 'Punish']
            df.stage.max()
            sns.stripplot(x='prev_result', y='CRB', order=treslt_order, data=stats_, ax=axes, size=8,
                          palette=treslt_palette,  split=True, alpha=0.7)

            ax=sns.boxplot(x='prev_result', y='CRB', order=treslt_order, ax=axes, data=stats_, palette=treslt_palette,
                                         boxprops=dict(alpha=.4), fliersize=0)
            plt.setp(ax.lines, color=".4")
            axes.axhline(y=0, color=lines_c, linestyle=':', linewidth=2)
            axes.set_ylabel('Repeating Bias',label_kwargs)
            axes.set_xlabel('Prev. outcome')

            # SAVING AND CLOSING PAGE
            sns.despine()
            pdf.savefig()
            plt.close()
            print('Interseesion completed succesfully!')

#
#     else:
#         print('Wait until Stgae Training for the intersession...')
#
#
#
