# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.lines import Line2D
# from matplotlib.backends.backend_pdf import PdfPages
# from academy_reports import utils
#
#
# # PLOT COLORS
# correct_first_c = 'green'
# correct_other_c = 'limegreen'
# miss_c = 'black'
# incorrect_c = 'red'
# punish_c = 'firebrick'
#
# water_c = 'teal'
# lines_c = 'gray'
# lines2_c = 'silver'
#
# vg_c = '#393b79'
# wmi_c = '#6b6ecf'
# wmds_c = '#9c9ede'
# wmdm_c = '#ce6dbd'
# wmdl_c = '#a55194'
#
# stim_c = 'gold'
# correct_th_c = 'green'
# repoke_th_c = 'orangered'
#
# label_kwargs = {'fontsize': 9}
#
# # BINNING
# """
# Touchscreen active area: 1440*900 pixels --> 403.2*252 mm
# Stimulus radius: 35pix (9.8mm)
# x_positions: 35-1405 pix --> 9.8-393.4mm
# """
# l_edge = 9.8
# r_edge = 393.4
# bins_resp = np.linspace(l_edge, r_edge, 11)
# bins_err = np.linspace(-r_edge, r_edge, 24)
#
#
#
def intersession(df, save_path_intersesion):
    print('Doing Intersession...')
#
#     ###fix incorrects
#     df.loc[((df.trial_result == 'miss') & (df['response_x'].notna()), 'trial_result')] = 'incorrect'
#
#     # CONVERT STRINGS TO LISTS
#     df = utils.convert_strings_to_lists(df, ['response_x', 'response_y', 'STATE_Incorrect_START', 'STATE_Incorrect_END',
#                                        'STATE_Fixation_START', 'STATE_Fixation_END', 'STATE_Fixation_break_START',
#                                        'STATE_Fixation_break_END', 'STATE_Response_window2_START',
#                                        'STATE_Response_window2_END'])
#
#     # RELEVANT VARIABLES
#     total_sessions = df.session.max()
#     tasks_list = df.task.unique().tolist()
#     weight = df.subject_weight.iloc[-1]
#
#     # RELEVANT COLUMNS
#     df['all_task'] = df['task'].copy()
#     df.loc[((df.task != 'LickTeaching') & (df.task != 'TouchTeaching'), 'task')] = 'StageTraining'
#
#     ### latencies
#     df['resp_latency'] = np.nan
#     df['lick_latency'] = np.nan
#     df.loc[
#         df['task'] == 'LickTeaching', ['lick_latency']] = df.STATE_Wait_for_reward_END - df.STATE_Wait_for_reward_START
#     df.loc[df['task'] == 'TouchTeaching', ['lick_latency']] = (df.STATE_Correct_first_reward_START.fillna(
#         0) + df.STATE_Miss_reward_START.fillna(0)) - df.STATE_Response_window_END
#     df.loc[
#         df['task'] == 'TouchTeaching', ['resp_latency']] = df.STATE_Response_window_END - df.STATE_Response_window_START
#
#     df['STATE_Response_window2_END'] = df['STATE_Response_window2_END'].apply(lambda x: [np.nan] if len(x) == 0 else x)
#     df['response_window_end'] = df['STATE_Response_window2_END'].apply(lambda x: x[-1])
#     df['response_window_end'] = df['response_window_end'].fillna(df['STATE_Response_window_END'])
#     df['reward_time'] = df['STATE_Correct_first_reward_START'].fillna(0) + df[
#         'STATE_Correct_other_reward_START'].fillna(0) + df['STATE_Miss_reward_START'].fillna(0)
#     df.loc[df['task'].str.contains("StageTraining"), ['lick_latency']] = df['reward_time'] - df['response_window_end']
#     df.loc[df['task'].str.contains("StageTraining"), ['resp_latency']] = df['response_window_end'] - df[
#         'STATE_Response_window_START']
#
#     ### relative weights
#     df['relative_weights'] = df.apply(lambda x: utils.relative_weights(x['subject'], x['subject_weight']), axis=1)
#
#
#     with PdfPages(save_path_intersesion) as pdf:
#         plt.figure(figsize=(11.7, 11.7))
#
#         # NUMBER OF TRIALS
#         axes = plt.subplot2grid((50, 50), (0, 0), rowspan=7, colspan=23)
#
#         task_palette = sns.set_palette(['black'], n_colors=len(tasks_list))
#         total_trials_s = df.groupby(['session'], sort=False)['trial', 'task'].max()
#         sns.lineplot(x=total_trials_s.index, y=total_trials_s.trial, hue=total_trials_s.task, style=total_trials_s.task,
#                      markers = True, ax=axes)
#
#         axes.hlines(y=[150], xmin=1, xmax=total_sessions, color=lines_c, linestyle=':')
#         axes.set_ylabel('Nº of trials', label_kwargs)
#         axes.set_xlabel('')
#         axes.get_xaxis().set_ticks([])
#         axes.legend(fontsize=8, loc='center', bbox_to_anchor=(0.15, 1.2))
#
#         # WEIGHT PLOT
#         axes = plt.subplot2grid((50, 50), (8, 0), rowspan=7, colspan=23)
#
#         sns.lineplot(x=df.session, y=df.relative_weights, palette=task_palette, hue=df.task, style=df.task,
#                      markers=True, ax=axes)
#         axes.hlines(y=[85], xmin=1, xmax=total_sessions, color=lines_c, linestyle=':')
#         axes.set_ylabel('Rel. \n Weight (%)')
#         axes.set_ylim(75, 95)
#         axes.set_xlabel('')
#         axes.get_xaxis().set_ticks([])
#         axes.get_legend().remove()
#
#         label = 'Last: ' + str(weight) + ' g'
#         axes.text(0.85, 0.95, label, transform=axes.transAxes, fontsize=8,  fontweight='bold', verticalalignment='top')
#
#         # LATENCIES
#         axes = plt.subplot2grid((50, 50), (16, 0), rowspan=7, colspan=23)
#         ymax= 10
#
#         # response latency
#         sns.lineplot(x=df.session, y=df.resp_latency, hue=df.task, style=df.task, markers=True, ax=axes,
#                      estimator=np.median)
#         # lick latency
#         task_palette = sns.set_palette([water_c], n_colors=len(tasks_list))
#         lick_latency_s = df.groupby(['session']).agg({'lick_latency': ['mean', 'median', 'std'], 'task': max})
#         a = sns.lineplot(x=df.session, y=df.lick_latency, hue=df.task, style=df.task, ax=axes, estimator=np.median)
#         sns.scatterplot(x=lick_latency_s.index, y=lick_latency_s.lick_latency.iloc[:, 1],
#                         hue=lick_latency_s.task.iloc[:, 0], style=lick_latency_s.task.iloc[:, 0], s=40, ax=axes)
#
#         axes.set_ylabel('Latency (sec)')
#         axes.set_ylim(0, ymax)
#         axes.set_xlim(0, total_sessions+0.5)
#         axes.set_xlabel('')
#         colors = [miss_c, water_c]
#         labels = ['Resp. lat.', 'Lick lat.']
#         lines = [Line2D([0], [0], color=colors[i], marker=',', markersize=6, ) for i in range(len(colors))]
#         axes.legend(lines, labels, fontsize=8, loc='center', bbox_to_anchor=(0.9, 0.8))
#
#
#         ############################################ STAGE TRAINING PLOTS #############################################
#
#         df = df[df['task'].str.contains("StageTraining")]
#         if df.shape[0] > 0:
#
#             # FIX DELAY TYPE IN THE FIRST SESSIONS
#             df['old'] = df.date.apply(lambda x: 'true' if x <= '2020/06/28' else 'false')
#             df.loc[((df.old == 'true') & (df.trial_type == 'WM_D') & (df.delay == 0)), 'delay_type'] = 'DS'
#             df.loc[((df.old == 'true') & (df.trial_type == 'WM_D') & (df.delay == 0.5)), 'delay_type'] = 'DM'
#             df.loc[((df.old == 'true') & (df.trial_type == 'WM_D') & (df.delay == 1)), 'delay_type'] = 'DL'
#
#
#             # FIX TTYPES
#             if (df['pwm_ds'] > 0).any():
#                 df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DS')), 'trial_type'] = 'WM_Ds'
#             if (df['pwm_dm'] > 0).any():
#                 df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DM')), 'trial_type'] = 'WM_Dm'
#             if (df['pwm_dl'] > 0).any():
#                 df.loc[((df.trial_type == 'WM_D') & (df.delay_type == 'DL')), 'trial_type'] = 'WM_Dl'
#
#             # FIX DELAYS
#             df.loc[(df.trial_type == 'VG'), 'delay'] = -1  # VG trials
#             df.loc[(df.trial_type == 'WM_I'), 'delay'] = -0.5  # WMI trials
#
#             ttypes = df.trial_type.unique().tolist()
#
#
#             # correct the order of the lists
#             if 'WM_Ds' in ttypes:
#                 idx = ttypes.index('WM_Ds')
#                 ttypes.pop(idx)
#                 ttypes.insert(1, "WM_Ds")
#                 if 'WM_I' in ttypes:
#                     idx = ttypes.index('WM_I')
#                     ttypes.pop(idx)
#                     ttypes.insert(1, "WM_I")
#
#             ###colors to plot columns
#             df['ttype_colors'] = vg_c
#             df.loc[(df.trial_type == 'WM_I', 'ttype_colors')] = wmi_c
#             df.loc[(df.trial_type == 'WM_Ds', 'ttype_colors')] = wmds_c
#             df.loc[(df.trial_type == 'WM_Dm', 'ttype_colors')] = wmdm_c
#             df.loc[(df.trial_type == 'WM_Dl', 'ttype_colors')] = wmdl_c
#             ttype_colors = df.ttype_colors.unique().tolist()
#
#             # THRESHOLDS & CHANCE
#             stim_width = df.width.mean() / 2
#             vg_correct_th = df.loc[df.trial_type == 'VG'].correct_th.mean() / 2
#             wm_correct_th = df.loc[df.trial_type != 'VG'].correct_th.mean() / 2
#             vg_repoke_th = df.loc[df.trial_type == 'VG'].repoke_th.mean() / 2
#             wm_repoke_th = df.loc[df.trial_type != 'VG'].repoke_th.mean() / 2
#
#             vg_chance_p = utils.chance_calculation(vg_correct_th)
#             wm_chance_p = utils.chance_calculation(wm_correct_th)
#             lines_list = [stim_width, vg_correct_th, vg_repoke_th]
#             lines_list_colors = [stim_c, correct_th_c, repoke_th_c]
#             x_min = df.session.min()
#
#             # CREATE REPONSES DF
#             ### needed columns before the unnest
#             df['responses_time'] = df.apply(lambda row: utils.create_responses_time(row), axis=1)
#             df['response_result'] = df.apply(lambda row: utils.create_reponse_result(row), axis=1)
#             ### unnest
#             resp_df = utils.unnesting(df, ['response_x', 'response_y', 'responses_time', 'response_result'])
#
#             # RELEVANT COLUMNS
#             ### responses latency (fix this)
#             resp_df['resp_latency'] = resp_df['responses_time'] - resp_df['STATE_Response_window_START']
#             ### errors
#             resp_df['error_x'] = resp_df['response_x'] - resp_df['x']
#             ### correct bool
#             resp_df['correct_bool'] = np.where(resp_df['correct_th'] / 2 >= resp_df['error_x'].abs(), 1, 0)
#             resp_df.loc[(resp_df.trial_result == 'miss', 'correct_bool')] = np.nan
#
#             # SUBDATAFRAMES
#             first_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='first', inplace=False).copy()
#             last_resp_df = resp_df.drop_duplicates(subset=['trial', 'session'], keep='last', inplace=False).copy()
#
#
#             # PLOTS
#
#             # ACCURACY TRIAL INDEX
#             axes = plt.subplot2grid((50, 50), (0, 27), rowspan=11, colspan=24)
#
#             ### first poke
#             task_palette = sns.set_palette([correct_first_c], n_colors=len(tasks_list))
#             sns.lineplot(x=first_resp_df.session, y=first_resp_df.correct_bool, hue=first_resp_df.subject,
#                          style=first_resp_df.subject, markers=True, ci=None, ax=axes)
#             ### last poke
#             task_palette = sns.set_palette([correct_other_c], n_colors=len(tasks_list))
#             sns.lineplot(x=last_resp_df.session, y=last_resp_df.correct_bool,  hue=last_resp_df.subject,
#                          style=last_resp_df.subject, markers=True, ci=None, ax=axes)
#
#             axes.hlines(y=[0.5, 1], xmin=df.session.min(), xmax=total_sessions, color=lines_c, linestyle=':')
#             axes.fill_between(df.session, vg_chance_p, 0, facecolor=lines_c, alpha=0.3)
#             axes.fill_between(df.session, wm_chance_p, 0, facecolor=lines2_c, alpha=0.3)
#             utils.axes_pcent(axes, label_kwargs)
#             axes.get_xaxis().set_ticks([])
#             axes.set_xlabel('')
#
#             #legend
#             colors = [correct_first_c, correct_other_c]
#             labels = ['First poke', 'Last poke']
#             lines = [Line2D([0], [0], color=colors[i], marker='o', markersize=6,
#                             markerfacecolor=colors[i]) for i in range(len(colors))]
#             axes.legend(lines, labels, fontsize=8, loc='center', bbox_to_anchor=(1, 0.9))
#
#
#             # STD TRIAL INDEX
#             axes = plt.subplot2grid((50, 50), (12, 27), rowspan=11, colspan=23)
#
#             ### first poke
#             task_palette = sns.set_palette([correct_first_c], n_colors=len(tasks_list))
#             sns.lineplot(x=first_resp_df.session, y=first_resp_df.error_x, hue=first_resp_df.subject,
#                          style=first_resp_df.subject, markers=True, ci=None, estimator=np.std, ax=axes)
#             ### last poke
#             task_palette = sns.set_palette([correct_other_c], n_colors=len(tasks_list))
#             sns.lineplot(x=last_resp_df.session, y=last_resp_df.error_x, hue=last_resp_df.subject,
#                          style=last_resp_df.subject, markers=True, ci=None, estimator=np.std, ax=axes)
#
#             axes.hlines(y=[stim_width], xmin=x_min, xmax=total_sessions, color=stim_c, linestyle=':')
#             axes.hlines(y=[vg_correct_th, wm_correct_th], xmin=x_min, xmax=total_sessions, color=correct_first_c,
#                         linestyle=':')
#             axes.hlines(y=[vg_repoke_th], xmin=x_min, xmax=total_sessions, color=repoke_th_c, linestyle=':')
#             axes.fill_between(first_resp_df.session, stim_width, 0, facecolor=stim_c, alpha=0.1)
#             axes.fill_between(first_resp_df.session, 160, 155, facecolor=lines_c, alpha=0.3) #chance
#             axes.set_ylabel('STD (mm)', label_kwargs)
#             axes.set_xlabel('')
#             axes.get_legend().remove()
#
#
#             # PROBABILITIES PLOT
#             df.loc[df['task'] != 'StageTraining_2B_V4', ['pwm_ds', 'pwm_dm', 'pwm_dl']] = df[
#                 ['pwm_ds', 'pwm_dm', 'pwm_dl']].multiply(df["pwm_d"], axis=0)
#
#             probs = df.groupby("session", as_index=True)[['pvg', 'pwm_i', 'pwm_d', 'pwm_ds', 'pwm_dm', 'pwm_dl']].mean()
#             probs_list = [probs.pvg, probs.pwm_i, probs.pwm_ds, probs.pwm_dm, probs.pwm_dl ]
#             probs_labels = ['VG', 'WM_I', 'WM_Ds', 'WM_Dm', 'WM_Dl']
#             probs_colors = [vg_c, wmi_c, wmds_c, wmdm_c, wmdl_c]
#
#             axes = plt.subplot2grid((50, 50), (25, 0), rowspan=4, colspan=50)
#             for idx, prob in enumerate(probs_list):
#                 sns.lineplot(x=probs.index, y=prob, ax=axes, color=probs_colors[idx])
#                 sns.scatterplot(x=probs.index, y=prob, ax=axes, color=probs_colors[idx], label=probs_labels[idx])
#
#             axes.get_legend().remove()
#             axes.hlines(y=[0.5, 1], xmin=x_min, xmax=total_sessions, color=lines_c, linestyle=':')
#             axes.set_xlabel('')
#             axes.get_xaxis().set_ticks([])
#             axes.set_ylim(-0.1, 1.1)
#             axes.set_ylabel('Probability \n appearance', label_kwargs)
#
#
#             # ACCURACY TRIAL INDEX TRIAL TYPE
#             axes = plt.subplot2grid((50, 50), (29, 0), rowspan=10, colspan=50)
#             ttype_palette = sns.set_palette(ttype_colors, n_colors=len(ttype_colors))
#
#             sns.lineplot(x=first_resp_df.session, y=first_resp_df.correct_bool, hue=first_resp_df.trial_type,
#                          style=first_resp_df.trial_type, markers=len(ttypes)*['o'], ax=axes, ci=None)
#             axes.hlines(y=[0.5, 1], xmin=x_min, xmax=total_sessions, color=lines_c, linestyle=':')
#             axes.fill_between(df.session, vg_chance_p, 0, facecolor=lines_c, alpha=0.3)
#             axes.fill_between(df.session, wm_chance_p, 0, facecolor=lines2_c, alpha=0.4)
#
#             utils.axes_pcent(axes, label_kwargs)
#             axes.legend(fontsize=8, title='Trial type', loc='center', bbox_to_anchor=(1.05, 0.5))
#
#             # STD TRIAL INDEX TRIAL TYPE
#             axes = plt.subplot2grid((50, 50), (40, 0), rowspan=10, colspan=50)
#             sns.lineplot(x=first_resp_df.session, y=first_resp_df.error_x, hue=first_resp_df.trial_type,
#                          style=first_resp_df.trial_type, markers=len(ttypes) * ['o'], ax=axes, estimator=np.std,ci=None)
#             axes.hlines(y=[stim_width], xmin=x_min, xmax=total_sessions, color=stim_c, linestyle=':')
#             axes.fill_between(first_resp_df.session, stim_width, 0, facecolor=stim_c, alpha=0.1)
#             axes.fill_between(first_resp_df.session, 160, 155, facecolor=lines_c, alpha=0.2)  # chance
#             axes.set_ylabel('STD (mm)', label_kwargs)
#             axes.get_legend().remove()
#
#         sns.despine()
#
#         # SAVING AND CLOSING PAGE 1
#         pdf.savefig()
#         plt.close()
#
#         if df.shape[0] > 0:
#
#             # PAGE 2
#             plt.figure(figsize=(11.7, 11.7))  # Apaisat
#
#             # ERRORS TRIAL TYPE
#             axes = plt.subplot2grid((50, 50), (0, 0), rowspan=9, colspan=50)
#
#             sns.lineplot(x=first_resp_df.session, y=first_resp_df.error_x, hue=first_resp_df.trial_type,
#                          style=first_resp_df.trial_type, markers=len(ttypes) * ['o'], ax=axes, ci=None)
#
#             axes.hlines(y=[-stim_width, stim_width], xmin=x_min, xmax=total_sessions, color=stim_c,
#                         linestyle=':')
#             axes.fill_between(first_resp_df.session, stim_width, -stim_width, color=stim_c, alpha=0.1)
#             axes.set_ylabel('$Errors\ (r_{t}-x_{t})\ (mm)%$', label_kwargs)
#             axes.set_ylim(-100, 100)
#             axes.set_xlabel('')
#             axes.get_legend().remove()
#
#             # STIMULUS DURATION PLOT
#             subset = df.loc[df['trial_type'] == 'WM_I']
#             subset['stim_respwin'] = subset['stim_duration'] - subset['fixation_time']
#             subset_color = (subset.ttype_colors.unique()).tolist()
#             task_palette = sns.set_palette(subset_color, n_colors=len(subset_color))
#
#             axes = plt.subplot2grid((50, 50), (11, 0), rowspan=5, colspan=50)
#             sns.lineplot(x=subset.session, y=subset.stim_respwin, style=subset.trial_type, markers=True, ax=axes)
#             axes.hlines(y=[0.2, 0.4, 0.6, 0.8, 1], xmin=x_min, xmax=total_sessions, color=lines2_c, linestyle=':', linewidth= 0.5)
#
#             axes.set_xlabel('')
#             axes.set_ylabel('Stim duration \n WM_I trials (sec)', label_kwargs)
#             axes.set_ylim([0, subset.stim_respwin.max() + 0.1])
#             axes.get_legend().remove()
#
#             last_day = round(subset.stim_respwin.iloc[-1], 2)
#             label = 'Last: ' + str(last_day) + ' sec'
#             axes.text(0.85, 1.05, label, transform=axes.transAxes, fontsize=8, fontweight='bold',
#                       verticalalignment='top')
#
#             # DELAY LENGHT PLOT
#             dtypes = df.delay_type.unique()
#             dtype_colors = []
#             for i in dtypes:
#                 if i == 'DS':
#                     dtype_colors.append(wmds_c)
#                 elif i == 'DM':
#                     dtype_colors.append(wmdm_c)
#                 elif i == 'DL':
#                     dtype_colors.append(wmdl_c)
#
#             dtypes_palette = sns.set_palette(dtype_colors, n_colors=len(dtype_colors))
#             axes = plt.subplot2grid((50, 50), (18, 0), rowspan=5, colspan=50)
#             sns.lineplot(x=df.session, y=df.delay, hue=df.delay_type, style=df.delay_type, markers=True, ax=axes)
#             axes.set_ylabel('Delay (sec)', label_kwargs)
#
#
#             ### SELECT LAST WEEK SESSIONS
#             first_resp_week = first_resp_df[
#                 (first_resp_df['session'] > total_sessions - 6) & (first_resp_df['session'] <= total_sessions)]
#             last_resp_week = last_resp_df[
#                 (last_resp_df['session'] > total_sessions - 6) & (last_resp_df['session'] <= total_sessions)]
#
#             # REPONSES HIST
#             ### order weekly ttypes list
#             week_ttypes = first_resp_week.trial_type.unique().tolist()
#             if 'WM_Ds' in week_ttypes:
#                 idx = week_ttypes.index('WM_Ds')
#                 week_ttypes.pop(idx)
#                 week_ttypes.insert(1, "WM_Ds")
#                 if 'WM_I' in week_ttypes:
#                     idx = week_ttypes.index('WM_I')
#                     week_ttypes.pop(idx)
#                     week_ttypes.insert(1, "WM_I")
#
#             axes_loc = [0, 11, 21, 31, 41]
#             for idx, ttype in enumerate(week_ttypes):
#                 subset = first_resp_week.loc[first_resp_week['trial_type'] == ttype]
#                 axes = plt.subplot2grid((50, 50), (31, axes_loc[idx]), rowspan=8, colspan=9)
#                 axes.set_title(ttype, fontsize=11, fontweight='bold')
#                 color = subset.ttype_colors.unique()
#
#                 sns.distplot(subset.response_x, kde=False, bins=bins_resp, color=color, ax=axes,
#                              hist_kws={'alpha': 0.9})
#                 sns.distplot(subset.x, kde=False, bins=bins_resp, color=lines2_c, ax=axes,
#                              hist_kws={'alpha': 0.4})
#                 axes.set_xlabel('$Responses\ (r_{t})\ (mm)%$', label_kwargs)
#                 axes.set_ylabel('')
#                 if ttype == 'VG':
#                     axes.set_ylabel('Nº of touches', label_kwargs)
#
#             # ERRORS HIST
#             for idx, ttype in enumerate(week_ttypes):
#                 subset = first_resp_week.loc[first_resp_week['trial_type'] == ttype]
#                 axes = plt.subplot2grid((50, 50), (42, axes_loc[idx]), rowspan=8, colspan=9)
#                 color = subset.ttype_colors.unique()
#                 correct_th = subset.correct_th.mean()
#
#                 sns.distplot(subset.error_x, kde=False, bins=bins_err, color=color, ax=axes,
#                              hist_kws={'alpha': 0.9})
#
#                 # vertical lines
#                 neg_lines_list = [-x for x in lines_list]
#                 all_lines = lines_list + neg_lines_list
#                 all_colors = lines_list_colors + lines_list_colors
#
#                 for idx, line in enumerate(all_lines):
#                     axes.axvline(x=line, color=all_colors[idx], linestyle=':', linewidth=1)
#                 axes.axvspan(-stim_width, stim_width, color=stim_c, alpha=0.1)
#
#
#                 axes.set_xlabel('$Errors\ (r_{t}-x_{t})\ (mm)%$', label_kwargs)
#                 axes.set_ylabel('')
#                 if ttype == 'VG':
#                     axes.set_ylabel('Nº of touches', label_kwargs)
#
#             sns.despine()
#
#             # SAVING AND CLOSING PAGE 2
#             pdf.savefig()
#             plt.close()
#
#         if df.shape[0] > 0:
#             # PAGE 3
#             plt.figure(figsize=(11.7, 11.7))  # Apaisat
#
#             # ACC TRIAL TYPE
#             axes = plt.subplot2grid((50, 50), (0, 0), rowspan=15, colspan=22)
#             x_axes = (first_resp_week.delay.unique()).tolist()
#             for idx, i in enumerate(x_axes):
#                 if i == -1.0:
#                     x_axes[idx] = 'VG'
#                 elif i == -0.5:
#                     x_axes[idx] = 'WM_I'
#             x_max = len(x_axes) - 1
#
#             # first reponses
#             sns.pointplot(x=first_resp_week.delay, y=first_resp_week.correct_bool, ax=axes, color=correct_first_c)
#             # last reponses
#             sns.pointplot(x=last_resp_week.delay, y=last_resp_week.correct_bool, ax=axes, color=correct_other_c)
#
#             axes.hlines(y=[0.5, 1], xmin=0, xmax=x_max, color=lines_c, linestyle=':')
#             chance_list = [vg_chance_p]
#             for i in (first_resp_week.trial_type.unique()):
#                 if i != 'VG':
#                     chance_list.append(wm_chance_p)
#
#             axes.fill_between(first_resp_week.trial_type.unique(), chance_list, 0, facecolor=lines_c, alpha=0.3)
#
#             axes.set_xlabel('Delay (sec)', label_kwargs)
#             axes.set_xticklabels(x_axes)
#             utils.axes_pcent(axes, label_kwargs)
#
#             # STD TRIAL TYPE
#             axes = plt.subplot2grid((50, 50), (0, 27), rowspan=15, colspan=22)
#             # first reponses
#             sns.pointplot(x=first_resp_week.delay, y=first_resp_week.error_x, ax=axes, color=correct_first_c,
#                          estimator=np.std)
#             # last reponses
#             sns.pointplot(x=last_resp_week.delay, y=last_resp_week.error_x, ax=axes, color=correct_other_c,
#                           estimator=np.std)
#             axes.hlines(y=[stim_width], xmin=0, xmax=len(ttypes) - 1, color=stim_c, linestyle=':')
#             axes.hlines(y=[vg_correct_th, wm_correct_th], xmin=0, xmax=len(ttypes) - 1, color=correct_first_c,
#                         linestyle=':')
#             axes.hlines(y=[vg_repoke_th], xmin=0, xmax=len(ttypes) - 1, color=repoke_th_c, linestyle=':')
#             axes.fill_between(first_resp_week.trial_type.unique(), stim_width, 0, facecolor=stim_c, alpha=0.1)
#             axes.fill_between(first_resp_week.trial_type.unique(), 160, 155, facecolor=lines_c, alpha=0.3)  # chance
#
#             axes.set_xticklabels(x_axes)
#             axes.set_xlabel('Delay (sec)')
#             axes.set_ylabel('STD (mm)')
#
#             # legend
#             colors = [correct_first_c, correct_other_c]
#             labels = ['First poke', 'Last poke']
#             lines = [Line2D([0], [0], color=colors[i], marker='o', markersize=6,
#                             markerfacecolor=colors[i]) for i in range(len(colors))]
#             axes.legend(lines, labels, fontsize=8, loc='center', bbox_to_anchor=(1, 0.9))
#
#             sns.despine()
#
#             # SAVING AND CLOSING PAGE 3
#             pdf.savefig()
#             plt.close()
#
#         print('intersession completed successfully')
