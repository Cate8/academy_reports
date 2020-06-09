import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from academy_reports import utils


# PLOT COLORS
correct_first_c = 'green'
correct_other_c = 'lightgreen'
miss_c = 'black'
incorrect_c = 'orangered'
punish_c = 'red'
water_c = 'teal'
lines_c = 'gray'
all_poke_c = '#5B1132'

# BINNING
"""
Touchscreen active area: 1440*900 pixels --> 403.2*252 mm
Stimulus radius: 35pix (9.8mm)
x_positions: 35-1405 pix --> 9.8-393.4mm
"""
l_edge = 9.8
r_edge = 393.4
bins = np.linspace(l_edge, r_edge, 6)


def stagetraining_daily (df, save_path, date):

    # RELEVANT COLUMNS
    df.loc[((df.trial_result == 'miss') & (df.response_x != '[]'), 'trial_result')] = incorrect_c

    df['colors'] = miss_c
    df.loc[(df.trial_result == 'correct_first', 'colors')] = correct_first_c
    df.loc[(df.trial_result == 'correct_other', 'colors')] = correct_other_c
    df.loc[(df.trial_result == 'incorrect_c', 'colors')] = incorrect_c
    # df.loc[((df.trial_result == 'miss') & (df.response_x != '[]'), 'colors')] = incorrect_c
    df.loc[(df.trial_result == 'punish', 'colors')] = punish_c

    # REMOVE LIST BRACKETS & UNNEST THE REPONSES
    df['response_x'] = df['response_x'].apply(lambda x: x.replace('[', '').replace(']', ''))
    df = utils.convert_strings_to_lists(df, ['response_x', 'response_y'])
    resp_df = utils.unnesting(df, ['response_x', 'response_y'])
    resp_df['response_x'] = pd.to_numeric(resp_df.response_x, errors='coerce')

    # RELEVANT COLUMNS
    # if set(['STATE_Miss_reward_START']).issubset(df.columns):
    #     df['lick_time'] = df.STATE_Correct_first_reward_START.fillna(0) + df.STATE_Miss_reward_START.fillna(0)
    # else:
    #     df['lick_time'] = df.STATE_Correct_first_reward_START
    # df['lick_latency'] = df.lick_time - df.STATE_Response_window_END


    # RELEVANT VARIABLES
    subject = df.subject.iloc[0]
    weight = df.subject_weight.iloc[0]
    total_trials = int(df.trial.iloc[-1])
    valid_trials = (df.loc[df['trial_result'] != 'miss']).shape[0]
    missed_trials = total_trials - valid_trials
    reward_drunk =  int(df.reward_drunk.iloc[-1])

    total_vg = df[df.trial_type == 'VG'].shape[0]
    total_wmi = df[df.trial_type == 'WM_I'].shape[0]
    total_wmd = df[df.trial_type == 'WM_D'].shape[0]

    total_acc_first_poke = df[df.trial_result == 'correct_first'].shape[0] / valid_trials
    total_acc_last_poke = df[(df.trial_result == 'correct_first') & (df.trial_result == 'correct_other')].shape[0] / valid_trials
    total_acc_vg = df[(df.trial_type == 'VG') & (df.trial_result == 'correct_first')].shape[0] / total_vg * 100

    if total_wmi > 0:
        total_acc_wmi = df[(df.trial_type == 'WM_I') & (df.trial_result == 'correct_first')].shape[0] / total_wmi * 100
        acc_wm = '  /  Acc WM Intro: ' + str(round(total_acc_wmi, 2))
    if total_wmd > 0:
        total_acc_wmd = df[(df.trial_type == 'WM_D') & (df.trial_result == 'correct_first')].shape[0] / total_wmd * 100
        acc_wmd = '  /  Acc WM Delay: ' + str(round(total_acc_wmd, 2))
        if total_wmi > 0:
            acc_wm = acc_wm + acc_wmd
        else:
            acc_wm = acc_wmd

    # PAGE 1:
    with PdfPages(save_path) as pdf:

        plt.figure(figsize=(8.3, 11.7))  # A4 vertical

        # HEADER
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=8, colspan=50)
        s1 = ('Subject: ' + str(subject) +
              '  /  Task: ' + str(df.task.iloc[0]) +
              '  /  Date: ' + str(date) +
              '  /  Box: ' + str(df.box.iloc[0]) +
              '  /  Weight: ' + str(weight) + " g" +
              '  /  Relative weight: ' + str(round(utils.relative_weights(subject, weight), 2)) + "%" +'\n')

        s2 = ('Total trials: ' + str(int(total_trials)) +
              '  /  Valid trials: ' + str(valid_trials) +
              '  /  Missed trials: ' + str(missed_trials) +
              '  /  Reward drunk: ' + str(reward_drunk) + " ul" + '\n')

        s3 = ('Prob VG: ' + str((df.pvg.iloc[0])*100) +
              '  /  Prob WM Intro: ' + str((df.pwm_i.iloc[0])*100) +
              '  /  Prob WM Delay: ' + str((df.pwm_d.iloc[0])*100) +
              '  /          '  +
              '  /  Delay s: ' + str((df.pwm_i.iloc[0]) * 100) +
              '  /  Delay m: ' + str((df.pwm_d.iloc[0]) * 100) +
              '  /  Delay l: ' + str((df.pwm_d.iloc[0]) * 100) + '\n')

        s4 = ('Acc first poke: ' + str(round(total_acc_first_poke, 2)) + '%' +
              '  /  Acc last poke: ' + str(round(total_acc_last_poke, 2) * 100) +
              '  /  Acc VG: ' + str(round(total_acc_vg, 2)) + acc_wm + '\n')


        axes.text(0.1, 0.9, s1 + s2 + s3 +s4, fontsize=8, transform=plt.gcf().transFigure)



        # SAVING AND CLOSING PAGE
        pdf.savefig()
        plt.close()