import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
import numpy as np
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages


def lickteaching_daily (df, save_path):
    # relevant columns
    df['trial_result'] = 0
    df.loc[(df.STATE_Correct_first_START > 0, 'trial_result')] = 1
    df['lick_latency'] = df.STATE_Wait_for_reward_END - df.STATE_Wait_for_reward_START

    #relevant variables
    date = datetime.fromtimestamp(df.STATE_Start_task_START.iloc[0]).strftime("%Y-%m-%d %H:%M:%S")

    total_trials = int(df.trial.iloc[-1])
    valid_trials = df.trial_result.sum() #int(data.valid_trials.iloc[-1])
    missed_trials = total_trials - valid_trials
    reward_drunk =  valid_trials * 8 #int(data.reward_drunk.iloc[-1])

    lick_latency_mean = df.lick_latency.mean()


    # Plot colors

    correct_c = 'green'
    other_correct_c = 'lightgreen'
    miss_c = 'black'
    incorrect_c = 'red'
    punish_c = 'orange'

    water_c = 'teal'

    lines_c = 'gray'
    lines2_c = 'silver'


    # PAGE 1:

    with PdfPages(save_path) as pdf:

        plt.figure(figsize=(8.3, 11.7))  # A4 vertical

        # HEADER
        axes = plt.subplot2grid((50, 50), (0, 1), rowspan=4, colspan=50)


        s1 = ('Subject name: ' + str(df.subject.iloc[0]) +
              '  /  Task: ' + str(df.task.iloc[0]) +
              '  /  Date: ' + str(date) +
              '  /  Box: ' + str(df.box.iloc[0]) +
              '  /  Weight: ' + str(df.subject_weight.iloc[0]) + " g" + '\n')

        s2 = ('Total trials: ' + str(int(total_trials)) +
              '  /  Valid trials: ' + str(valid_trials) +
              '  /  Missed trials: ' + str(missed_trials) +
              '  /  Reward drunk: ' + str(reward_drunk) + " ul"+ '\n')

        axes.text(0.1, 0.9, s1+s2, fontsize=8, transform=plt.gcf().transFigure)


        # Trial result
        df['y_val']=1
        colors = [miss_c, correct_c]
        labels = ['miss', 'correct']
        lines = [Line2D([0], [0], color=c, marker='o', markersize=4, markerfacecolor=c) for c in colors]

        sns.scatterplot(x=df.trial, y=df.y_val, hue=df.trial_result, palette=colors, s=30, ax=axes)
        axes.axis('off')
        axes.legend(lines, labels, fontsize=6, loc='center', bbox_to_anchor=(0.95, 1.25))

        # Lick latency plot
        axes = plt.subplot2grid((50, 50), (5, 0), rowspan=8, colspan=50)
        sns.scatterplot(x=df.trial, y=df.lick_latency, color=water_c, s=30, ax=axes)
        sns.lineplot(x=df.trial, y=df.lick_latency, color=water_c, ax=axes)
        axes.hlines(y=5, xmin=0, xmax=total_trials, color='gray', linestyle=':')
        axes.set_ylabel('Lick latency (sec)')

        label = 'Mean: ' +str(round(lick_latency_mean, 1)) + ' sec'
        axes.text(0.85, 1.2, label, transform=axes.transAxes, fontsize=8, verticalalignment='top')

        sns.despine()


         # SAVING AND CLOSING PAGE
        pdf.savefig()
        plt.close()
