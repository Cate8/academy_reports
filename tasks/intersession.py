import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

# PLOT COLORS
correct_c = 'green'
miss_c = 'black'
water_c = 'teal'
all_poke_c = '#5B1132'
second_poke_c = '#96367C'
first_poke_c = '#C97F74'

def intersession(df, save_path_intersesion):

    # RELEVANT COLUMNS
    df['trial_result'] = 0
    df.loc[(df.STATE_Correct_first_START > 0, 'trial_result')] = 1
    df['lick_latency'] = df.STATE_Wait_for_reward_END - df.STATE_Wait_for_reward_START

    # RELEVANT VARIABLES


    with PdfPages(save_path_intersesion) as pdf:

        plt.figure(figsize=(8.3, 11.7))  # A4 vertical

        # NUMBER OF TRIALS
        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=6, colspan=50)


        ### number of trials
        total_trials_s = df.groupby(['session'], sort=False)['trial'].max()
        sns.scatterplot(total_trials_s.index, total_trials_s, color=miss_c,  s=30, ax=axes, label='All')
        sns.lineplot(total_trials_s.index, total_trials_s, color=miss_c, ax= axes)
        ### number of valid trials
        valid_trials_s = df.groupby(['session'], sort=False)['trial_result'].sum()
        sns.scatterplot(valid_trials_s.index, valid_trials_s, color=all_poke_c, s=30, ax=axes, label='Valids')
        sns.lineplot(valid_trials_s.index, valid_trials_s, color=all_poke_c, ax=axes)

        axes.set_ylabel('Number of trials')
        axes.legend()
        sns.despine()


        # axes.set_xticks(df.session)
        # axes.set_xlabel("", label_kwargs)
        # , label_kwargs)
        #
        # colors = [all_poke_c, first_poke_c]
        # labels = ['All trials', 'Valid']
        # lines = [Line2D([0], [0], color=c, marker='o', markersize=3, markerfacecolor=c) for c in colors]
        # axes.legend(lines, labels, fontsize=6)


        # # LICKPORT LATENCY
        # axes = plt.subplot2grid((50, 50), (16, 26), rowspan=12, colspan=26)
        # # axes.plot(df.session, df.lickport_latency, c=other_c, marker='o', markersize=3)
        # # axes.plot(df.session, df.lickport_latency, c=other_c, marker='o', markersize=3)
        #
        # axes.errorbar(df.session, df.licklat_miss_mean, yerr=df.licklat_miss_std.values / np.sqrt(total_sessions),
        #               c=other_c, marker='o', markersize=2)
        # axes.errorbar(df.session, df.licklat_correct_first_mean,
        #               yerr=df.licklat_correct_first_std.values / np.sqrt(total_sessions),
        #               c=first_correct_c, marker='o', markersize=2)
        #
        # axes.set_xlabel("Session number", label_kwargs)
        # axes.set_ylabel("Lickport latency (sec)", label_kwargs)
        # axes.set_xticks(df.session)
        #
        # colors = [first_correct_c, other_c]
        # labels = ['Correct', 'Miss']
        # lines = [Line2D([0], [0], color=c, marker='o', markersize=5, markerfacecolor=c) for c in colors]
        # axes.legend(lines, labels, fontsize=6)
        #

        #weight

        #reponsesscatterplot vs session
        # SAVING AND CLOSING PAGE 1
        pdf.savefig()
        plt.close()
