import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from academy_reports import utils
from datetime import timedelta, datetime
from matplotlib.lines import Line2D
# from matplotlib.patches import Patch


def ecohab_report (df, save_path):

    ##################### PARSE #####################

    # ECOHAB detects weird RDIF lectures sometimes: good lectures finish in 04 and are consistently detected
    all_subjects, all_ecohab_tags, all_colors = utils.subjects_tags()

    ### Remove aberrant detections
    match_tags = []
    df['tag'] = df['RFID_detected'].apply(lambda x: x.strip("0"))  # remove final 0

    for x in df.tag.unique():
        for tag in all_ecohab_tags:  # loop thought correct tags
            if x in tag:  # check if parts of the incorrect tag coincide with real tags
                match_tags.append(x)

    unique_match_tags = []  # list with the matching tags
    for x in match_tags:
        if x not in unique_match_tags:
            unique_match_tags.append(x)

    df = df.loc[((df['tag'].isin(match_tags)))] # remove aberrant detections

    ### Create a column containing the corrected tags
    def tag_correction(x):
        if len(x) < 10:
            for tag in all_ecohab_tags:  # loop thought correct tags
                if x in tag:
                    return tag
        else:
            return x
    df['tag'] = df['tag'].apply(lambda x: tag_correction(x))

    ### add subject names column
    df['subject'] = df['tag'].replace(all_ecohab_tags, all_subjects)

    ### Create a column with a colr assigned to each subject
    df['colors'] = df['subject'].replace(all_subjects, all_colors)

    # Box inference
    df['prev_antena'] = df['Antena_number'].shift()

    df['box'] = None
    df.loc[((df['Antena_number'] == 1) & (df['prev_antena'] == 1)), 'box'] = 'A'
    df.loc[((df['Antena_number'] == 8) & (df['prev_antena'] == 8)), 'box'] = 'A'
    df.loc[((df['Antena_number'] == 1) & (df['prev_antena'] == 8)), 'box'] = 'A'
    df.loc[((df['Antena_number'] == 8) & (df['prev_antena'] == 1)), 'box'] = 'A'
    df.loc[((df['Antena_number'] == 8) & (df['prev_antena'] == 7)), 'box'] = 'A'
    df.loc[((df['Antena_number'] == 1) & (df['prev_antena'] == 2)), 'box'] = 'A'
    df.loc[((df['Antena_number'] == 2) & (df['prev_antena'] == 2)), 'box'] = 'B'
    df.loc[((df['Antena_number'] == 3) & (df['prev_antena'] == 3)), 'box'] = 'B'
    df.loc[((df['Antena_number'] == 2) & (df['prev_antena'] == 3)), 'box'] = 'B'
    df.loc[((df['Antena_number'] == 3) & (df['prev_antena'] == 2)), 'box'] = 'B'
    df.loc[((df['Antena_number'] == 2) & (df['prev_antena'] == 1)), 'box'] = 'B'
    df.loc[((df['Antena_number'] == 3) & (df['prev_antena'] == 4)), 'box'] = 'B'
    df.loc[((df['Antena_number'] == 4) & (df['prev_antena'] == 4)), 'box'] = 'C'
    df.loc[((df['Antena_number'] == 5) & (df['prev_antena'] == 5)), 'box'] = 'C'
    df.loc[((df['Antena_number'] == 4) & (df['prev_antena'] == 5)), 'box'] = 'C'
    df.loc[((df['Antena_number'] == 5) & (df['prev_antena'] == 4)), 'box'] = 'C'
    df.loc[((df['Antena_number'] == 4) & (df['prev_antena'] == 3)), 'box'] = 'C'
    df.loc[((df['Antena_number'] == 5) & (df['prev_antena'] == 6)), 'box'] = 'C'
    df.loc[((df['Antena_number'] == 6) & (df['prev_antena'] == 6)), 'box'] = 'D'
    df.loc[((df['Antena_number'] == 7) & (df['prev_antena'] == 7)), 'box'] = 'D'
    df.loc[((df['Antena_number'] == 6) & (df['prev_antena'] == 7)), 'box'] = 'D'
    df.loc[((df['Antena_number'] == 7) & (df['prev_antena'] == 6)), 'box'] = 'D'
    df.loc[((df['Antena_number'] == 6) & (df['prev_antena'] == 5)), 'box'] = 'D'
    df.loc[((df['Antena_number'] == 7) & (df['prev_antena'] == 8)), 'box'] = 'D'

    # Skipped detections: events without logic order
    null = df.loc[df['box'].isnull()]
    print('\n Skipped %: ' + str(round((null.shape[0] / df.shape[0]) * 100, 2)))

    # Create a datetime column
    df['Datetime'] = df['Date'].str.cat(df['Time'], sep=':')
    df['Datetime'] = df['Datetime'].apply(lambda x: datetime.strptime(x, '%Y.%m.%d:%H:%M:%S.%f'))


    ################# RELEVANT VARIABLES ################
    n_subjects = len(df.subject.unique())
    n_days= len(df.Date.unique())
    boxes = df.box.dropna().unique().tolist()
    boxes.sort()
    n_boxes=len(boxes)

    box_palette = ['m', 'mediumseagreen', 'greenyellow', 'salmon']
    label_kwargs = {'fontsize': 9}

    #################### PLOT #####################

    ############## PAGE 1 ##############
    with PdfPages(save_path) as pdf:

        plt.figure(figsize=(11.7, 15))

        #### HEADER
        s1 = ('Subjects: ' + str(n_subjects) +
              '  /  Days: ' + str(n_days) +
              '  /  Cages: ' + str(n_boxes) + '\n')

        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=1, colspan=50)
        axes.axis('off')
        # axes
        axes.text(0.1, 0.9, s1, fontsize=8, transform=plt.gcf().transFigure)  # header


        ### PLOT 1: Box raster plot by days
        # select plot width depending on the number of subjects
        row_w= int(30/n_subjects)
        y = 1

        # vertical lines
        days_at_8 = []
        days_at_20 = []
        for date in df['Date'].unique():
            day_datetime = datetime.strptime(date + ':08:00:00.000', '%Y.%m.%d:%H:%M:%S.%f')
            night_datetime = datetime.strptime(date + ':20:00:00.000', '%Y.%m.%d:%H:%M:%S.%f')
            days_at_8.append(day_datetime)
            days_at_20.append(night_datetime)
        x_min = df['Datetime'].min() - timedelta(hours=1)
        x_max = df['Datetime'].max() + timedelta(hours=1)

        #plot
        for n, subject in enumerate(df.subject.unique()):  # loop thoug differnt subjects
            axes = plt.subplot2grid((50, 50), (y, 0), rowspan=row_w, colspan=50)
            subset = df.loc[df['subject'] == subject]
            subset['y_order'] = subset['box'].replace(boxes, np.arange(0, n_boxes))
            try:
                sns.scatterplot(x='Datetime', y='y_order', data=subset, ax=axes, hue='y_order', palette=box_palette,
                                edgecolor='none', s=8)
            except:
                print(subject)
                print(subset.box.unique())
                print(subset.y_order.unique())
            for lines in range(len(days_at_8) - 1):
                axes.axvspan(days_at_20[lines], days_at_8[lines + 1], facecolor='lightgray', zorder=0)
            # axes.eventplot(days_at_8, color='black', linelengths=20, lineoffsets=0, linewidths=1)
            # axes.eventplot(days_at_20, color='black', linelengths=20, lineoffsets=0, linewidths=1)

            #axes
            axes.set_ylim(-1, 4)
            axes.set_ylabel(subject)
            axes.set_yticks([])
            axes.set_xlim(x_min, x_max)
            if n != 0:
                try:
                    axes.get_legend().remove()
                except:
                    pass
                if n < len(df.subject.unique()) - 1:
                    axes.get_xaxis().set_visible(False)
            else:
                lines = [Line2D([0], [0], color=box_palette[i], marker='o', markersize=7,
                                 markerfacecolor=box_palette[i], linestyle=None) for i in range(len(box_palette))]
                axes.legend(lines, boxes, title='Box', loc='center', bbox_to_anchor=(1, 1), fontsize=6, title_fontsize=8)
                axes.get_xaxis().set_visible(False)

            y=y+2



        ### PLOT 2: Box occupancy histograms
        # caluclate time (hours) by subject box and date
        group_df = df.groupby(['subject', 'box', 'Date']).agg({'Duration': 'sum', 'colors': 'max'}).reset_index()
        group_df['hours'] = group_df['Duration'] / (1000 * 60 * 60)  # DURATION IN MS --> HOURS

        y = [31, 31, 41, 41]
        x = [0, 14, 0, 14]

        for i, box in enumerate(boxes):
            axes = plt.subplot2grid((50, 50), (y[i], x[i]), rowspan=9, colspan=11)
            subset = group_df.loc[group_df['box'] == box]
            order = subset.groupby(["subject", 'colors'])["hours"].median().sort_values(ascending=False).reset_index()
            sns.barplot(data=order, x="subject", y="hours", ax=axes,
                            palette=sns.set_palette(sns.color_palette(order.colors)))
            axes.set_title('Box: ' + str(box), fontsize=10, fontweight='bold', y=1.0, pad=-14)
            axes.set_xticklabels(axes.get_xticklabels(), rotation=70)

        ### PLOT 3: Antenas histogram
        axes = plt.subplot2grid((50, 50), (31, 30), rowspan=10, colspan=10)
        sns.countplot(x='Antena_number', data=df, palette='GnBu', ax=axes)

        # SAVING AND CLOSING PAGE
        sns.despine()
        pdf.savefig()
        plt.close()








