from academy_reports import utils
from academy_reports import settings
import os
import pandas as pd
from datetime import datetime
from tasks.lickteaching_daily import lickteaching_daily
from tasks.touchteaching_daily import touchteaching_daily
from tasks.stagetraining_daily import stagetraining_daily
from tasks.intersession import intersession


# MAIN
def main():
    raw_paths = utils.path_generator(settings.data_directory, '.csv')

    if not os.path.exists(settings.data_directory):
        os.makedirs(settings.data_directory)

    dfs = []
    for path in raw_paths:
        df = pd.read_csv(path, sep=';')
        dfs.append(df)

        subject = os.path.basename(path)
        subject, ext = os.path.splitext(subject)
        print('Starting report '+str(subject))

        save_directory = os.path.join(settings.save_directory, subject)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # INTERSESSIONS
        file_name_intersesion = subject + '_intersession.pdf'
        save_path_intersesion = os.path.join(save_directory, file_name_intersesion)
        intersession(df.copy(), save_path_intersesion)

        # DAILY REPORTS
        for sess, session in df.groupby('session'):
            subject = session.subject.iloc[0]
            task = session.task.iloc[0]
            print(task[0:13])
            print(sess)
            stage = session.stage.iloc[0]
            date = datetime.fromtimestamp(session.STATE_Start_task_START.iloc[0]).strftime("%Y%m%d-%H%M%S")

            file_name = subject + '_' + task + '-' + str(stage) + '_' + date + '.pdf'
            save_path = os.path.join(save_directory, file_name)

            if task == 'LickTeaching':
                lickteaching_daily(session.copy(), save_path, date)
            elif task == 'TouchTeaching':
                touchteaching_daily(session.copy(), save_path, date)
            elif task[0:13] == 'StageTraining':
                stagetraining_daily(session.copy(), save_path, date)
            else:
                print('Task not found for file:', path, 'task:', task)

    # GLOBAL DF
    global_df = pd.concat(dfs)
    save_directory = os.path.join(settings.global_directory)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    utils.create_csv(global_df, save_directory + 'global_trials.csv')
    print('END!')


# MAIN
if __name__ == "__main__":
    main()

