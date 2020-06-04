from academy_reports import utils
from academy_reports import settings
import pandas as pd
from datetime import datetime
import os
from tasks.lickteaching_daily import lickteaching_daily


# MAIN
def main():
    raw_paths = utils.path_generator(settings.data_directory, '.csv')

    if not os.path.exists(settings.data_directory):
        os.makedirs(settings.data_directory)

    for path in raw_paths:
        df = pd.read_csv(path, sep=';')
        for sess, session in df.groupby('session'):
            subject = session.subject[0]
            task = session.task[0]
            stage = session.stage[0]
            date = datetime.fromtimestamp(session.STATE_Start_task_START.iloc[0]).strftime("%Y%m%d-%H%M%S")
            file_name = subject + '_' + task + '-' + str(stage) + '_' + date + '.pdf'

            save_directory = os.path.join(settings.save_directory, subject)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            save_path = os.path.join(save_directory, file_name)


            if task == 'LickTeaching':
                lickteaching_daily(session.copy(), save_path)
            else:
                print('Task not found for file:', path, 'task:', task)



# MAIN
if __name__ == "__main__":
    main()

