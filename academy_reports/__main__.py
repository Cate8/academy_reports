from academy_reports import utils
from academy_reports import settings
from academy_reports import arg
import os
import pandas as pd
from datetime import datetime
from tasks.lickteaching_daily import lickteaching_daily
from tasks.touchteaching_daily import touchteaching_daily
from tasks.stagetraining_daily import stagetraining_daily
from tasks.intersession import intersession
from tasks.ecohab_report import ecohab_report
import warnings
warnings.filterwarnings('ignore')


# MAIN
def main():
    # VSRT REPORTS
    try:
        path = arg.file[0]
        file_name = os.path.basename(path)
        file_name = file_name.split(".")[0]
        print('making report for', file_name)

        df = pd.read_csv(path, sep=';')

        subject = df.subject.iloc[0]
        task = df.task.iloc[0]
        date = datetime.fromtimestamp(df.STATE_Start_task_START.iloc[0]).strftime("%Y%m%d-%H%M%S")

        save_directory = os.path.join(settings.save_directory_manual, subject)

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        file_name = file_name + '.pdf'
        save_path = os.path.join(save_directory, file_name)

        if task == 'LickTeaching':
            lickteaching_daily(df, save_path, date)
        elif task == 'TouchTeaching':
            touchteaching_daily(df, save_path, date)
        elif task[0:13] == 'StageTraining':
            stagetraining_daily(df, save_path, date)
        else:
            print('Task not found for file:', path, 'task:', task)


    except:
        raw_paths = utils.path_generator(settings.data_directory, '.csv')

        if not os.path.exists(settings.data_directory):
            os.makedirs(settings.data_directory)

        dfs = []
        for path in raw_paths:

            #sort, only analyze general csvs
            subject = os.path.basename(path)
            print(subject)
            if len(subject) <= 8:
                df = pd.read_csv(path, sep=';')
                dfs.append(df)

                subject, ext = os.path.splitext(subject)
                print('')
                print('Starting report '+str(subject))

                save_directory = os.path.join(settings.save_directory, subject)
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                # INTERSESSIONS
                try:
                    file_name_intersesion = subject + '_intersession.pdf'
                    save_path_intersesion = os.path.join(save_directory, file_name_intersesion)
                    intersession(df.copy(), save_path_intersesion)
                except:
                    print('Error performing the intersession')
                    pass

                # DAILY REPORTS
                for sess, session in df.groupby('session'):
                    subject = session.subject.iloc[0]
                    task = session.task.iloc[0]
                    stage = session.stage.iloc[0]
                    date = datetime.fromtimestamp(session.STATE_Start_task_START.iloc[0]).strftime("%Y%m%d-%H%M%S")
                    print(task[0:13])
                    print(sess)
                    print(date)

                    file_name = subject + '_' + task + '-' + str(stage) + '_' + date + '.pdf'
                    save_path = os.path.join(save_directory, file_name)

                    if not os.path.exists(save_path): #ONLY DONE IF NOT EXISTS
                        try:
                            if task == 'LickTeaching':
                                lickteaching_daily(session.copy(), save_path, date)
                            elif task == 'TouchTeaching':
                                touchteaching_daily(session.copy(), save_path, date)
                            elif task[0:13] == 'StageTraining':
                                stagetraining_daily(session.copy(), save_path, date)
                            else:
                                print('Task not found for file:', path, 'task:', task)
                        except:
                            print('Error performing the intersession')
                            pass
                    else:
                        print('Already done!')

        # GLOBAL DF
        try:
            global_df = pd.concat(dfs)
            save_directory = os.path.join(settings.save_directory)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            utils.create_csv(global_df, save_directory + 'global_trials.csv')
            print('END!')

        except:
            pass

    # ECOHAB REPORTS
    # merge different csvs
    datatype=None #gui or script
    column_names= ['Date', 'Time', 'Antena_number', 'Duration', 'RFID_detected']
    try:
        raw_paths = utils.path_generator(settings.data_directory2, '.csv')
        datatype='script'
    except:
        raw_paths = utils.path_generator(settings.data_directory2, '.txt')
        datatype='gui'

    if not os.path.exists(settings.data_directory2):
        os.makedirs(settings.data_directory2)

    dfs = []
    for path in raw_paths:
        if datatype =='script':
            df = pd.read_csv(path, sep='\t')
            # Calculate the event duration column (ms)
            df['times'] = pd.to_timedelta(df.Time)
            df['Duration'] = df['times'] - df['times'].shift()
            df['Duration'] = round(df['Duration'].dt.total_seconds() * 1000) # ms values
            # #correct changes of day
            day_ms = 24 * 60 * 60 * 1000
            df.loc[df['Duration'] < 0, 'Duration'] = df['Duration'] + day_ms
            #select columns of interest
            df=df[column_names]
            dfs.append(df)

        elif datatype =='gui':
            df = pd.read_csv(path, sep='\t', names=column_names)
            dfs.append(df)

        else:
            print('weird datatype')

    merged_df = pd.concat(dfs)
    save_directory2 = os.path.join(settings.save_directory2)
    if not os.path.exists(save_directory2):
        os.makedirs(save_directory2)
    utils.create_csv(merged_df, save_directory2 + 'merged_df.csv')

    # Make the report
    print('-----------------------------------------------------------------')
    print('Making Ecohab report')
    date_s=merged_df.Date.iloc[0]
    date_e=merged_df.Date.iloc[-1]
    print('From '+str(date_s))
    print('To '+str(date_e))

    file_name = 'Ecohab_report_' + date_s.replace('.','') + '_' + date_e.replace('.','') + '.pdf'
    save_path = os.path.join(save_directory2, file_name)
    ecohab_report(merged_df, save_path)
    # try:
    #     ecohab_report(df, save_path)
    # except:
    #     print('Error performing the EcoHAB report')
    
    


# MAIN
if __name__ == "__main__":
    main()

