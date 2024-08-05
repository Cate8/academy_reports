from academy_reports import utils
from academy_reports import settings
from academy_reports import arg
import os
import pandas as pd
from datetime import timedelta, datetime

from report_tasks.S1 import daily_report_S1
from report_tasks.S2 import daily_report_S2
from report_tasks.S3 import daily_report_S3
from report_tasks.S4_5 import daily_report_S4_5
from report_tasks.water_calibration import report_water_calibration
from report_tasks.intersession import intersession
from report_tasks.temperature_reports import temperature_reports

import warnings
import traceback
from matplotlib.backends.backend_pdf import PdfPages
warnings.filterwarnings('ignore')




# MAIN
def main():


    #################################### DAILY & INTERSESSIONS ####################################
    print('')
    print('Generating dailies')


    # first we try to make a manual report by passing the path of a csv
    try:

        path = arg.file[0]
        file_name = os.path.basename(path)
        file_name = file_name.split(".")[0]

        df = pd.read_csv(path, sep=';')

        subject = df.subject.iloc[0]
        task = df.task.iloc[0]
        date = datetime.fromtimestamp(df.TRIAL_START.iloc[0]).strftime("%Y%m%d-%H%M%S")

        save_directory = os.path.join(settings.save_directory_manual, subject)

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        file_name = file_name + '.pdf'
        save_path = os.path.join(save_directory, file_name)

        if task == 'S3':
            daily_report_S3(df, save_path, date)
        elif task == 'S4_1':
            daily_report_S4_5(df, save_path, date)
        else:
            print('Task not found for file:', path, 'task:', task)

        print("succesfully manual report for file: ", path)

    # if path is not passed do for all the csvs
    except:

        print("---   starting water calibration plot")
        try:
            calibration_path = settings.calibration_path # the path is on the "setting" file
            df = pd.read_csv(calibration_path, sep=';')
            save_path = calibration_path[:-3] + 'pdf'
            report_water_calibration(df, save_path)
            print("---   water calibration plot succesfully done")
        except Exception as error:
            print(traceback.format_exc())
            print("---   error in water calibration plot")



        raw_paths = utils.path_generator(settings.data_directory, '.csv')

        if not os.path.exists(settings.data_directory):
            os.makedirs(settings.data_directory)

        dfs = []
        for path in raw_paths:

            #sort, only analyze general csvs
            subject = os.path.basename(path)
            # print(subject)
            if len(subject) <= 10:
                df = pd.read_csv(path, sep=';')
                dfs.append(df)

                subject, ext = os.path.splitext(subject)
                print('')
                print('Starting intersession report ' + str(subject))

                save_directory = os.path.join(settings.save_directory, subject)
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)


                #file_name_intersession = subject + '_intersession.pdf'
                #save_path_intersession = os.path.join(save_directory, file_name_intersession)
                #intersession(df.copy(), save_path_intersession)
                #print('intersession correct for subject: ', str(subject))
                
                # INTERSESSIONS
                try:
                    file_name_intersession = subject + '_intersession.pdf'
                    save_path_intersession = os.path.join(save_directory, file_name_intersession)
                    intersession(df.copy(), save_path_intersession)
                    print('intersession correct for subject: ', str(subject))
                except:
                    print('Error performing the intersession for the subject: ', str(subject))
                    pass
                

                # DAILY REPORTS
                for sess, session in df.groupby('session'):
                    subject = session.subject.iloc[0]
                    task = session.task.iloc[0]
                    stage = session.stage.iloc[0]
                    try:
                        date = datetime.fromtimestamp(session.TRIAL_START.iloc[0]).strftime("%Y%m%d-%H%M%S")
                    except:
                        date = session.TRIAL_START.iloc[0]

                    file_name = subject + '_' + task + '-' + str(stage) + '_' + date + '.pdf'

                    print("checking file: ", file_name)
                    save_path = os.path.join(save_directory, file_name)

                    if not os.path.exists(save_path): #ONLY DONE IF NOT EXISTS

                        
                        if task == 'S1':
                            print('Doing daily for file:', file_name)
                            try:
                                daily_report_S1(session.copy(), save_path, date)
                                print('correct daily for file:', file_name)
                            except Exception as error:
                                print(traceback.format_exc())
                                pdf_pages = PdfPages(save_path)
                                pdf_pages.close()
                                print('Error daily for file:', file_name)
                        elif task == 'S2':
                            print('Doing daily for file:', file_name)
                            try:
                                daily_report_S2(session.copy(), save_path, date)
                                print('correct daily for file:', file_name)
                            except Exception as error:
                                print(traceback.format_exc())
                                pdf_pages = PdfPages(save_path)
                                pdf_pages.close()
                                print('Error daily for file:', file_name)                        
                        elif task == 'S3':
                            print('Doing daily for file:', file_name)
                            try:
                                daily_report_S3(session.copy(), save_path, date)
                                print('correct daily for file:', file_name)
                            except Exception as error:
                                print(traceback.format_exc())
                                pdf_pages = PdfPages(save_path)
                                pdf_pages.close()
                                print('Error daily for file:', file_name)

                        elif task == 'S4_1':
                            print('Doing daily for file:', file_name)
                            try:
                                daily_report_S4_5(session.copy(), save_path, date)
                                print('correct daily for file:', file_name)
                            except Exception as error:
                                print(traceback.format_exc())
                                pdf_pages = PdfPages(save_path)
                                pdf_pages.close()
                                print('Error daily for file:', file_name)


                        elif task == 'S4_2':
                            print('Doing daily for file:', file_name)
                            try:
                                daily_report_S4_5(session.copy(), save_path, date)
                                print('correct daily for file:', file_name)
                            except Exception as error:
                                print(traceback.format_exc())
                                pdf_pages = PdfPages(save_path)
                                pdf_pages.close()
                                print('Error daily for file:', file_name)

                        elif task == 'S4_3':
                            print('Doing daily for file:', file_name)
                            try:
                                daily_report_S4_5(session.copy(), save_path, date)
                                print('correct daily for file:', file_name)
                            except Exception as error:
                                print(traceback.format_exc())
                                pdf_pages = PdfPages(save_path)
                                pdf_pages.close()
                                print('Error daily for file:', file_name)

                        elif task == 'S4_4':
                            print('Doing daily for file:', file_name)
                            try:
                                daily_report_S4_5(session.copy(), save_path, date)
                                print('correct daily for file:', file_name)
                            except Exception as error:
                                print(traceback.format_exc())
                                pdf_pages = PdfPages(save_path)
                                pdf_pages.close()
                                print('Error daily for file:', file_name)

                        elif task == 'S4_5':
                            print('Doing daily for file:', file_name)
                            try:
                                daily_report_S4_5(session.copy(), save_path, date)
                                print('correct daily for file:', file_name)
                            except Exception as error:
                                print(traceback.format_exc())
                                pdf_pages = PdfPages(save_path)
                                pdf_pages.close()
                                print('Error daily for file:', file_name)
                        
                        elif task == 'S4_5_batchA':
                            print('Doing daily for file:', file_name)
                            try:
                                daily_report_S4_5(session.copy(), save_path, date)
                                print('correct daily for file:', file_name)
                            except Exception as error:
                                print(traceback.format_exc())
                                pdf_pages = PdfPages(save_path)
                                pdf_pages.close()
                                print('Error daily for file:', file_name)

                        elif task == 'S4_5_batchA':
                                print('Doing daily for file:', file_name)
                                try:
                                    daily_report_S4_5(session.copy(), save_path, date)
                                    print('correct daily for file:', file_name)
                                except Exception as error:
                                    print(traceback.format_exc())
                                    pdf_pages = PdfPages(save_path)
                                    pdf_pages.close()
                                    print('Error daily for file:', file_name)

                        elif task == 'S4_5_single_pulse':
                            print('Doing daily for file:', file_name)
                            try:
                                daily_report_S4_5(session.copy(), save_path, date)
                                print('correct daily for file:', file_name)
                            except Exception as error:
                                print(traceback.format_exc())
                                pdf_pages = PdfPages(save_path)
                                pdf_pages.close()
                                print('Error daily for file:', file_name)

                        elif task == 'S4_5_train_pulse':
                            print('Doing daily for file:', file_name)
                            try:
                                daily_report_S4_5(session.copy(), save_path, date)
                                print('correct daily for file:', file_name)
                            except Exception as error:
                                print(traceback.format_exc())
                                pdf_pages = PdfPages(save_path)
                                pdf_pages.close()
                                print('Error daily for file:', file_name)


                    else:
                        print('Already done!')

        # GLOBAL DF
        print('Generating global df')
        global_df = pd.concat(dfs)
        save_directory = os.path.join(settings.data_directory)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        utils.create_csv(global_df, save_directory + '/global_trials.csv')
        print('END!')
#################################### TEMPERATURE & HUMIDITY REPORTS ####################################
    # try:
    print('')
    print('Generating temperature reports')
    df = pd.read_csv(settings.data_directory3, sep=';')

    date_format = '%Y/%m/%d %H:%M:%S'
    df['date_format'] = df['date'].apply(lambda x: datetime.strptime(x, date_format).date())
    last_date = df['date_format'].iloc[-1]
    

    filename = 'temperatures_' + str(last_date) + '.pdf'
    save_path = os.path.join(settings.save_directory3, filename)

    temperature_reports(df, last_date, save_path, settings.setup)

# MAIN
if __name__ == "__main__":
    main()

