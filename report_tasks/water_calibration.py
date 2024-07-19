import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os



def report_water_calibration(df, save_path):

    # Remove index
    remove_index = [0, 1, 2, 3, 4, 5, 6, 7, 12]
    remove_column = 'Unnamed: 6'
    df.drop(remove_index, inplace=True)
    df.drop(remove_column, axis=1, inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # formatting the data
    df['date'] = df['date'].str.split().str[0]
    df['port'] = df['port'].replace({5: 'right_port5', 2: 'left_port2'})
    df['pulse_duration'] = df['pulse_duration'] * 1000       #milliseconds
    df['water'] = df['water'].apply(lambda x: f"{x:.1f}")    #just one decimal
    df['date'] = df['date'].str.replace('2024/', '', regex=False)   # Remove '2024/'

    # DataFrame per i dati 'right'
    df['port'] = df['port'].replace({5: 'right_port5', 2: 'left_port2'})

    # Df for right(5) port data
    df_r = df[df['port'] == 'right_port5'].copy()

    # Df for left (2) port data
    df_l = df[df['port'] == 'left_port2'].copy()

    # Convert 'volumes' and 'pulses' columns to lists of floats
    df_l['volumes'] = df['volumes'].apply(lambda x: [float(i) for i in x.split(',')])
    df_l['pulses'] = df['pulses'].apply(lambda x: [float(i) for i in x.split(',')])
    df_r['volumes'] = df['volumes'].apply(lambda x: [float(i) for i in x.split(',')])
    df_r['pulses'] = df['pulses'].apply(lambda x: [float(i) for i in x.split(',')])

    df_l.loc[:, 'volumes'] = df['volumes'].apply(lambda x: [float(i) for i in x.split(',')])
    df_l.loc[:, 'pulses'] = df['pulses'].apply(lambda x: [float(i) for i in x.split(',')])
    df_r.loc[:, 'volumes'] = df['volumes'].apply(lambda x: [float(i) for i in x.split(',')])
    df_r.loc[:, 'pulses'] = df['pulses'].apply(lambda x: [float(i) for i in x.split(',')])

    # Explode the 'volumes' and 'pulses' columns
    # This will create a new row for each value in the list while keeping the other column values intact.
    df_exp_l = df_l.explode('volumes').explode('pulses').reset_index(drop=True)
    df_exp_r = df_r.explode('volumes').explode('pulses').reset_index(drop=True)


    #preparing the data: converting units and roundings
    df_exp_l['pulses'] = df_exp_l['pulses'] * 1000    #milliseconds
    df_exp_r['pulses'] = df_exp_r['pulses'] * 1000    #milliseconds

    #just one decimal
    df_exp_l['pulse_duration'] = df_exp_l['pulse_duration'].apply(lambda x: f"{x:.1f}")
    df_exp_r['pulse_duration'] = df_exp_r['pulse_duration'].apply(lambda x: f"{x:.1f}")

    #PLOTTING
    #PDF PAGE ORGANIZATION

    fig, axs = plt.subplots(3, 2, figsize=(20, 20))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.4)

    # Appiattisci l'array bidimensionale in un array unidimensionale
    axs = axs.flatten()


    # Plot 1: volumes during calibration left (port2)
    axs[0].scatter(df_exp_l['date'], df_exp_l['volumes'], color="green", s=70)
    axs[0].set_title('Water volumes left (Port2)', fontsize=22)  # Imposta esplicitamente la dimensione qui
    axs[0].set_ylabel('volume (uL)', fontsize=18)
    axs[0].tick_params(axis='x', labelsize=16, rotation=45)
    axs[0].tick_params(axis='y', labelsize=16)

    axs[0].axhline(y=5, color='gray', linestyle='--', linewidth=2)
    axs[0].axhline(y=5.2, color='gray', linestyle='--', linewidth=2)
    axs[0].axhline(y=4.8, color='gray', linestyle='--', linewidth=2)
    x_min, x_max = axs[0].get_xlim()
    axs[0].fill_between([x_min, x_max], 4.8, 5.2, color='yellow', alpha=0.2)

    # Plot 2: volumes during calibration right (port5)
    axs[1].scatter(df_exp_r['date'], df_exp_r['volumes'], color="purple", s=70)
    axs[1].set_title('Water volumes right (Port5)', fontsize=22)  # Imposta esplicitamente la dimensione qui
    axs[1].set_ylabel('volume (uL)', fontsize=18)
    axs[1].tick_params(axis='x', labelsize=16, rotation=45)
    axs[1].tick_params(axis='y', labelsize=16)

    axs[1].axhline(y=5, color='gray', linestyle='--', linewidth=2)
    axs[1].axhline(y=5.2, color='gray', linestyle='--', linewidth=2)
    axs[1].axhline(y=4.8, color='gray', linestyle='--', linewidth=2)
    x_min, x_max = axs[0].get_xlim()
    axs[1].fill_between([x_min, x_max], 4.8, 5.2, color='yellow', alpha=0.2)

    # Plot 3: pulse duration left (port2)
    axs[2].plot(df_exp_l['date'], df_exp_l['pulse_duration'], color="green", label='Pulse Duration')
    axs[2].scatter(df_exp_l['date'], df_exp_l['pulse_duration'], color="green", s=50)
    axs[2].set_title('Final pulse durations left (Port2)', fontsize=22)  # Imposta esplicitamente la dimensione qui
    axs[2].set_ylabel('Duration (msec)', fontsize=18)
    axs[2].tick_params(axis='x', labelsize=16, rotation=45)
    axs[2].tick_params(axis='y', labelsize=16)

    # Plot 4: pulse duration right (port5)
    axs[3].plot(df_exp_r['date'], df_exp_r['pulse_duration'], color="purple", label='Pulse Duration')
    axs[3].scatter(df_exp_r['date'], df_exp_r['pulse_duration'], color="purple", s=50)
    axs[3].set_title('Final pulse durations right (Port5)', fontsize=22)  # Imposta esplicitamente la dimensione qui
    axs[3].set_ylabel('Duration (msec)', fontsize=18)
    axs[3].tick_params(axis='x', labelsize=16, rotation=45)
    axs[3].tick_params(axis='y', labelsize=16)

    # Plot 5: pulses during calibration left (port2)
    axs[4].scatter(df_exp_l['date'], df_exp_l['pulses'], color="green", s=70)
    axs[4].set_title('Pulses duration left (Port2)', fontsize=22)  # Imposta esplicitamente la dimensione qui
    axs[4].set_ylabel('Duration (msec)', fontsize=18)
    axs[4].tick_params(axis='x', labelsize=16, rotation=45)
    axs[4].tick_params(axis='y', labelsize=16)
    axs[4].axhline(y=7.6, color='gray', linestyle='--', linewidth=2)

    # Plot 6: pulses during calibration right (port5)
    axs[5].scatter(df_exp_r['date'], df_exp_r['pulses'], color="purple", s=70)
    axs[5].set_title('Pulses duration right (Port2)', fontsize=22)  # Imposta esplicitamente la dimensione qui
    axs[5].set_ylabel('Duration (msec)', fontsize=18)
    axs[5].tick_params(axis='x', labelsize=16, rotation=45)
    axs[5].tick_params(axis='y', labelsize=16)
    axs[5].axhline(y=8.5, color='gray', linestyle='--', linewidth=2)


    pdf_pages = PdfPages(save_path)
    # Save the plot to the PDF
    pdf_pages.savefig()
    # Close the PdfPages object to save the PDF file


    pdf_pages.close()

















