import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

df = pd.read_excel('C:/academy_reports/academy_reports/sessions/weight/weight.xlsx')
df.to_csv('C:/academy_reports/academy_reports/sessions/weight/weight.csv', encoding='utf-8', index=False)

path = "C:/academy_reports/academy_reports/sessions/weight/weight.csv"
save_path = path[:-3] + 'pdf'
def daily_report_weight(path, save_path):
    df = pd.read_csv(path, sep=',')

raw_df = pd.read_csv(path, sep=',')

df = raw_df.copy()

df_10 = df[df['id'] == 10]
df_10 = df_10.sort_values(by='stage')

df_16 = df[df['id'] == 16]
df_16 = df_16.sort_values(by='stage')

df_17 = df[df['id'] == 17]
df_17 = df_17.sort_values(by='stage')

# PLOT 1: relative weight thought time
axes = plt.subplot2grid((3, 3), (0, 0), rowspan=1, colspan=3)

plt.plot(df_10['date'], df_10['relative_weight'], label='010', color="orange")
plt.plot( df_16['date'], df_16['relative_weight'], label='016', color="turquoise")
plt.plot(df_17['date'], df_17['relative_weight'], label='010', color="magenta")

plt.xticks(rotation=90, fontsize=5)
plt.yticks(fontsize=5)

# Aggiungere titoli e etichette
plt.title('relative weight through time', fontsize=7)
plt.xlabel('Time', fontsize=7)
plt.ylabel('relative weight (%)', fontsize=7)
plt.legend()

# PLOT 2: relative weight vs trials stage
df_averages = raw_df.copy()
df_averages = df_averages.sort_values(by='stage')









pdf_pages = PdfPages(save_path)
# Save the plot to the PDF
pdf_pages.savefig()
# Close the PdfPages object to save the PDF file
pdf_pages.close()

root_folder = "C:\\academy_reports\\academy_reports\\sessions\\weight\\weight.csv"

#Recursively search for CSV files
for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith(".csv"):
            print('csv file: ', file)
            path = os.path.join(root, file)
            print('csv path: ', path)
            save_path = path[:-3] + 'pdf'

            #if not "raw" in file and "S3" in file and not os.path.exists(save_path): # requires manually delete the generated pdf in order to generate a new one
            if not "raw" in file and "S3" in file: # overwrite the pdf to the previous one

                daily_report_weight(path, save_path)

print("a")