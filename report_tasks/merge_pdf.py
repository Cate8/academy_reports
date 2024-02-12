import PyPDF2
import os
import datetime
import time

# Folder containing consecutive PDFs with the same name
folder_path = 'C:\\academy_reports\\academy_reports\\sessions\\merge'


# Get the list of PDF files
pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]

# Create a dictionary where the keys are the first 5 letters of the file name
# and values are lists of files with the same prefix
file_dict = {}
for pdf_file in pdf_files:
    prefix = pdf_file[:5]  # Extract the first 5 letters of the file name
    if prefix not in file_dict:
        file_dict[prefix] = []
    file_dict[prefix].append(pdf_file)

# Crea a PdfMerger object for each group of files with the same prefix
for prefix, files in file_dict.items():
    pdf_merger = PyPDF2.PdfMerger()

    # Reverse the order of added PDF files
    for pdf_file in reversed(files):
        pdf_file_path = os.path.join(folder_path, pdf_file)
        pdf_merger.append(pdf_file_path)  # Passa il percorso del file PDF

    # current date
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # merge PDF e save in the same folder
    output_filename = f'merged_{prefix}_{current_time}.pdf'
    output_path = os.path.join(folder_path, output_filename)
    with open(output_path, 'wb') as output_file:
        pdf_merger.write(output_file)
    pdf_merger.close()

    # Delete single PDF files used
    for pdf_file in files:
        pdf_file_path = os.path.join(folder_path, pdf_file)
        os.remove(pdf_file_path)