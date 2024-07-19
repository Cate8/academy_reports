import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import timedelta, datetime
from pathlib import Path
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from academy_reports import settings


temp_c = 'purple'
hum_c = 'teal'
lines_c = 'gray'

temp_min = 20
temp_max = 24
hum_min = 30
hum_max = 65

def temperature_reports(df, last_date, save_path, setup):

    ############## PARSING #############
    date_format = '%Y/%m/%d %H:%M:%S'
    df['hour'] = df['date'].apply(lambda x: datetime.strptime(x, date_format).hour)
    last_date_df = df.loc[df['date_format'] == last_date]
    last_week_df = df.loc[df['date_format'] >= last_date - timedelta(days=7)]

    ############## PAGE 1 ##############
    with PdfPages(save_path) as pdf:
        plt.figure(figsize=(11.7, 16.5))  # A4

        #### HEADER
        header = ('Room: ' + str(setup) +'  /  Date: ' + str(last_date))

        axes = plt.subplot2grid((50, 50), (0, 0), rowspan=10, colspan=50)
        axes.text(0.1, 0.9, header, fontsize=20, transform=plt.gcf().transFigure)  # header

        ### PLOT 1: DAILY TEMPERATURE
        sns.lineplot(x='hour', y='temperature', ax=axes, marker='o', data=last_date_df, linewidth=2, zorder=15, color=temp_c)
        for temp in [temp_min, temp_max]:
            axes.axhline(temp, linestyle=':', color=lines_c)
        axes.set_ylabel('Temperature (ºC)')


        ### PLOT 2: DAILY HUMIDITY
        axes = plt.subplot2grid((50, 50), (12, 0), rowspan=10, colspan=50)
        sns.lineplot(x='hour', y='humidity', ax=axes, marker='o', data=last_date_df, linewidth=2, zorder=20, color=hum_c)
        for hum in [hum_min, hum_max]:
            axes.axhline(hum, linestyle=':', color=lines_c)
        axes.set_ylabel('Humidity (%)')


        ### PLOT 3: WEEKLY MEASUREMENTS
        axes = plt.subplot2grid((50, 50), (26, 0), rowspan=10, colspan=50)
        # TEMPERATURE WEEKLY
        sns.lineplot(x='date_format', y='temperature', ax=axes, marker='o', data=last_week_df, linewidth=2, zorder=20,
                     color=temp_c)
        axes.set_ylabel('Temperature (ºC)', color=temp_c)
        axes.set_ylim(18, 26)
        # HUMIDITY WEEKLY
        axes2 = axes.twinx()
        sns.lineplot(x='date_format', y='humidity', ax=axes2, marker='o', data=last_week_df, linewidth=2, zorder=20,
                     color=hum_c)
        axes2.set_ylabel('Humidity (%)', color=hum_c)
        axes2.set_ylim(30, 65)
        axes2.set_xlabel('Date')


        # SAVING AND CLOSING PAGE
        sns.despine()
        pdf.savefig()
        plt.close()

    now = datetime.now()
    hour = now.hour

    if hour in [12, 13, 14]:
        # send the email
        sender_email = "bcb.lab.bcn@gmail.com"
        sender_password = settings.app_password_google
        #recipient_email = "balmaserrano@gmail.com"
        recipient_email = "larasedo@ub.edu"
        path = save_path
        name = Path(path).name
        subject = "temperatura cellex4A"
        body = name

        with open(path, "rb") as attachment:
            # Add the attachment to the message
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {name}",
        )

        message = MIMEMultipart()
        message['Subject'] = subject
        message['From'] = sender_email
        message['To'] = recipient_email
        html_part = MIMEText(body)
        message.attach(html_part)
        message.attach(part)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())