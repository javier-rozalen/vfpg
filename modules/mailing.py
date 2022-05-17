import smtplib,getpass
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.audio import MIMEAudio
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.utils import formatdate

class email_code():
    """
    Example of use
    -------------------
    
    my_mail = email_code()   
    my_mail.login() 
    receiver = 'receiver_address@gmail.com'    
    subject = 'Subject of the message'
    text = 'Text of the message'
    files = [path_to_file_1, path_to_file_2, ...]
    my_mail.send(receiver,subject,text,image)
    
    """
    def __init__(self):
        self.port = 587
        self.server = 'smtp.gmail.com'
        try:
            self.smtpObj = smtplib.SMTP(self.server,self.port)	# Sets the mail server domain and the port it uses
        except Exception as e:
            print('The following error occurred when establishing a connection with the server:\n')
            print(e)
        
    def login(self):
        """ Establishes connection with smtp server and does the log in """
        
        self.username = 'javirozalen.code'
        self.password = getpass.getpass('Enter your email password: ')
        
        try:
            self.smtpObj.ehlo()	# Says "hello" to the SMTP email server
            self.smtpObj.starttls()	# Starts TLS encryption (this is only valid when working on port 587. For port 465 use SSL encryption)
            self.smtpObj.login(self.username+'@gmail.com',self.password)
        except Exception as e:
            print('The following went wrong when logging in:\n')
            print(e)
        
    def send(self,receiver,subject,text,files):
        """ Sends the message along with the attached files """
        self.receiver = receiver
        self.subject = subject
        self.message = text+'\n\nTess'
        self.files = files
        
        msg = MIMEMultipart()
        
        # Text 
        msg['From'] = self.username
        msg['To'] = self.receiver
        msg['Date'] = formatdate(localtime=True)
        msg['Subject'] = self.subject
        msg.attach(MIMEText(self.message))
    
        # Files
        if len(self.files) > 0 :
            for filename in self.files:
                if filename != '':
                    file = open(filename,'rb')
                    file_data = file.read()
                    try:
                        real_file = MIMEImage(file_data,name=filename)
                    except TypeError:
                        pass
                    try:
                        real_file = MIMEAudio(file_data,name=filename)
                    except TypeError:
                        pass
                    try:
                        real_file = MIMEText(file_data,name=filename)
                    except TypeError:
                        pass
                    try:
                        real_file = MIMEApplication(file_data,name=filename)
                    except TypeError:
                        print('File is not an application. No more filetypes known, sorry')
                    msg.attach(real_file)      
                    file.close()
            
        # Sending the email
        try:
            self.smtpObj.sendmail(self.username,self.receiver,msg.as_string())
        except Exception as e:
            print('When delivering the message, this went wrong:\n')
            print(e)
            
        self.smtpObj.quit()
    



