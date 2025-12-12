import smtplib
from email.message import EmailMessage
import os

def send_email(manager_email, suit_id, image_path):
    EMAIL_ADDRESS = 'harshrameshnerkar@gmail.com'
    EMAIL_PASSWORD = 'xvtf gdnk zjnd iuds'  # App Password (NOT real Gmail password)

    msg = EmailMessage()
    msg['Subject'] = f'🚨 Helmet Violation - Suit #{suit_id}'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = manager_email
    msg.set_content(f"A violation has been detected.\n\nSuit ID: {suit_id}")

    with open(image_path, 'rb') as f:
        img_data = f.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

    print("[INFO] Sending Email...")
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)
    print("[INFO] Email Sent ✅")
