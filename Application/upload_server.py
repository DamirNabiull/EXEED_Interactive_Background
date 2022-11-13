from cryptography.fernet import Fernet
from aiohttp import web
import json
import yadisk
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib as smtp

y_disk: yadisk.YaDisk
email: str
smtp_pass: str
token: str

routes = web.RouteTableDef()
@routes.post("/uploadAndSend")
async def post_upload(request):
    global y_disk, email, smtp_pass

    data = await request.json()
    final_video, to_send_email, current_id = data['final_video'], data['to_send_email'], data['current_id']

    print('************* UPLOAD VIDEO *************')
    dir = config['disk_dir']
    dest_path = f'/{dir}/{current_id}.mp4'
    print('UPLOAD')
    try:
        y_disk.upload(final_video, dest_path, timeout=10)
    except Exception as e:
        print(e)
    print('UPLOADED')
    video_url = y_disk.get_download_link(dest_path)
    print('GET LINK')
    print(video_url)

    print('Send started')

    if not re.match(r"[^@]+@[^@]+\.[^@]+", to_send_email):
        pass
    else:
        server = smtp.SMTP_SSL('smtp.yandex.com', 465)
        server.set_debuglevel(1)
        server.ehlo(email)
        server.login(email, smtp_pass)
        server.auth_plain()
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "Exeed Video"
        msg['From'] = email
        msg['To'] = to_send_email
        msg_text = MIMEText(f'''<p>
Спасибо, что посетили шоурум EXEED! Скачать ваше видео и поделиться в социальных сетях вы сможете по <a href="{video_url}"><u>ссылке</u></a>
<br><br>
Надеемся, что вам понравилось время, которое вы провели с нами. Еще больше информации о модельном ряде, комплектациях и дополнительных сервисах вы сможете узнать на сайте <a href="https://exeed.ru/">exeed.ru</a>
<br><br>
Найти дилера в своем городе и узнать больше о автомобилях в наличии и ценах:<br>
<a href="https://exeed.ru/dealers/">exeed.ru/dealers</a>
<br><br>
С наилучшими пожеланиями,<br>
команда EXEED Россия
</p>''', 'html')
        # part1 = MIMEText(video_url, 'plain')
        msg.attach(msg_text)

        server.sendmail(email, to_send_email, msg.as_string())
        server.quit()
    print('Send ended')

    data = {'status': 200}
    return web.json_response(data=data)


if __name__ == '__main__':
    config = json.load(open('config.json'))
    key = b'-HzoIZjl0UaJfzRdTW-LD-ik9q96yvEioeH9RME1XHs='
    cipher_suite = Fernet(key)
    smtp_pass = cipher_suite.decrypt(config['smtp_pass'].encode()).decode()
    token = cipher_suite.decrypt(config['token'].encode()).decode()
    email = config['login']

    y_disk = yadisk.YaDisk(token=token)
    print(y_disk.check_token())

    app = web.Application()
    app.add_routes(routes)
    web.run_app(app, host='0.0.0.0', port=8080)