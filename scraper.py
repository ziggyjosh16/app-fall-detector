import os
import requests
from bs4 import BeautifulSoup


def getimages(searchname, filedir, num):
    dataurl = 'https://www.google.com/search?ei=IKF8XOX8J9uAr7wPxKaj-Ag&yv=3&q={}&tbm=isch&vet=10ahUKEwjljPKMy-fgAhVbwIsBHUTTCI8QuT0IdygB.IKF8XOX8J9uAr7wPxKaj-Ag.i&ved=0ahUKEwjljPKMy-fgAhVbwIsBHUTTCI8QuT0IdygB&ijn=1&start={}&asearch=ichunk&async=_id:rg_s,_pms:s,_fmt:pc'
    count = 1
    try:
        os.makedirs(filedir)
    except BaseException:
        print("Directory Already Exists.")
    for i in range(0, num):
        res = requests.get(dataurl.format(searchname, i * 100))
        soup = BeautifulSoup(res.text, "lxml")
        for ele in soup.select('img'):
            imgurl = ele.get('data-src') or ele.get('src')
            with open(filedir + "\\" + str(count) + '.jpg', 'wb') as f:
                res2 = requests.get(imgurl)
                f.write(res2.content)
                f.close()
            count = count + 1


print("Working.")
getimages("person sitting in chair", "photos\\sitting", 200)
print("Done.")
