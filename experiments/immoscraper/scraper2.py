import bs4 as bs
import urllib.request
import time
from datetime import datetime
import pandas as pd
import json
import os


PATH = os.path.dirname(os.path.abspath(__file__))
import http.cookiejar as cook
for seite in range(1457, 3000):

    print("Loop " + str(seite) + " startet.")

    df = pd.DataFrame()
    l = []

    try:

        req = urllib.request.Request("https://www.immobilienscout24.de/Suche/S-2/P-" + str(seite)+ "/Haus-Kauf" ,
                                     data=None, headers={
                'User-Agent': 'Mozilla/9.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'})
        #cookies=cook.FileCookieJar("C:/Users/ITOP_Seiter/AppData/Roaming/Mozilla/Firefox/Profiles/tqsat58v.default-release/cookies.sqlite")
        #cookieProc=urllib.request.HTTPCookieProcessor(cookiejar=cookies)
        opener=urllib.request.build_opener()#cookieProc)
        soup = bs.BeautifulSoup(
            opener.open(req),'lxml')
        print("Aktuelle Seite: " + "https://www.immobilienscout24.de/Suche/S-2/P-" + str(seite) + "/Haus-Kauf")
        #print(soup.find_all("a"))
        for paragraph in soup.find_all("a"):
            #print(str(paragraph.get("href")))
            if r"/expose/" in str(paragraph.get("href")):
                #print('test2222')
                l.append(paragraph.get("href").split("#")[0])

            l = list(set(l))
        print(l)

        for item in l:
            try:

                req = urllib.request.Request(
                    "https://www.immobilienscout24.de"+ item,
                    data=None, headers={
                        'User-Agent': 'Mozilla/9.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'})
                #cookies = cook.FileCookieJar(
                #    "C:/Users/ITOP_Seiter/Desktop/cookies.csv")
                    #:/Users/ITOP_Seiter/AppData/Roaming/Mozilla/Firefox/Profiles/tqsat58v.default-release/cookies.sqlite")
                #cookieProc = urllib.request.HTTPCookieProcessor(cookiejar=cookies)
                opener = urllib.request.build_opener()#cookieProc)
                soup = bs.BeautifulSoup(
                    opener.open(req).read(),'lxml')
                print(soup)

                data = pd.DataFrame(
                    json.loads(str(soup.find_all("script")).split("keyValues = ")[1].split("}")[0] + str("}")),
                    index=[str(datetime.now())])

                data["URL"] = str(item)

                beschreibung = []

                for i in soup.find_all("pre"):
                    beschreibung.append(i.text)

                data["beschreibung"] = str(beschreibung)

                df = df.append(data)
                print(beschreibung)

            except Exception as e:
                print(str(datetime.now()) + ": " + str(e))
                l = list(filter(lambda x: x != item, l))
                print("ID " + str(item) + " entfernt.")
                time.sleep(3660)
        print("Exportiert CSV")
        df.to_csv(PATH+"/rohdaten" + str(datetime.now())[:19].replace(":", "").replace(".", "") + ".csv", sep=";",
                  decimal=",", encoding="utf-8", index_label="timestamp")

        print("Loop " + str(seite) + " endet.\n")

    except Exception as e:
        print(str(datetime.now()) + ": " + str(e))

print("FERTIG!")