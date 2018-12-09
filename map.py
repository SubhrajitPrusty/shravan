import requests
import json

def getCoord():
    LINK = "http://10.42.0.1:5000/send"
    r = requests.get(LINK)
    print(r.json())

    return r.json()

def getAddr(lat, lon):
    LINK = "https://nominatim.openstreetmap.org/reverse?format=jsonv2&zoom=18"
    r = requests.get(LINK, params={"lat": lat, "lon" : lon})

    if r.status_code == 200:
        return r.json()
    else:
        return r.text


# print(getAddr(12.9988,77.6251))
# getCoord()
