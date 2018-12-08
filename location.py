import android
import time
from urllib import request

droid = android.Android()
droid.startLocating()

while True:
    try:
        event = droid.eventWaitFor('location', 1000)
        provider = event.result['data']['gps']['provider']
        if provider == 'gps':
            lat = str(event.result['data']['gps']['latitude'])
            lon = str(event.result['data']['gps']['longitude'])

            print(lat, lon, end="")
            print("\r")
            req = request.urlopen("http://10.42.0.1:5000/?lat={}&lon={}".format(lat,lon))
        else:
            print("cannot access gps", end="")
            print("\r")
    except Exception as e:
        print(e, end="")
        print("\r")
