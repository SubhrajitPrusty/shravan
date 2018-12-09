import geocoder
g = geocoder.ip('me')
lat, lon = g.latlng
print(lat, lon)
from weather import Weather, Unit

weather = Weather(unit=Unit.CELSIUS)

weather = Weather(Unit.CELSIUS)
lookup = weather.lookup_by_latlng(lat, lon)
condition = lookup.condition
print(condition.text)
