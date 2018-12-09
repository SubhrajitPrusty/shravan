import smtplib
import geocoder
g = geocoder.ip('me')

res = ""

import reverse_geocoder as rg

def reverseGeocode(coordinates):
	result = rg.search(coordinates)

	return result[0]['name']

if __name__=="__main__":

	coordinates = (g.latlng)

	res = reverseGeocode(coordinates)

	gmail_user = "tsibasish@gmail.com"
	gmail_pwd = "bhubanes"
	TO = 'subhrajit1997@gmail.com'
	SUBJECT = "EMERGENCY! HELP!"
	TEXT = "Hey! I have run into some problems. Please pick me up. I'm at " + res
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.ehlo()
	server.starttls()
	server.login(gmail_user, gmail_pwd)
	BODY = '\r\n'.join(['To: %s' % TO,
			'From: %s' % gmail_user,
			'Subject: %s' % SUBJECT,
			'', TEXT])

	server.sendmail(gmail_user, [TO], BODY)
	print ('email sent')
