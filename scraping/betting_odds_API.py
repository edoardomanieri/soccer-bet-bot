import requests

headers = {
	'apikey': 'b39e65f0-04d8-11ea-9c7b-7d4f77ee3e57',
}

params = (
	('sport', 'soccer'),
	('country', 'italy'),
	('league', 'soccer-italy-serie-a')
)

response = requests.get('https://app.oddsapi.io/api/v1/odds', headers=headers, params=params)

print(response.json())