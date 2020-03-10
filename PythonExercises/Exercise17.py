import requests
from bs4 import BeautifulSoup
url = 'https://www.nytimes.com/'

r = requests.get(url)
r_html = r.text
soup = BeautifulSoup(r_html, 'lxml')
for title in soup.findAll('h2'):
    print(title.text)