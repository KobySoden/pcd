import requests
import sys
from bs4 import BeautifulSoup
import validators

#download location
path = "videos/"

#timeout value for requests
TIMEOUT = 5

def get_video_links(archive_url):
  try:
    #create response object
    r = requests.get(archive_url, timeout=TIMEOUT)
  except:
    return None
  #create beautiful-soup object
  soup = BeautifulSoup(r.content,'html5lib')
  #find all links on web-page
  links = soup.find_all()
  
  #print all the links
  print("All Links on the site")
  for link in links:
      #print(link.prettify())
      if link.get('href') != None and validators.url(link.get('href')):
        #print(link.keys)
        print(link.get('href'))
  
  #filter the link ending with .mp4
  video_links = [link.get('src') for link in links if link.get('src') != None and link.get('src').endswith('mp4')]
  for link in links:
    if link.get('href') != None and link.get('href').endswith('mp4'):
        video_links.append(archive_url+link.get('href'))
        print("Found vid from href")
  return video_links

def download_video_series(video_links):
  i = 0
  for link in video_links:
    
    # iterate through all links in video_links
    # and download them one by one
    #obtain filename by splitting url and getting last string
    file_name = link.split('/')[-1]  
 
    print ("Downloading file:%s"%file_name)
 
    #create response object
    r = requests.get(link, stream = True, timeout=TIMEOUT)
 
    #download started
    with open(path+file_name, 'wb') as f:
      for chunk in r.iter_content(chunk_size = 1024*1024):
        if chunk:
          f.write(chunk)
    #increment number of videos downloaded
    i += 1

    print ("%s downloaded!\n"%file_name)
  return
 
if __name__ == "__main__":
  # specify the URL of the archive here
  archive_url = sys.argv[1]
  
  #getting all video links
  video_links = get_video_links(archive_url)
  
  #print video links
  print(video_links)
  #download all videos
  download_video_series(video_links)
