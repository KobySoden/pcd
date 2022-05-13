import sys
from vidDownloader import *

# Include the database directory
sys.path.insert(1, 'database')
from database.pirate_data import *



if __name__ == "__main__":
    # Create Instance to the PirateData in Fauna Database
    pirate_data = PirateData()

    #get all records in the db
    records = pirate_data.get_all_url_records()

    print(records)
    #loop through records
    
    for key, record in records.items():
        try:
            result = pirate_data.get_pcd_last(
                url_key=key
            )
            timestamp = datetime.fromtimestamp(result).strftime("%c")
            #check if we have ever scanned for pirated content
            #if result == 0:
                #print(key)
            if datetime.fromtimestamp(result) < datetime.today():
                print(key)
                links = get_video_links(key)
                if links != None: download_video_series(links)
                
                #TODO check every file for piracy
                for file in os.listdir("videos"):
                    video = "videos/"+file
                    input = "python app.py 1 1 -1 " + video + " videos/original.mp4"

                    print("checking for pirated content")
                    piracy = os.system(input)
                    print(piracy)
                    if piracy == "9000":
                        print("Piracy Detected")
                    else:
                        print("No piracy found")

            #print(f"\"{key}\" last PCD timestamp: {timestamp}")
        except FaunaDB.FaunaDBException as exception:
            print(exception)

        