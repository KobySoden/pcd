import sys
from vidDownloader import *
import subprocess

# Include the database directory
sys.path.insert(1, 'database')
from database.pirate_data import *

if __name__ == "__main__":
    # Create Instance to the PirateData in Fauna Database
    pirate_data = PirateData()

    #get all records in the db
    records = pirate_data.get_all_url_records()

    #loop through records
    for key, record in records.items():
        try:
            result = pirate_data.get_pcd_last(
                url_key=key
            )
        except FaunaDB.FaunaDBException as exception:
            print(exception)
            exit("FAUNA FAILED")
        
        timestamp = datetime.fromtimestamp(result).strftime("%c")

        #check if we have scanned for pirated content today
        if datetime.fromtimestamp(result) < datetime.today():
            links = get_video_links(key)
            if links != None: download_video_series(links)
            
            #TODO check every file for piracy
            for file in os.listdir("videos"):
                print("checking for pirated content in: ", file)
                
                video = "videos/"+file
                input = " python ./app.py 1 1 -1 " + video + " videos/original.mp4"
                
                #call pcd here
                os.system(input)
                
                #read pcd output from file
                f = open("piracy.txt", "r")
                piracy = f.read()
                f.close()

                if piracy == "9000":
                    print("Piracy Detected")
                else:
                    print("No piracy found")

        