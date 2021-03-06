import sys
from vidDownloader import *
import subprocess

# Include the database directory
sys.path.insert(1, 'database')
from database.pirate_data import *

#anaconda environment for running pcd
CONDA_ENV = "Tensor_CV_Fauna"

if __name__ == "__main__":
    # Create Instance to the PirateData in Fauna Database
    pirate_data = PirateData()

    #get all records in the db
    records = pirate_data.get_all_url_records()

    #loop through records
    for key, record in records.items():
        print("checking database entry", key)
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
            else:
                continue

            print(len(os.listdir("videos")))
            #no videos downloaded go onto next site
            if len(os.listdir("videos")) == 0:
                continue
            
            print("test")
            for suspect_file in os.listdir("videos"):
                print(suspect_file)
                #loop through all the videos we want to compare the downloaded files with
                pirate = "videos/"+suspect_file

                for legit_video in os.listdir("originals"):
                    print("checking similarity of:", legit_video, " and ", suspect_file)
                    original = "originals/" + legit_video

                    #this command starts both videos at their first frame and goes to the end of pirate
                    input = " python ./app.py 1 1 -1 " + original + " " + pirate
    
                    #call pcd here
                    os.system("conda activate " + CONDA_ENV + " && " + input)

                    #read pcd output from file
                    f = open("piracy.txt", "r")
                    piracy = f.read()
                    f.close()

                    if piracy == "9000":
                        print("Piracy Detected")
                        pirate_data.set_pirated_content_boolean(key, True)
                    else:
                        print("No piracy found")
                        pirate_data.set_pirated_content_boolean(key, False)
                    pirate_data.set_pcd_last(url_key=key, pcd_last=datetime.now().timestamp())

                #delete all the downloaded files
                os.remove(os.path.join("videos", suspect_file))
                #reset links
                links = None
    print("Successfully chacked database for piracy")
