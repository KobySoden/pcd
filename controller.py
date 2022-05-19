import sys
from vidDownloader import *
import subprocess

# Include the database directory
sys.path.insert(1, 'database')
from database.pirate_data import *

<<<<<<< HEAD
#anaconda environment for running pcd
CONDA_ENV = "Tensor_CV_Fauna"

=======
>>>>>>> 0a3df7ad052cb7a30b7844df8d82d2827c4136ce
if __name__ == "__main__":
    # Create Instance to the PirateData in Fauna Database
    pirate_data = PirateData()

    #get all records in the db
    records = pirate_data.get_all_url_records()

    #loop through records
    for key, record in records.items():
<<<<<<< HEAD
        print("checking database entry", key)
=======
>>>>>>> 0a3df7ad052cb7a30b7844df8d82d2827c4136ce
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
<<<<<<< HEAD
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
                
=======

            #TODO check every file for piracy
            for suspect_file in os.listdir("videos"):
                #loop through all the videos we want to compare the downloaded files with
                pirate = "videos/"+suspect_file
>>>>>>> 0a3df7ad052cb7a30b7844df8d82d2827c4136ce
                for legit_video in os.listdir("originals"):
                    print("checking similarity of:", legit_video, " and ", suspect_file)
                    original = "originals/" + legit_video

                    #this command starts both videos at their first frame and goes to the end of pirate
                    input = " python ./app.py 1 1 -1 " + original + " " + pirate
<<<<<<< HEAD
                    
                    #call pcd here
                    os.system("conda activate " + CONDA_ENV + " && " + input)
=======
                    #call pcd here
                    os.system("conda activate Tensor_CV_Fauna && " + input)
>>>>>>> 0a3df7ad052cb7a30b7844df8d82d2827c4136ce

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

<<<<<<< HEAD
                #delete all the downloaded files
                os.remove(os.path.join("videos", suspect_file))
                #reset links
                links = None
    print("Successfully chacked database for piracy")
=======
                    #delete all the downloaded files
                    os.remove(os.path.join("videos", suspect_file))
                    #reset links
                    links = None
>>>>>>> 0a3df7ad052cb7a30b7844df8d82d2827c4136ce

