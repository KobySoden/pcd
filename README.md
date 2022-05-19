# pcd

This project is a subsection of a larger project but it can operate on its own. By downloading this repository you will have a fully functioning pirated content detector. It must be connected to a FaunaDB database to run on a website. Otherwise it will work with two videos. 

Usage: 
       conda activate Tensor_CV_Fauna
       pcd w/ database: python controller.py 
       
       pcd: python app.py [start frame 1] [start frame 2] [last frame] [path1.mp4] [path2.mp4] 

       video downloads: python vidDownloader.py [www.example.com]
       

download files from google drive to run the project
https://drive.google.com/drive/folders/1Mtz0iB7cyjVbLAPZQdSokmbEAMT13w-D?usp=sharing

Setup: 
       Requires an [anconda environment](https://www.anaconda.com/) running python 3.6.x  
       
       Name this anaconda environment "Tensor_CV_Fauna"
       
       Install requirements in the anaconda environment with
       python -m pip install -r requirements.txt
       
       All videos being searched for go into the originals folder.
       Everything else will be downloaded into the videos folder
       
       Connect to the database using a .env file containing FAUNA_SECRET_KEY=<yourkey>
