# Visual Odometry

## Generate the requird images form the given Oxford_data
First, change the directory to Code:  
```bash
cd Oxford_dataset  
```
and Create a folder /data add the file DataPreparation.py to the Oxford_data to run: 
```bash
python3 DataPreparation.py  
```
You will see the code running, where the images are prepared as per the requirement and generated in the /data folder.  



Running the code again will output dragon baby's video with the car being detected using the Lucas-Kanade tracker.  

## For running robust LK tracker
To Run the visual Odometry for the generated data in the /data folder. 

```bash
python3 visualOdometry.py  
````


 