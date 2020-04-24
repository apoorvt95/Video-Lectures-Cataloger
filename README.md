# Video Lectures Cataloger
Video Lectures Cataloger is an android built as a requirement for course CSE 535 Mobile Computing at Arizona State University. This android application automatically scans a mobile device's storage and categorizes video lectures according to the context. The app first transcribes the video into textual information and then performs NLP to ascertain the subject of the video. 

## Please Follow the instructions below to setup the project
### Server
1) Copy the FogServer folder in your system.
2) Install Node and Python 3.6.
3) Open a terminal inside the FogServer directory and run 'npm install' command.
4) Run 'pip install -r requirements.txt' to install all python dependencies.
5) OPTIONAL: If you are using windows replace python3 as python in line#63
6) Run 'node server.js' command, the server should be up and running.
You can also retrain the classifier model for more labels.

# Android App
1) Install Android Studio
2) Import the MobileComputingProject
3) In Screen2 Activity, enter the IP of your local machine in the HTTP send function. 
3) Build the project, an android apk should be built.
4) An apk is already included in the repository. 
5) Install the APK on the phone. 
6) Give the required permissions.


## Running the Project
1) Once the APK is installed, try uploading a video lecture of subjects related to Computer Science, Physics & Geography.
2) Sample videos are included under folder SampleVideos.

## Design
Below is the overall design of the system.
![System Design](/SystemDesign/sysdesign.png "System Design")