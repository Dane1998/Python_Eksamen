# Python_Eksamen - Emoji Kamera

#### Projektet går ud på at træne et netværk som via dit webcam, vil kunne genkende ansigtsudtryk. Der skal hertil generes en emoji som matcher det udtryk den opfanger.      
#### Følgene teknologier er brugt:   
*  Keras
*  OpenCV
*  NumPy  
*  Pillow  
*  Tkinter  
*  Threading 

### Installations Guide 
#### Vi har brugt python 3.9 og pip til at installere vores libraries.  
#### Du skal åbne Command Promt som administator og kan derefter taste: 
#### pip install Keras  
#### pip install opencv-python  
#### pip install Numpy 
#### pip install Pillow  
#### pip install tk 
#### pip install tersorflow 
#### Du har nu installeret alle de libraries der kræver for at køre programmet. 
#### Programmet kræver at du har et webcam tilsluttet til din computer. 

#### Du skal i emoji_cam.py filen på linje 56, give den stien til din egen haarcascade_frontalface_default.xml fil. 
#### Eksembel kunne dette være stien
(C:/Users/danie/AppData/Local/Programs/Python/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml) 
#### For at køre programmet cd'er du dig vej hen til der hvor projekter er gemt, stiller dig i projekt mappen og skriver "emoji_cam.py"
  
  
### Status for projektet 
#### Programmet kører som det skal og vi kom i mål med vores ide. Programmet er dog ikke deployet til en server. 
#### Kameraret "plejer" at være dygtig til at genkende dit ansigtsudtryk, og den er hurtig til at finde den passende emoji.
