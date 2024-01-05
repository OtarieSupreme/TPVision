# TPVision Polytech Nancy 2024
## But : 
Détecte les objets sur une vidéo avec YoloV8 et mesure leur distance par rapport à un damier positionné dans la vidéo.

## Utilisation :
### Calibration de la caméra utilisée (source : https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
La caméra utilisée doit avoir une focale fixe.
Pour calibrer la caméra, il faut prendre des photos d'un damier de calibration (style jeu d'échecs) sous différents angles.
Utiliser le fichier `calibration.py` pour calculer les coefficients de distorsions. Ceux ci seront enregistrés dans un fichier `calibration.yaml`.
Le chemin et le format des photos ainsi que le nombre de points sur le damier peuvent être modifiés au début du fichier `calibration.py`.

### Utilisation du programme :
Ajouter la vidéo à la racine. 
Modifier les différents paramètres dans le fichier `main.py`.
Lancer `main.py`.
La première étape consiste à déplacer et à changer l'échelle de la vue de dessus (Bird eye view) pour que le contenu rentre dans la fenêtre.
Appuyer sur `Entrée` pour lancer la détection et les mesures

## Dépendances 
- Numpy
- OpenCV
- Ultralytics
- Glob
## Sources :
- https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- https://github.com/ultralytics/ultralytics
