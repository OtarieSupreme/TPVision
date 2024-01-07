import cv2
import numpy as np
import os, sys
import yaml
from ultralytics import YOLO
import time


# Permets de calculer la matrice de transformation pour la vue de dessus à partir de la position des points du damier dans l'image et dans le monde réel
def getTransformMatrix(threedpoints, twodpoints, image, rotation = None, translation = [0.5, 0.5], scale = 1.0):
    src = np.array(twodpoints, np.float32)
    dst = (np.array(threedpoints, np.float32)[:,0:2])*CHECKERBOARD_SIZE # En multipliant par CHECKERBOARD_SIZE, on obtient 1 pixel = 1 mm
    dst *= scale
    translation = np.array(translation)
    dst += np.multiply(translation, image.shape[0:2])
    distantPointIndex = [0, CHECKERBOARD[0]-1, CHECKERBOARD[0]*(CHECKERBOARD[1]-1), CHECKERBOARD[0]*CHECKERBOARD[1]-1] # On prend les 4 coins du damier (plus précis que de prendre 4 coins proches les uns des autres)
    M = cv2.getPerspectiveTransform(src[distantPointIndex],dst[distantPointIndex])

    return M

def birdEyeView(image, M):
    birdEye = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return birdEye


def undistort(img, param) :
    #undistorted_img = cv2.remap(img, param[0], param[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    # Utiliser cv2.remap est plus rapide qu'utiliser cv2.undistort mais nécessite d'utilser cv2.initUndistortRectifyMap avant 
    undistorted_img = cv2.remap(img, param[0], param[1], cv2.INTER_LINEAR) 

    # On rogne l'image pour enlever les bords noirs
    x, y, w, h = param[2]
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    
    return undistorted_img
    


#--------------------------------- Paramètres ---------------------------------------------------------------#
##############################################################################################################

CHECKERBOARD = (7, 10) # Nombre de points sur le damier (x, y)
CHECKERBOARD_SIZE = 25 # Taille des cases du damier en mm
scale = 1.0 # 1 pixel = 1 mm si scale = 1.0
translation = [0.5, 0.5] # Translation de l'image dans la vue de dessus (en % de la taille de l'image)
rotation = None # Rotation de l'image dans la vue de dessus (non implémenté)
imPath = './DamiersCalibration/IMG_20240102_170712.jpg' # Facultatif, si on ne veut pas utiliser de vidéo
vidPath = './video.mp4' # Vidéo à utiliser
calibration_file = "./calibration.yaml" # Fichier de calibration généré par le script de calibration

# Paramètres pour YOLO
yoloModel = "yolov8n.pt" # Modèle YOLO à utiliser
useClassFilter = False # Filtre ou non les classes détectées 
classFilter = ["person", "cup"] # Liste des classes à utiliser 
useConfFilter = True # Filtre ou non les objets par rapport à leur confiance 
minConfidence = 0.4 # Seuil de confiance pour la détection des objets 

##############################################################################################################






# (Détection du damier) Arrête l'itération quand la précision spécifiée est atteinte ou quand le nombre d'itérations spécifié est atteint
criteria = (cv2.TERM_CRITERIA_EPS +
			cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 

# Vecteur pour les points 3D dans le monde réel
threedpoints = [] 
# Vecteur pour les points 2D dans le plan de l'image.
twodpoints = [] 

# Grille de points 3D dans le monde réel que l'on va utiliser pour le damier
objectp3d = np.zeros((1, CHECKERBOARD[0] 
					* CHECKERBOARD[1], 
					3), np.float32) 
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
							0:CHECKERBOARD[1]].T.reshape(-1, 2) 





# On ouvre la vidéo pour avoir la première image (si la vidéo ne s'ouvre pas, on utilise l'image de test)
cap = cv2.VideoCapture(vidPath) 

if not cap.isOpened():
    print("Echec de l'ouverture de la video \n Utilisation de l'image de test")
    image = cv2.imread(imPath)
    useVideo = False
else :
    ret, frame = cap.read()
    if ret:
        image = frame
        useVideo = True
    
    else :
        print("Echec de la capture de la première image de la video \n Utilisation de l'image de test") 
        image = cv2.imread(imPath)
        useVideo = False
cap.release()
size = (image.shape[1], image.shape[0])

# cv2.imshow("Image retenue", image)
# print("Appuyer sur une touche pour continuer")
# cv2.waitKey(0)
# cv2.destroyAllWindows()

    



# Load the calibration parameters from calibration.yaml


# Récupération des paramètres de calibration de la caméra dans le fichier .yaml génére par le script de calibration
if os.path.isfile(calibration_file):
    with open(calibration_file, 'r') as f:
        loadeddict = yaml.safe_load(f)
        camera_matrix = np.asarray(loadeddict.get('camera_matrix'))
        dist_coeff = np.asarray(loadeddict.get('dist_coeff'))
else :
    print("Impossible de trouver le fichier de calibration")
    sys.exit(0)


newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, size, 1, size)
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeff, None, newcameramtx, size, 5) # Calcule les matrices de transformation pour la fonction cv2.remap
dist_params = (mapx, mapy, roi)




undistorted_image = undistort(image, dist_params) # On corrige la distorsion de l'image


# On cherche les coins du damier dans l'image qu'on utilisera pour connaître la position du sol par rapport à la caméra
# On utilisera aussi le damier comme référence pour calculer la distance des objets détectés
# Pour cela, on réutilise le code du script de calibration https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
grayColor = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY) 
ret, corners = cv2.findChessboardCorners(
                    grayColor, CHECKERBOARD, None) 
if ret == True: 
    threedpoints = objectp3d[0]
    corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria) 

    twodpoints[:] = corners2[:,0,:]
    points = cv2.drawChessboardCorners(undistorted_image,  
                                          CHECKERBOARD,  
                                          corners2, ret) 
else:
    print("Impossible de trouver les coins du damier")
    sys.exit(0)




#cv2.imshow("Image", undistorted_image)



M = getTransformMatrix(threedpoints, twodpoints, undistorted_image, rotation, translation, scale) # On calcule la matrice de transformation pour la vue de dessus


birdEye = birdEyeView(undistorted_image, M)
cv2.imshow("Bird eye view", birdEye)

print("Déplacer l'image avec ZQSD, changer l'échelle avec P et M, valider avec Entrée, quitter avec Echap")

# Cette partie du code permet de déplacer l'image dans la vue de dessus et de changer l'échelle
while True:
    key = cv2.waitKey(1)

    if key == ord('q'):
        translation[0] -= 0.01
    elif key == ord('s'):
        translation[1] += 0.01
    elif key == ord('d'):
        translation[0] += 0.01
    elif key == ord('z'):
        translation[1] -= 0.01
    elif key == ord('p'):
        scale += 0.1
    elif key == ord('m'):
        scale -= 0.1

    elif key == 27: # Echap pour quitter
        cv2.destroyAllWindows()
        print("Bye bye")
        sys.exit(0)

    elif key == 13: # Entrée pour valider
        cv2.destroyAllWindows()
        break

    if key == ord('z') or key == ord('q') or key == ord('s') or key == ord('d') or key == ord('p') or key == ord('m'): # Update the image
        #print("Translation: ", translation)
        #print("Scale: ", scale)
        M = getTransformMatrix(threedpoints, twodpoints, undistorted_image, rotation, translation, scale)
        birdEye = birdEyeView(undistorted_image, M)
        cv2.imshow("Bird eye view", birdEye)





# Partie vidéo
if not useVideo:
    print("Video non utilisée, fin du programme")
    sys.exit(0)
        
cap = cv2.VideoCapture(vidPath) 

if not cap.isOpened():
    print("Echec de l'ouverture de la video, fin du programme")
    exit()



print("Chargement du modèle YOLO V8 : ", yoloModel)
model = YOLO(yoloModel)

print("Traitement de la vidéo")
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Echec de la capture de la vidéo, fin du programme")
        break
    

    undistorted_image = undistort(frame, dist_params) # On corrige la distorsion de l'image

    birdEye = birdEyeView(undistorted_image, M) # On transforme l'image pour avoir la vue de dessus avec la matrice de transformation calculée précédemment
    results = model(undistorted_image, verbose=False) # On détecte les objets dans l'image avec YOLO

    annotated_frame = results[0].plot()
    
    # On utilise le premier point du damier comme origine du repère, on calculera la distance des objets par rapport à ce point
    referencePoint = np.array(twodpoints, dtype=int)[0]
    # Calcule la position du point de référence dans la vue de dessus en utilisant la même formule que warpPerspective()
    xref = (referencePoint[0]*M[0,0] + referencePoint[1]*M[0,1] + M[0,2])/(referencePoint[0]*M[2,0] + referencePoint[1]*M[2,1] + M[2,2])
    yref = (referencePoint[0]*M[1,0] + referencePoint[1]*M[1,1] + M[1,2])/(referencePoint[0]*M[2,0] + referencePoint[1]*M[2,1] + M[2,2])
    referencePoint = (int(xref), int(yref))
    cv2.circle(birdEye, referencePoint, 5, (0, 255, 0), -1)


    names = model.names
    for box in results[0].boxes:
        ignoredForClass = []
        name = names[int(box.cls)]
        if name in classFilter or not useClassFilter :
            ignoredForConf = []
            if box.conf > minConfidence or not useConfFilter :

                # On prend le point le plus bas du bounding box pour calculer la distance par rapport au damier
                boxPoint = (int(box.xyxy[0, 0]+ (box.xyxy[0, 2]-box.xyxy[0, 0])/2) , int(box.xyxy[0, 3]))
                # On calcule la position du point dans la vue de dessus en utilisant la même formule que warpPerspective()
                x = (boxPoint[0]*M[0,0] + boxPoint[1]*M[0,1] + M[0,2])/(boxPoint[0]*M[2,0] + boxPoint[1]*M[2,1] + M[2,2])
                y = (boxPoint[0]*M[1,0] + boxPoint[1]*M[1,1] + M[1,2])/(boxPoint[0]*M[2,0] + boxPoint[1]*M[2,1] + M[2,2])
                boxPoint = (int(x), int(y))

                cv2.circle(birdEye, boxPoint, 5, (255, 0, 0), -1)
                cv2.line(birdEye, boxPoint, referencePoint, (0, 0, 255), 2)
                pixelDifference = (boxPoint[0]-referencePoint[0], boxPoint[1]-referencePoint[1])

                # On peut maintenant calculer la distance en mm puisqu'on connaît la position des deux points sur l'image et l'échelle de l'image
                cv2.putText(birdEye, str(round(np.linalg.norm(pixelDifference)/scale, 2)) + " mm", (boxPoint[0]+10, boxPoint[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # Affiche la classe de l'objet
                cv2.putText(birdEye, name, (boxPoint[0]+10, boxPoint[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else :
                ignoredForConf.append(name)
        else :
            ignoredForClass.append(name)
    
    print("Ignorés pour la classe: ", ignoredForClass)
    print("Ignorés pour la confiance: ", ignoredForConf)
    print("FPS: ", 1.0 / (time.time() - start_time))

    cv2.imshow("Result", birdEye)

    if cv2.waitKey(1) == 27: # Echap pour quitter
        break

cap.release()
cv2.destroyAllWindows()






