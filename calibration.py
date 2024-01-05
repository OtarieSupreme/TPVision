import numpy as np
import cv2
import yaml
import glob

# Code de calibration réutilisé de : https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

#--------------------------------- Paramètres ---------------------------------------------------------------#
##############################################################################################################
CHECKERBOARD = (7, 10) # Nombre de points sur le damier (x, y)
imagesFolder = './DamiersCalibration/' # Dossier contenant les images de calibration
imageFormat = '.jpg' # Format des images de calibration
##############################################################################################################






criteria = (cv2.TERM_CRITERIA_EPS +
			cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 


threedpoints = [] 
twodpoints = [] 


objectp3d = np.zeros((1, CHECKERBOARD[0] 
					* CHECKERBOARD[1], 
					3), np.float32) 
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
							0:CHECKERBOARD[1]].T.reshape(-1, 2) 
prev_img_shape = None


images = glob.glob(imagesFolder + '*' + imageFormat)
image = cv2.imread(images[0]) 
h, w = image.shape[:2] 

for filename in images: 
    image = cv2.imread(filename) 
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
    ret, corners = cv2.findChessboardCorners(
                    grayColor, CHECKERBOARD, None) 

    if ret == True: 
        threedpoints.append(objectp3d) 
  
        corners2 = cv2.cornerSubPix( 
            grayColor, corners, (11, 11), (-1, -1), criteria) 
        twodpoints.append(corners2) 
  
        image = cv2.drawChessboardCorners(image,  
                                          CHECKERBOARD,  
                                          corners2, ret) 
    else:
        print("Impossible de trouver les coins du damier")
  
    cv2.waitKey(0)

cv2.destroyAllWindows() 




ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None) 

print(" Camera matrix:") 
print(matrix) 

print("\n Distortion coefficient:") 
print(distortion) 

print("\n Rotation Vectors:") 
print(r_vecs) 

print("\n Translation Vectors:") 
print(t_vecs) 




# Enregistre les parametres de calibration dans un fichier lisible avec openCV

data = {'camera_matrix': np.asarray(matrix).tolist(), 'dist_coeff': np.asarray(distortion).tolist()}

with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)

	
print("Calibration file saved")