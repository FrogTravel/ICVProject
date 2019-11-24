import face_alignment
from skimage import io
import pickle
import os
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
directory = '/root/data/LS3D-W/Menpo-3D'
preds=[]
for file in os.listdir(directory):
    if file.endswith('.jpg') or file.endswith('.png'):
        input = io.imread(os.path.join(directory, file))
        pred = fa.get_landmarks(input)
        preds.append(pred)