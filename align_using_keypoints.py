# import the necessary packages
#from .helpers import FACIAL_LANDMARKS_IDXS
#from .helpers import shape_to_np
import dutils
dutils.init()
import numpy as np
import cv2
import collections
#class FaceAligner:
#    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
#        desiredFaceWidth=256, desiredFaceHeight=None):
#        # store the facial landmark predictor, desired output left
#        # eye position, and desired output face width + height
#        self.predictor = predictor
#        self.desiredLeftEye = desiredLeftEye
#        self.desiredFaceWidth = desiredFaceWidth
#        self.desiredFaceHeight = desiredFaceHeight
#        # if the desired face height is None, set it to be the
#        # desired face width (normal behavior)
#        if self.desiredFaceHeight is None:
#            self.desiredFaceHeight = self.desiredFaceWidth

pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }
#class FaceAligner:
def align(im,landmarks, desiredLeftEye=(0.35, 0.35),
    desiredFaceWidth=256, desiredFaceHeight=None):
    # store the facial landmark predictor, desired output left
    # eye position, and desired output face width + height
    # self.desiredLeftEye = desiredLeftEye
    # self.desiredFaceWidth = desiredFaceWidth
    # self.desiredFaceHeight = desiredFaceHeight
    # if the desired face height is None, set it to be the
    # desired face width (normal behavior)
    #if self.desiredFaceHeight is None:
    #    self.desiredFaceHeight = self.desiredFaceWidth
    if desiredFaceHeight is None:
        desiredFaceHeight = desiredFaceWidth

    '''
    for eye in ['eye1','eye2']:
        #locals()[f'{eye}_mean_x'] = np.mean(landmarks[pred_types[eye].slice,0])
        #locals()[f'{eye}_mean_y'] = np.mean(landmarks[pred_types[eye].slice,1])
        exec(f'{eye}_mean_x = np.mean(landmarks[pred_types[eye].slice,0])',locals())
        exec(f'{eye}_mean_y = np.mean(landmarks[pred_types[eye].slice,1])',locals())
    '''
    eye1_mean_x = np.mean(landmarks[pred_types['eye1'].slice,0])
    eye1_mean_y = np.mean(landmarks[pred_types['eye1'].slice,1])
    eye2_mean_x = np.mean(landmarks[pred_types['eye2'].slice,0])
    eye2_mean_y = np.mean(landmarks[pred_types['eye2'].slice,1])

    dY = eye2_mean_y - eye1_mean_y
    dX = eye2_mean_x - eye1_mean_x
    angle = np.degrees(np.arctan2(dY,dX)) #- 180

    desiredRightEyeX = 1.0 - desiredLeftEye[0]
    dist = np.sqrt((dX ** 2) + (dY ** 2))

    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    eyesCenter = ((eye1_mean_x + eye2_mean_x) // 2,
        (eye1_mean_y + eye2_mean_y) // 2)
    M = cv2.getRotationMatrix2D(eyesCenter,angle,scale)

    tX = desiredFaceWidth *0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0,2] += (tX - eyesCenter[0])
    M[1,2] += (tY - eyesCenter[1])

    w,h = (desiredFaceWidth, desiredFaceHeight)
    output = cv2.warpAffine(im, M, (w, h),
        flags=cv2.INTER_CUBIC)
    return output
if __name__ == '__main__':
    import face_alignment
    from skimage import io

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    im = io.imread('AsouYumi.jpg')
    landmarks, = fa.get_landmarks(im)
#==================================
    plt.figure()
    plt.imshow(im)
    for landmark in landmarks:
        plt.scatter(landmark[0],landmark[1])
    plt.draw()
    plt.savefig('im1_landmarks.png')
#==================================
    aligned = align(im,landmarks, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None)

    im2 = io.imread('GettyImages-1495234870.jpg')
    landmarks2, = fa.get_landmarks(im2)
    aligned2 = align(im2,landmarks2, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None)
    dutils.img_save(im[:,:,0],'im.png')
    dutils.img_save(im2[:,:,0],'im2.png')
    dutils.img_save(aligned[:,:,0],'aligned.png')
    dutils.img_save(aligned2[:,:,0],'aligned2.png')
    p47()
