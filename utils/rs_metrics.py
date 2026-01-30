import cv2
import numpy as np
from utils.fsc import FSC


def adjust_gamma(image, gamma=1.0):
    invgamma = 1.0 / gamma
    return np.array(np.power(image / 255.0, invgamma) * 255, dtype=np.uint8)

def drawMatches(img0, img1, kp0, kp1, path, color='green'):
    import matplotlib.pyplot as plt
    kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
    kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
    matches = [cv2.DMatch(i, i, 1) for i in range(len(kp0))]
    matchColor = (0, 255, 0) if color == 'green' else (255, 0, 0)
    show = cv2.drawMatches(img1, kp1, img0, kp0, matches, None, matchColor=matchColor)
    plt.imshow(show)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.gca().set_frame_on(False)
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()

def compute_rmse_and_inliers(H, matchedPoints1, matchedPoints2, RMSE_MAX_LIMIT):
    matchedPoints1_hom = np.vstack([matchedPoints1.T, np.ones(matchedPoints1.shape[0])])

    Y_ = np.dot(H, matchedPoints1_hom)

    Y_[0, :] /= Y_[2, :]
    Y_[1, :] /= Y_[2, :]

    E = np.sqrt(np.sum((Y_[:2, :] - matchedPoints2.T) ** 2, axis=0))

    inliersIndex = E < 3
    cleanedPoints1 = matchedPoints1[inliersIndex, :]
    cleanedPoints2 = matchedPoints2[inliersIndex, :]

    cleanedPoints2, IA = np.unique(cleanedPoints2, axis=0, return_index=True)
    cleanedPoints1 = cleanedPoints1[IA, :]

    cleanedPoints = np.hstack([cleanedPoints1, cleanedPoints2])
    cleanedPoints = cleanedPoints.astype(float)

    cleanedPoints1_hom = np.vstack([cleanedPoints[:, :2].T, np.ones(cleanedPoints.shape[0])])
    Y_ = np.dot(H, cleanedPoints1_hom)

    Y_[0, :] /= Y_[2, :]
    Y_[1, :] /= Y_[2, :]

    E = np.sqrt(np.sum((Y_[:2, :] - cleanedPoints[:, 2:4].T) ** 2, axis=0))

    if len(E) < 3:
        rmse = RMSE_MAX_LIMIT
    else:
        rmse = np.sqrt(np.sum(E ** 2) / E.size)

    return rmse, cleanedPoints1, cleanedPoints2



def compute_rs_metrics(batch, RMSE_MAX_LIMIT = 5, RMSE_SUCCESS = 3, showFlag = False):
    #-----------------------------<<<<<<<<<<<<
    init_matchedPoints1 = batch['mkpts0_f'].cpu().numpy()
    init_matchedPoints2 = batch['mkpts1_f'].cpu().numpy()
    H = batch['H_0to1'][0, :, :].cpu().numpy()

    if len(init_matchedPoints1) >= 3:
        _, cccccrmse, out_matchedPoints1, out_matchedPoints2 = FSC(init_matchedPoints1, init_matchedPoints2, 'affine', 5, 'Point')
        i_NOM = len(out_matchedPoints1)

        # NCM metric ------------
        i_RMSE, out_cleanedPoints1, out_cleanedPoints2 = compute_rmse_and_inliers(H, out_matchedPoints1, out_matchedPoints2, RMSE_MAX_LIMIT)
        i_NCM = len(out_cleanedPoints1)

        if i_RMSE > RMSE_MAX_LIMIT:
            i_RMSE = RMSE_MAX_LIMIT

        if i_RMSE < RMSE_SUCCESS:
            i_Success = 1
        else:
            i_Success = 0
        # -------------------------

        # Precision metric ------------
        if i_NOM == 0:
            i_Precision = 0
        else:
            i_Precision = i_NCM / i_NOM
        # -------------------------

        # Recall metric - -------------
        _, init_cleanedPoints1, init_cleanedPoints2 = compute_rmse_and_inliers(H, init_matchedPoints1,
                                                                            init_matchedPoints2, RMSE_MAX_LIMIT)
        
        i_NC = len(init_cleanedPoints1)
        if i_NC == 0:
            i_Recall = 0
        else:
            i_Recall = i_NCM / i_NC
        
    else:
        i_Success = 0
        i_RMSE = RMSE_MAX_LIMIT
        i_NCM = 0
        i_Recall = 0
        i_Precision = 0

    return {'i_Success': i_Success, 'i_NCM':i_NCM, 'i_RMSE': i_RMSE, 'i_Precision': i_Precision, 'i_Recall': i_Recall}