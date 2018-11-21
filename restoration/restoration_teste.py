import numpy as np

class Restoration:

    def run_restoration(self, img):
        img_temp = np.copy(img)
        img_temp[0:100] = 0
        return img_temp
