import os
import cv2

A = "features/fullFrame-210x260px"   #Input
B = "features/fullFrame-227x227px"  #Output

def resize(src_path, out):
    COUNT = 0
    for dir_1 in sorted(os.listdir(src_path)):
        d_2 = os.path.join(src_path,dir_1)
        # print(dir_1)
        for dir_2 in sorted(os.listdir(d_2)):
            d_3 = os.path.join(d_2,dir_2)
            # print(d_3)
            for dir_3 in sorted(os.listdir(d_3)):
                # print(os.path.join(d_3,dir_3))
                image_path = os.path.join(d_3,dir_3)
                image = cv2.resize(cv2.imread(image_path), (227, 227),
                           interpolation=cv2.INTER_CUBIC)
                i_folder = os.path.join(out,dir_1,dir_2)
                if not os.path.exists(i_folder):
                    os.makedirs(i_folder)
                trg_path = os.path.join(out,dir_1,dir_2,dir_3)
                cv2.imwrite(trg_path,image)
                COUNT+=1
                print(COUNT)


if __name__ == '__main__':
    resize(A, B)