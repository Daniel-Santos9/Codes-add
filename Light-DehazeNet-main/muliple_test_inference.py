from inference import image_haze_removel
from PIL import Image
import torchvision
import os
import argparse
import cv2

import time

video = "dataset-completo"
pasta_imagem = "malignant"

def multiple_dehaze_test(directory):


    print(directory)
    images = []
    filenames_list = [] #eu
    for filename in os.listdir(directory):
        gray =  Image.open(os.path.join(directory,filename))
        img = gray.convert("RGB")
        if img is not None:
            filenames_list.append(filename) #eu
            images.append(img)
    

#data_folder = "E:/Light-DehazeNet Implementation/query hazy images/outdoor natural/"

    
    print("Total de imagens: " + str(len(images)))
    c=0
    qtd=len(images)
    localarq = video + "/" + pasta_imagem
    src = "dataset_result/" + localarq
    #os.mkdir(src)
    for i in range(len(images)):
        img = images[i]
        pasta = src+"/"+filenames_list[i] #eu
        dehaze_image = image_haze_removel(img)
        torchvision.utils.save_image(dehaze_image, pasta) #eu
        #torchvision.utils.save_image(dehaze_image, "vis_results/dehaze_img("+str(c+1)+").jpg")
        c=c+1
        pct = ((c/qtd)*100)
        print(str(pct)+"%"+" concluído...")
        print(str(qtd-c)+" imagens restantes...")

    
if __name__ == "__main__":


    ap = argparse.ArgumentParser()

    texto_tempo = "tempo_"+ pasta_imagem

    default_end = "query_hazy_images/" + video + "/" + pasta_imagem + "/"
    ap.add_argument("-td", "--test_directory", default=default_end, required=False, help="path to test images directory")
    args = vars(ap.parse_args())



    inicio = time.time()
    
    multiple_dehaze_test(args["test_directory"])
    fim = time.time()
    t_ex = fim - inicio
    arquivo = open("dataset_result/" + video+ "/"+texto_tempo,'w')
    arquivo.write(str(t_ex))
    arquivo.close()
