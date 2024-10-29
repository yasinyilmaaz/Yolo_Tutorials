import cv2
import numpy as np
import os

path = os.path.join(os.getcwd(), "media",'mehmet.jpeg')
img = cv2.imread(path)
img = cv2.resize(img, (800,600))
# print(img)

img_width = img.shape[1]
img_height = img.shape[0]


img_blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB=True, crop=False)
# blovFromImage : resmi blob formatına çevirir. 4 boyutlu tensör oluşturur
# swapRB = True : gelen resmi RGB ye çevirir
# crop = False : resmi kırpma işlemi yapmaz
print(img_blob.shape)

# Bölüm 3: YOLOv3 Ağırlıklarını ve YOLOv3 Yapısını Yükleme
# labels etiketlerin isimlerini tutar
labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
         "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
         "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
         "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
         "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
         "bottle","wineglass","cup","fork","knife","spoon","bowl","banana","apple",
         "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
         "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
         "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
         "book","clock","vase","scissors","teddybear","hairdrier","toothbrush"]

# algılanan her nesneye ayrı bir renk vermek için renkler oluşturulur
colors = ["0,255,255","0,0,255","255,0,0","255,255,0","0,255,0"]
colors = [np.array(color.split(",")).astype("int") for color  in colors]
colors = np.array(colors)
colors = np.tile(colors, (18, 1)) # (diziyi x ve y ekseninde çoğaltır)

# YOLOv3 ağırlıklarını ve yapılarını yükleme
path_cfg = os.path.join(os.getcwd(),'yolov3.cfg')
path_weights = os.path.join(os.getcwd(),'yolov3.weights')
model = cv2.dnn.readNetFromDarknet(path_cfg,path_weights)
layers = model.getLayerNames() # modeldeki layerları alır
# print(layers) # layerların isimlerini yazdırır

output_layer = [layers[layer-1] for layer in model.getUnconnectedOutLayers()] # çıktı layerlarını alır
# model.getUnconnectedOutLayers() : modelin çıktıların hangi layerlardan alınacağını belirler


model.setInput(img_blob) # modelin girişine resmi verir

detection_layers = model.forward(output_layer) # çıktı layerlarını alır
# print(detection_layers) # çıktı layerlarının boyutunu yazdırır
# print(len(detection_layers))

################# NON-MAXIMUM SUPPRESSİON OPERATİON:1#####################
ids_list = []
boxes_list = []
confidences_list = []
################# end of operation #######################################

# Bölüm 4: Algılanan Nesneleri Görselleştirme
for detection_layer in detection_layers:
    for object_detection in detection_layer:

        scores = object_detection[5:] #güven skorlarını alır
        predicted_id = np.argmax(scores) # max degerli indexsi verir
        confidence = scores[predicted_id] # max degerli indexin skoru

        if confidence > 0.30:
            label = labels[predicted_id]
            bounding_box = object_detection[:4] * np.array([img_width,img_height,img_width,img_height]) # bounding box koordinatlarını alır
            (box_center_x, box_center_y, box_width, box_height) =  bounding_box.astype("int")

            start_x = int(box_center_x - (box_width/2))
            start_y = int(box_center_y - (box_height/2))

            ################# NON-MAXIMUM SUPPRESSİON OPERATİON:2#####################
            ids_list.append(predicted_id) # algılanan nesnenin id sini listeye ekler
            confidences_list.append(float(confidence)) # algılanan nesnenin güven skorunu listeye ekler
            boxes_list.append([start_x, start_y, int(box_width), int(box_height)])


# Non-Maximum Suppression işlemi
max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
#0.4 = nmsThreshold : nesnelerin birbirine olan benzerliğini belirler
#0.5 = confThreshold : güven skorunu belirler

for max_id in max_ids:
    
    max_class_id = max_id
    box = boxes_list[max_class_id]
    start_x = box[0]
    start_y = box[1]
    start_width = box[2]
    start_height = box[3]

    predicted_id = ids_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidences_list[max_class_id]

################# end of operation #######################################

    end_x = start_x + box_width
    end_y = start_y + box_height

    box_color = colors[predicted_id]
    box_color = [int(each) for each in box_color]

    label = "{}: {:.2f}".format(label,confidence*100)
    print("predicted object {}".format(label))

    cv2.rectangle(img, (start_x,start_y), (end_x, end_y), box_color, 3)
    cv2.putText(img, label, (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color,2)
cv2.imread(r"C:\Users\90505\Desktop\yoloV4_tutorials\output.png", img)

cv2.imshow("Detections Window", img)
# cv2.imwrite("output.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
