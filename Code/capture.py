import cv2

cam = cv2.VideoCapture(0)
cv2.namedWindow("Capture image")
path = "DATASET/Karan"
img_counter = 1

while True:
    ret,frame = cam.read()
    if not ret:
        print("Failed to take image")
        break
    cv2.imshow("test",frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        print("Capturing Done")
        break
    elif k%256 == 32:
        img_name = "{}.png".format(img_counter)
        cv2.imwrite(f'{path}/{img_name}',frame)
        print("Captured")
        img_counter +=1


    

cam.release()
cv2.destroyAllWindows()