import cv2
import numpy as np

vid = cv2.VideoCapture("traffic.avi")
backsub=cv2.createBackgroundSubtractorKNN()#arkaplan çıkarma fonksiyonu 
#BackgroundSubtractorMOG2 kursta bu fonksiyon kullanılıyor fakat burada gölgeler işin içine girdiği için hatalı sayım yapıyor KNN'de gölgeler gri renkli ve sayılmıyor
count=0

while True:
    ret,frame=vid.read()
    if ret:
        fgmask=backsub.apply(frame)#arkaplan çıkarılmış hali
        cv2.imshow("img",fgmask)

        cv2.line(frame,(50,0),(50,300),(0,255,0),2)
        cv2.line(frame, (70, 0), (70, 300), (0, 255, 0), 2)

        countours,hierarchy= cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        try:hierarchy=hierarchy[0]
        except:hierarchy=[]

        for countour,hier in zip(countours,hierarchy):
            (x,y,w,h)=cv2.boundingRect(countour)#x,y koordinat değerleri w,h genişlik ve yükseklik değerleri
            if w>40 and h>40:# genişlik ve yükseklik 40'dan büyükse o bölgeye dikdörtgen çiziyoruz bu orada araba var demek
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                if x>50 and x<70:# koordinatlar benim çizdiğim kordinatlardan geçiyorsa buradan araba geçti demek sayacı artırıyoruz
                    count+=1
        cv2.putText(frame,"car: "+str(count),(90,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)

        cv2.imshow("Car Counter",frame)
        cv2.imshow("Fgmask",fgmask)

        if cv2.waitKey(20) & 0xFF==ord('q'):
            break

vid.release()
cv2.destroyAllWindows()

