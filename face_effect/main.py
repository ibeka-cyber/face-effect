import cv2 as cv
import mediapipe as mp
from math import hypot

camera = cv.VideoCapture(1)
nose_img = cv.imread('1.png')

#Yüz ağının oluşturulması
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

solBurun_x, solBurun_y, sagBurun_x, sagBurun_y, merkez_x, merkez_y = 0, 0, 0, 0, 0, 0

while camera.isOpened(): #Projeyi döngüye sokup kameradan okunan karelerin videoya dönüşmesi sağlanır
    _, frame = camera.read() #Görüntüdeki karelerin okunması sağlanır
    RGBimg = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #Mediapipe RGB formatında çalıştığı için dönüşüm yapılır

    yer_isaretleri = faceMesh.process(RGBimg) #RGBimg resmindeki noktaların bulunması sağlanır. Yüzdeki 468 adet noktanın koordinatlarını döndürür.
    if yer_isaretleri.multi_face_landmarks: #Eğer yer_isaretleri boş değilse
        for face_landmarks in yer_isaretleri.multi_face_landmarks: #yer_isaretlerinin içinde gez
            for yer_isareti_id, yer_isareti in enumerate(face_landmarks.landmark): #Noktalarla tek tek işlem yapılması, numaralandırılması
                boy, en, genislik = RGBimg.shape #Kameranın çözünürlüğünün bulunması
                positionX, positionY = int(yer_isareti.x * en), int(yer_isareti.y * boy) #Yer işaretleriyle kameranın çözünürlüğü çarpıldığında o noktanın bulunduğu çözünürlük bulunur

                if yer_isareti_id == 49: #Burnun sol kısmının x ve y koordinatlarının alınması
                    solBurun_x, solBurun_y = positionX, positionY
                if yer_isareti_id == 279: #Burnun sağ kısmının x ve y koordinatlarının alınması
                    sagBurun_x, sagBurun_y = positionX, positionY
                if yer_isareti_id == 5: #Burnun merkez kısmının x ve y koordinatlarının alınması
                    merkez_x, merkez_y = positionX, positionY

            burun_genislik = int(hypot(solBurun_x - sagBurun_x, solBurun_y - sagBurun_y * 1.1)) #Burnun genişliğinin hesaplanması
            burun_yukseklik = int(burun_genislik * 0.8) #Burnun yüksekliğinin göz kararı oranlanması

            if (burun_genislik and burun_yukseklik) > 0:
                nose = cv.resize(nose_img, (burun_genislik, burun_yukseklik)) #Sonradan eklenecek burnun yeniden boyutlandırılması

            sol_ust = (int(merkez_x - burun_genislik / 2), int(merkez_y - burun_yukseklik / 2)+14) #Burnun sol üst noktasının hesaplanması

            burun_bolgesi = frame[sol_ust[1]: sol_ust[1] + burun_yukseklik, sol_ust[0]: sol_ust[0] + burun_genislik] #Sol üst noktaya göre burun bölgesini hesaplama

            #maskeleme işlemleri
            nose_gray = cv.cvtColor(nose, cv.COLOR_BGR2GRAY) #Maskeleme için burnu gri yapma işlemi
            _, nose_mask = cv.threshold(nose_gray, 25, 255, cv.THRESH_BINARY_INV) #Burun üzerinde sonradan nesne eklenmesi için maske oluşturulması
            no_nose = cv.bitwise_and(burun_bolgesi, burun_bolgesi, mask=nose_mask) #Maskeleme işlemindeki beyaz alanların silinmesi
            final_nose = cv.add(no_nose, nose) #Maskelediğimiz kısma fotoğrafın eklenmesi
            frame[sol_ust[1]: sol_ust[1] + burun_yukseklik, sol_ust[0]: sol_ust[0] + burun_genislik] = final_nose #Yaptığımız işlemi kameranın üzerine eklemek

    cv.imshow("Burun Degistirme Filtresi", frame) #Okunan ve işlem yapılan karelerin ekranda gösterilmesi sağlanır
    if cv.waitKey(1) & 0xFF == ord('.'):  # Klavyeden nokta (.) tuşuna basıldığında döngüden çıkılması yani programın kapatılması sağlanır.
        break