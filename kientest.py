from PIL import ImageFont, ImageDraw, Image
import numpy as np
from easyocr import Reader
import cv2


anh = cv2.imread('im2.jpg')
anh = cv2.resize(anh, (800, 600))
duong_dan_font = "./times.ttf"
font = ImageFont.truetype(duong_dan_font, 32)
b,g,r,a = 0,255,0,0

xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
mo = cv2.GaussianBlur(xam, (5, 5), 0)
vien = cv2.Canny(mo, 10, 200)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _ = cv2.findContours(vien, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

for c in contours:
    chu_vi = cv2.arcLength(c, True)
    gan_dung = cv2.approxPolyDP(c, 0.02 * chu_vi, True)
    print(gan_dung)
    if len(gan_dung) == 4:
        hinh_bien_so = gan_dung
        break

(x, y, w, h) = cv2.boundingRect(hinh_bien_so)
bien_so = xam[y:y + h, x:x + w]

doc = Reader(['en'])
phan_loai = doc.readtext(bien_so)

if len(phan_loai) == 0:
    van_ban = "Không thấy bảng số xe"
    anh_pil = Image.fromarray(anh)
    ve = ImageDraw.Draw(anh_pil)
    ve.text((150, 500), van_ban, font = font, fill = (b, g, r, a))
    anh = np.array(anh_pil) 
    cv2.waitKey(0)
else:
    cv2.drawContours(anh, [hinh_bien_so], -1, (255, 0, 0), 3)
    van_ban ="Biển số: " + f"{phan_loai[0][1]}"
    anh_pil = Image.fromarray(anh) 
    ve = ImageDraw.Draw(anh_pil)
    ve.text((200, 500), van_ban, font = font, fill = (b, g, r, a))
    anh = np.array(anh_pil) 
    cv2.imshow('VMU', anh)
    cv2.waitKey(0)
