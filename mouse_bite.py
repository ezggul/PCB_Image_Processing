import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

#XML dosyasını işleme
def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append((name, xmin, ymin, xmax, ymax))
    return objects

#Resimleri yükleme
original_image = cv2.imread('Reference/01.jpg')
distorted_image = cv2.imread('rotation/Mouse_bite_rotation/01_mouse_bite_03.jpg')

if original_image is None:
    raise FileNotFoundError("Original image not found. Please check the path.")
if distorted_image is None:
    raise FileNotFoundError("Distorted image not found. Please check the path.")

#Grayscale'e çevirme
gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
gray_distorted = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2GRAY)

#ORB dedektörü
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray_original, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_distorted, None)

#BFMatcher ile en yakın eşleşmeleri bulma
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

#Mesafeye göre eşleşmeleri sıralama
matches = sorted(matches, key=lambda x: x.distance)

#En iyi eşleşmeleri seçme
good_matches = matches[:30]

#Eşleşen noktaları çıkarma
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

#Homografi matrisini hesaplama
H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

#Yamuk olan fotoğrafı düzeltme
height, width, channels = original_image.shape
aligned_image = cv2.warpPerspective(distorted_image, H, (width, height))

#Yeşil bölgeleri tespit etme
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 40, 40])  # Yeşil için alt sınır
upper_green = np.array([85, 255, 255])  # Yeşil için üst sınır
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

#Morfolojik işlemlerle maskeyi iyileştirme
kernel = np.ones((5, 5), np.uint8)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

#Farklılıkları tespit etme
difference = cv2.absdiff(original_image, aligned_image)
gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
masked_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=green_mask)

#Eşikleme işlemi
_, thresh = cv2.threshold(masked_diff, 25, 255, cv2.THRESH_BINARY)

#Morfolojik işlemler
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

#Konturları bulma
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Mouse bite yerlerini kare içine alma
marked_image = aligned_image.copy()
detected_points = []
for contour in contours:
    area = cv2.contourArea(contour)
    if 15 < area < 200:  # Alan aralığını daralttık
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(marked_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        detected_points.append((x + w/2, y + h/2))  # Karelerin merkez noktalarını kaydetme

#XML dosyası
xml_file_path = 'Annotations/Mouse_Bite/01_mouse_bite_03.xml'

xml_objects = parse_xml(xml_file_path)

#Doğruluk oranı
threshold = 10
correct_matches = 0
for detected_point in detected_points:
    for obj in xml_objects:
        name, xmin, ymin, xmax, ymax = obj
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        if abs(center_x - detected_point[0]) < threshold and abs(center_y - detected_point[1]) < threshold:
            correct_matches += 1
            break

total_detected = len(detected_points)
total_true = len(xml_objects)
accuracy = correct_matches / total_true if total_true > 0 else 0

print(f"Toplam Tespit Edilen: {total_detected}")
print(f"Toplam Gerçek: {total_true}")
print(f"Doğru Eşleşmeler: {correct_matches}")
print(f"Doğruluk Oranı: {accuracy * 100:.2f}%")

#XML verilerini görselleştirme
for obj in xml_objects:
    name, xmin, ymin, xmax, ymax = obj
    cv2.rectangle(marked_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # XML verilerini yeşil karelerle işaretleme

# Sonuçları gösterme
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.title('Distorted Image')
plt.imshow(cv2.cvtColor(distorted_image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 3)
plt.title('Aligned Image')
plt.imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 4)
plt.title('Marked Image with Mouse Bites')
plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
plt.show()
