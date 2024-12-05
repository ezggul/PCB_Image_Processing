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

#XML koordinatlarıyla tespit ettiklerimizi karşılaştırma fonk.
def compare_points(detected_points, xml_objects, threshold=10):
    matched_points = []
    for point in detected_points:
        px, py = point
        for obj in xml_objects:
            name, xmin, ymin, xmax, ymax = obj
            if xmin - threshold <= px <= xmax + threshold and ymin - threshold <= py <= ymax + threshold:
                matched_points.append((point, obj))
                break
    return matched_points

#Resimleri yükleme
original_image = cv2.imread('Reference/01.jpg')
distorted_image = cv2.imread('rotation/Missing_hole_rotation/01_missing_hole_03.jpg')

if original_image is None:
    raise FileNotFoundError("Original image not found. Please check the path.")
if distorted_image is None:
    raise FileNotFoundError("Distorted image not found. Please check the path.")

#Grayscale'e çevirme
gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
gray_distorted = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2GRAY)

#ORB dedektörü ile özellik noktaları ve tanımlayıcıları bulma
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

#Farklılıkları tespit etme
difference = cv2.absdiff(original_image, aligned_image)
gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

#Eşikleme işlemi
_, thresh = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

#Konturları bulma
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Eksik yerleri yuvarlak içine alma
marked_image = aligned_image.copy()
detected_points = []
for contour in contours:
    if cv2.contourArea(contour) > 100:  #Küçük gürültüleri engellemek için bir alan filtresi
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius) + 5  #Yarıçapı artırma
        cv2.circle(marked_image, center, radius, (0, 0, 255), 10)  #Daire kalınlığını artırma
        detected_points.append((x, y))  #Noktaların merkezini kaydetme

#XML dosyasını alma ve işlemlerini gerçekleştirme
xml_file_path = 'Annotations/Missing_Hole/01_missing_hole_03.xml'

xml_objects = parse_xml(xml_file_path)
matched_points = compare_points(detected_points, xml_objects)

#Doğruluk oranını hesaplama
total_detected = len(detected_points)
total_true = len(xml_objects)
correct_matches = len(matched_points)
accuracy = correct_matches / total_true if total_true > 0 else 0

print(f"Toplam Tespit Edilen: {total_detected}")
print(f"Toplam Gerçek: {total_true}")
print(f"Doğru Eşleşmeler: {correct_matches}")
print(f"Doğruluk Oranı: {accuracy * 100:.2f}%")

#XML verilerini görselleştirme
for obj in xml_objects:
    name, xmin, ymin, xmax, ymax = obj
    cv2.rectangle(marked_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 5)  #yeşil ile işaretlenecek

#Sonuçlar
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
plt.title('Marked Image with Missing Holes')
plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))

plt.show()
