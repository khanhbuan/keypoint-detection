import os
import cv2
import math
import numpy as np
import albumentations as A
# from src.models.modelmodule import ModelModule
from src.data.components.dataset import dataset
# from src.models.components.backbone import backbone
from src.data.components.transformed_dataset import transformed_dataset

       #       RED           GREEN          BLACK          CYAN           YELLOW        MAGENTA         GREEN          BLUE 
colors = [[000,000,255], [000,255,000], [000,000,000], [255,255,000], [000,255,255], [255,000,255], [000,255,000], [255,000,000], \
       #      CYAN          BLACK           YELLOW        GREEN           BLUE           CYAN          MAGENTA          CYAN
          [255,255,000], [000,000,000], [000,255,255], [000,255,000], [255,000,000], [255,255,000], [255,000,255], [255,255,000], \
       #      BLUE            GRAY           NAVY           PINK         MAGENTA          CYAN           PINK          YELLOW         GREEN
          [255,000,000], [128,128,128], [000,000,128], [203,192,255], [255,000,255], [255,255,000], [203,192,255], [000,255,255], [000,255,000]]

indexes1 = [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10, 11]]
indexes2 = [[1, 22], [3, 20], [5, 18], [7, 16], [9, 14], [11, 12]]
indexes3 = [[0, 23], [1, 22], [2, 21], [3, 20], [4, 19], [5, 18], [6, 17], [7, 16], [8, 15], [9, 14], [10, 13], [11, 12]]

def get_point(keypoints, idx):
   return (int(keypoints[idx][0]), int(keypoints[idx][1]))

def get_norm_vector(keypoints, indexes, idx):
   p1, p2 = indexes[idx]
   point1, point2 = get_point(keypoints, p1), get_point(keypoints, p2)
   
   if point1 == (0, 0) or point2 == (0, 0):
      return None
   
   direction_vector = (point2[0] - point1[0], point2[1] - point1[1])
   norm_vector = (-direction_vector[1], direction_vector[0])
   return norm_vector

def get_dir_vector(keypoints, indexes, idx):
   p1, p2 = indexes[idx]
   point1, point2 = get_point(keypoints, p1), get_point(keypoints, p2)
   
   if point1 == (0, 0) or point2 == (0, 0):
      return None
   
   dir_vector = (point2[0] - point1[0], point2[1] - point1[1])
   return dir_vector

def calc_midpoint(keypoints, indexes, idx):
   p1, p2 = indexes[idx]
   point1, point2 = get_point(keypoints, p1), get_point(keypoints, p2)
   midpoint = (int((point2[0]+point1[0])/2), int((point2[1] + point1[1])/2))
   return midpoint

def calc_intersection(point1, u_vector1, point2, u_vector2):
   if point1 == (0, 0) or point2 == (0, 0) or u_vector1 is None or u_vector2 is None:
      return None
   
   (x1, y1) = point1
   (a1, b1) = u_vector1
   (x2, y2) = point2
   (a2, b2) = u_vector2

   A = np.array([[a1, -a2], [b1, -b2]])
   B = np.array([x2 - x1, y2 - y1])
   
   try:
      t1, _ = np.linalg.solve(A, B)
      intersection_x = x1 + t1 * a1
      intersection_y = y1 + t1 * b1
      return (int(intersection_x), int(intersection_y))
   
   except np.linalg.LinAlgError:
      return None

def calc_angle(n_vector1, n_vector2):
   x1, y1 = n_vector1[0], n_vector1[1]
   x2, y2 = n_vector2[0], n_vector2[1]
   cos_alpha = abs(x1*x2 + y1*y2) / (math.sqrt((x1**2 + y1**2)*(x2**2+y2**2)))
   alpha = math.acos(cos_alpha)
   return math.degrees(alpha)

def calc_distance(point1, point2, point3):
   (x1, y1), (x2, y2), (x3, y3) = point1, point2, point3

   A = y2 - y3
   B = x3 - x2
   C = x2*y3 - x3*y2
   d = abs(A*x1 + B*y1 + C) / (math.sqrt(A**2 + B**2))
   return d

def calc1(img, keypoints, indexes):
   h, _, _  = img.shape
   scale = int(h/12 - 100/3)
   distances = []
   for l in indexes:
      point1, point2, point3 = get_point(keypoints, l[0]), get_point(keypoints, l[1]), get_point(keypoints, l[2])
      if point1 == (0, 0) or point2 == (0, 0) or point3 == (0, 0):
         distances.append(-math.inf)
         continue
      d = calc_distance(point1, point2, point3)
      distances.append("{:.3f}".format(d/scale * 10))
   return distances

def calc2(keypoints, indexes):
   angles = []
   res = []
   for id in range(len(indexes) - 1):
      """calculate intersection"""
      u1 = get_dir_vector(keypoints, indexes, id)
      p1 = get_point(keypoints, indexes[id][0])
      u2 = get_dir_vector(keypoints, indexes, id+1)
      p2 = get_point(keypoints, indexes[id+1][0])
      intersection = calc_intersection(p1, u1, p2, u2)
      
      if intersection is None:
         res.append("unknown")
      else:
         point1 = calc_midpoint(keypoints, indexes, idx = id)
         point2 = calc_midpoint(keypoints, indexes, idx = id+1)
         key = (point1[0] - point2[0], point1[1] - point2[1])
         query = (intersection[0] - point2[0], intersection[1] - point2[1])
         cross_product = key[0]*query[1] - key[1]*query[0]
         if cross_product < 0:
            res.append("right")
         else:
            res.append("left")

      """calculate angle"""
      n1 = get_norm_vector(keypoints, indexes, id)
      n2 = get_norm_vector(keypoints, indexes, id+1)

      if n1 is None or n2 is None:
         angle = -math.inf
         angles.append(angle)
      else:
         angle = calc_angle(n1, n2)
         if res[-1] == 'right':
            angle = -1 * angle
         angles.append("{:.3f}".format(angle))

   return angles

def expand(start_point, end_point, scale):
   direction = (end_point[0] - start_point[0], end_point[1] - start_point[1])
   new_start_point = (start_point[0] - scale * direction[0], start_point[1] - scale * direction[1])
   new_end_point = (end_point[0] + scale * direction[0], end_point[1] + scale * direction[1])

   return new_start_point, new_end_point

def draw_point(img, radius, idx, target, pred=None):
   overlay = img.copy()

   for i in range(len(target)):
      if pred is not None:
         x_pred = int(pred[i][0])
         y_pred = int(pred[i][1])
         cv2.circle(img, (x_pred, y_pred), radius=radius, thickness=1, color=[000,000,255])
         cv2.circle(overlay, (x_pred, y_pred), radius=radius, thickness=1, color=[000,000,255])

      x_target = int(target[i][0])
      y_target = int(target[i][1])
      cv2.circle(overlay, (x_target, y_target), radius=radius, thickness=-1, color=colors[4])

   alpha = 0.5
   img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
   cv2.imwrite(os.path.join('./visualization', 'output' + str(idx) + '.png'), img)

def draw_line(img, keypoints, indexes1, indexes2, angles, distances, thickness, idx):
   """draw line to calculate the spinal displacement"""
   for l in indexes1:
      point1, point2, point3 = get_point(keypoints, l[0]), get_point(keypoints, l[1]), get_point(keypoints, l[2])
      if point1 == (0, 0) or point2 == (0, 0) or point3 == (0, 0):
         continue
      point4 = (point1[0] - (point2[0] - point3[0]), point1[1] - (point2[1] - point3[1]))
      cv2.line(img, point2, point3, colors[0], thickness=thickness)
      cv2.line(img, point1, point4, colors[1], thickness=thickness)
   
   """draw line to calculate the hump angle"""
   for id in range(len(indexes2)):
      p1, p2 = indexes2[id]
      point1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
      point2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))
      if point1 == (0, 0) or point2 == (0, 0):
         continue
      point1, point2 = expand(point1, point2, scale=2)
      
      cv2.line(img, point1, point2, colors[3], thickness=thickness)

   """add angles to image"""
   img = cv2.putText(img, "angle:", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
   for id, angle in enumerate(angles):
      if angle == -math.inf:
         continue
      if float(angle) >= 0:
         angle = 'a' + str(id+2) + str(id+3) + ": +" + str(angle) + ' degree'
      else:
         angle = 'a' + str(id+2) + str(id+3) + ": " + str(angle) + ' degree'
      img = cv2.putText(img, angle, (10, 150 + 25*(id+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

   """add distances to image"""
   img = cv2.putText(img, "distance:", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
   for id, d in enumerate(distances):
      if d == -math.inf:
         continue
      d = "d" + str(id+2) + str(id+3) + ": " + d + "mm"
      img = cv2.putText(img, d, (300, 150 + 25*(id+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

   cv2.imwrite(os.path.join("./render", 'output'+ str(idx) + '.png'), img)

def draw_cobb_c2c7(img, keypoints, cobb, idx):
   point1, point2, point3, point4 = get_point(keypoints, idx=1), get_point(keypoints, idx=22), \
                                    get_point(keypoints, idx=11), get_point(keypoints, idx=12)

   cv2.circle(img, point1, radius=3, thickness=-1, color=colors[1])
   cv2.circle(img, point2, radius=3, thickness=-1, color=colors[1])
   cv2.circle(img, point3, radius=3, thickness=-1, color=colors[1])
   cv2.circle(img, point4, radius=3, thickness=-1, color=colors[1])

   mid1, mid2 = calc_midpoint(keypoints, indexes2, idx=0), calc_midpoint(keypoints, indexes2, idx=5)
   n1, n2 = get_norm_vector(keypoints, indexes2, idx=0), get_norm_vector(keypoints, indexes2, idx=5)

   end1 = (mid1[0] + n1[0], mid1[1] + n1[1])
   mid1, end1 = expand(mid1, end1, scale=5)
   if n2 is not None:
      end2 = (mid2[0] - n2[0], mid2[1] - n2[1])
      mid2, end2 = expand(mid2, end2, scale=5)

   cv2.line(img, mid1, end1, colors[4], thickness=1)
   if mid2 is not None and n2 is not None:
      cv2.line(img, mid2, end2, colors[4], thickness=1)
 
   point1, point2 = expand(point1, point2, scale=2)
   point3, point4 = expand(point3, point4, scale=2)

   cv2.line(img, point1, point2, colors[3], thickness=1)
   cv2.line(img, point3, point4, colors[3], thickness=1)
   img = cv2.putText(img, "Cobb_C2C7: " + str(cobb) + " degree", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

   cv2.imwrite(os.path.join("./cobb", 'output'+ str(idx) + '.png'), img)

def cal_cobb_c2c7(keypoints, indexes):
   n1 = get_norm_vector(keypoints, indexes, 0)
   n2 = get_norm_vector(keypoints, indexes, 5)
   
   if n1 is None or n2 is None:
      angle = -math.inf
   else:
      angle = calc_angle(n1, n2)
      angle = "{:.3f}".format(angle)
   
   return angle

def cal_slope(keypoints, indexes):
   n2 = get_norm_vector(keypoints, indexes, idx=0)
   n7 = get_norm_vector(keypoints, indexes, idx=4)
   base = (0, h)

   if n2 is None:
      angle2 = -math.inf
   else:
      angle2 = calc_angle(n2, base)
      angle2 = "{:.3f}".format(angle2)
   if n7 is None:
      angle7 = -math.inf
   else:
      angle7 = calc_angle(n7, base)
      angle7 = "{:.3f}".format(angle7)
   return angle2, angle7

def draw_slope(img, keypoints, slope2, slope7, idx):
   point1, point2 = get_point(keypoints, idx=1), get_point(keypoints, idx=22)
   cv2.circle(img, point1, radius=3, thickness=-1, color=colors[1])
   cv2.circle(img, point2, radius=3, thickness=-1, color=colors[1])
   point3 = (point1[0] + 20, point1[1])

   point4, point5 = get_point(keypoints, idx=10), get_point(keypoints, idx=13)
   cv2.circle(img, point4, radius=3, thickness=-1, color=colors[1])
   cv2.circle(img, point5, radius=3, thickness=-1, color=colors[1])
   point6 = (point4[0] + 20, point4[1])

   point11, point2 = expand(point1, point2, scale=3)
   cv2.line(img, point11, point2, colors[0], thickness=1)
   point1, point3 = expand(point1, point3, scale=6)
   cv2.line(img, point1, point3, colors[4], thickness=1)
   
   point44, point5 = expand(point4, point5, scale=3)
   cv2.line(img, point44, point5, colors[0], thickness=1)
   point4, point6 = expand(point4, point6, scale=6)
   cv2.line(img, point4, point6, colors[4], thickness=1)

   img = cv2.putText(img, "C2 Slope: " + str(slope2) + " degree", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
   img = cv2.putText(img, "C7 Slope: " + str(slope7) + " degree", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
   cv2.imwrite(os.path.join("./slope", 'output'+ str(idx) + '.png'), img)

def cal_draw_c1c7_sva(img, keypoints):
   h, _, _  = img.shape
   scale = int(h/12 - 100/3)
   point7, point1 = get_point(keypoints, idx=10), get_point(keypoints, idx=24)
   cv2.circle(img, point1, radius=2, thickness=-1, color=colors[4])
   cv2.circle(img, point7, radius=2, thickness=-1, color=colors[1])
   dis = abs(point7[0] - point1[0])

   dis = "{:.3f}".format(dis/scale * 10)

   point1x = (point1[0], point1[1] + 350)
   point7x = (point7[0], point7[1] - 50)

   cv2.line(img, point1, point1x, colors[0], thickness=1)
   cv2.line(img, point7, point7x, colors[0], thickness=1)
   cv2.line(img, point7x, (point1[0], point7x[1]), colors[4], thickness=1)
   img = cv2.putText(img, "C1C7_SVA: " + str(dis) + "mm", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
   cv2.imwrite(os.path.join("./sva", 'output'+ str(idx) + '.png'), img)
   
def cal_draw_c2c7_sva(img, keypoints):
   h, _, _  = img.shape
   scale = int(h/12 - 100/3)

   point1, point2, point3, point4 = get_point(keypoints, idx=0), get_point(keypoints, idx=1), \
                                    get_point(keypoints, idx=22), get_point(keypoints, idx=23)

   u1 = (point3[0] - point1[0], point3[1] - point1[1])
   u2 = (point4[0] - point2[0], point4[1] - point2[1])

   intersection = calc_intersection(point1, u1, point2, u2)
   point7 = get_point(keypoints, idx=10)
   cv2.line(img, point7, (intersection[0], point7[1]), colors[1])

   cv2.circle(img, point1, radius=2, thickness=-1, color=colors[1])
   cv2.circle(img, point2, radius=2, thickness=-1, color=colors[1])
   cv2.circle(img, point3, radius=2, thickness=-1, color=colors[1])
   cv2.circle(img, point4, radius=2, thickness=-1, color=colors[1])
   cv2.circle(img, point7, radius=2, thickness=-1, color=colors[1])
   cv2.circle(img, intersection, radius=2, thickness=-1, color=colors[1])
   cv2.line(img, point1, point3, colors[0], thickness=1)
   cv2.line(img, point2, point4, colors[0], thickness=1)


   dis = abs(point7[0] - intersection[0])
   dis = "{:.3f}".format(dis/scale * 10)

   intersectionx = (intersection[0], intersection[1] + 350)
   point7x = (point7[0], point7[1] - 50)

   cv2.line(img, intersection, intersectionx, colors[0], thickness=1)
   cv2.line(img, point7, point7x, colors[0], thickness=1)
   img = cv2.putText(img, "C2C7_SVA: " + str(dis) + "mm", (10, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
   cv2.imwrite(os.path.join("./sva", 'output'+ str(idx) + '.png'), img)

def cal_FSU(keypoints, indexes):
   angles = []
   res = []

   for id in range(0, len(indexes)-3, 2):
      u1 = get_dir_vector(keypoints, indexes, id)
      p1 = get_point(keypoints, indexes[id][0])
      u2 = get_dir_vector(keypoints, indexes, id+3)
      p2 = get_point(keypoints, indexes[id+3][0])
      intersection = calc_intersection(p1, u1, p2, u2)
      
      if intersection is None:
         res.append("unknown")
      else:
         point1 = calc_midpoint(keypoints, indexes, idx = id)
         point2 = calc_midpoint(keypoints, indexes, idx = id+3)
         key = (point1[0] - point2[0], point1[1] - point2[1])
         query = (intersection[0] - point2[0], intersection[1] - point2[1])
         cross_product = key[0]*query[1] - key[1]*query[0]
         if cross_product < 0:
            res.append("right")
         else:
            res.append("left")

      n1 = get_norm_vector(keypoints, indexes, id)
      n2 = get_norm_vector(keypoints, indexes, id+3)

      if n1 is None or n2 is None:
         angle = -math.inf
         angles.append(angle)
      else:
         angle = calc_angle(n1, n2)
         if res[-1] == 'right':
            angle = -1 * angle
         angles.append("{:.3f}".format(angle))

   return angles

def draw_FSU(img, img_path, keypoints, indexes, angles):
   for id in range(0, len(indexes)-3, 2):
      p1, p2 = get_point(keypoints, indexes[id][0]), get_point(keypoints, indexes[id][1])
      p3, p4 = get_point(keypoints, indexes[id+3][0]), get_point(keypoints, indexes[id+3][1])
      cv2.circle(img, p1, radius=2, thickness=-1, color=colors[1])
      cv2.circle(img, p2, radius=2, thickness=-1, color=colors[1])
      cv2.circle(img, p3, radius=2, thickness=-1, color=colors[1])
      cv2.circle(img, p4, radius=2, thickness=-1, color=colors[1])

      p11, p22 = expand(p1, p2, scale=3)
      p33, p44 = expand(p3, p4, scale=3)
      cv2.line(img, p11, p22, colors[id], thickness=2)
      cv2.line(img, p33, p44, colors[id], thickness=2)

   if not os.path.exists(os.path.join("FSU", img_path.split("/")[-2])):
      os.mkdir(os.path.join("FSU", img_path.split("/")[-2]))

   img = cv2.putText(img, "FSU angle:", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
   for id, angle in enumerate(angles):
      if angle == -math.inf:
         continue
      angle = 'FSU C' + str(id+2) + "_C" + str(id+3) + ": " + str(angle) + ' degree'
      img = cv2.putText(img, angle, (10, 150 + 25*(id+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

   cv2.imwrite(os.path.join("./FSU",  str(img_path.split('/')[-2]), str(img_path.split('/')[-1])), img)

if __name__ == "__main__":
   transform = A.Compose([
        A.Resize(height=700, width=700),
        A.CenterCrop(height=384, width=384),
   ],bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']), 
     keypoint_params = A.KeypointParams(format="xy", remove_invisible=False))
   
   dataset = transformed_dataset(dataset=dataset("./data"), transform=transform)

   # ckpt_path = "./logs/train/runs/2024-08-07_09-52-59/checkpoints/epoch_397.ckpt"
   # model = ModelModule.load_from_checkpoint(net=backbone(), checkpoint_path=ckpt_path)

   FSU = {}

   for idx, sample in enumerate(dataset):
      img_path, _ = dataset.dataset.__getitem__(idx)
      input, target = sample

      # input = input[None,:,:,:].to("cuda")
      # pred = model(input)
      # pred = pred[0]["keypoints"][0].cpu().detach().numpy()
      # pred = pred + np.array([158, 158, 0])

      target = target["keypoints"][0].detach().numpy()
      target = target + np.array([158, 158, 0])

      img = cv2.imread(img_path)
      h, w, _ = img.shape

      transform = A.Compose([
         A.Resize(height=h, width=h)
         ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
      
      transformed = transform(image=np.zeros((700, 700, 3), dtype=np.float32), keypoints=target)
      target = transformed["keypoints"]
      target += np.array([w/2-h/2, 0, 0])

      # transformed = transform(image=np.zeros((700, 700, 3), dtype=np.float32), keypoints=pred)
      # pred = transformed["keypoints"]
      # pred += np.array([w/2-h/2, 0, 0])

      # draw_point(img.copy(), 2, -1, idx, target)

      distances = calc1(img.copy(), target, indexes1)
      angles = calc2(target, indexes2)
      cobb = cal_cobb_c2c7(target, indexes2)

      draw_line(img.copy(), target, indexes1, indexes2, angles, distances, thickness=1, idx=idx)
      draw_cobb_c2c7(img.copy(), target, cobb, idx=idx)
      angles = cal_FSU(target, indexes3)

      draw_FSU(img.copy(), img_path, target, indexes3, angles)

      if "Nghiêng" not in img_path:
         FSU[img_path] = angles

      else:
         slope2, slope7 = cal_slope(target, indexes2)
         draw_slope(img.copy(), target, slope2, slope7, idx=idx)
         img3 = img.copy()
         cal_draw_c1c7_sva(img3, target)
         cal_draw_c2c7_sva(img3, target)
   
   keys = list(FSU.keys())
   for id1 in range(len(keys) - 1):
      for id2 in range(id1+1, len(keys)):
         if keys[id1].split("/")[-2] == keys[id2].split("/")[-2]:
            if "Cúi" in keys[id1]:
               cúi = keys[id1]
               ưỡn = keys[id2]
            else:
               cúi = keys[id2]
               ưỡn = keys[id1]

            os.mkdir(os.path.join("ROM", keys[id1].split("/")[-2]))
            img1 = cv2.imread(cúi)
            img2 = cv2.imread(ưỡn)
            img1 = cv2.putText(img1, "ROM:", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            img2 = cv2.putText(img2, "ROM:", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            for idx in range(len(FSU[cúi])):
               if isinstance(FSU[cúi][idx], float) or isinstance(FSU[ưỡn][idx], float):
                  continue
               
               num = "{:.3f}".format(abs(float(FSU[cúi][idx]) - float(FSU[ưỡn][idx])))
               ROM = 'ROM C' + str(idx+2) + "_C" + str(idx+3) + ": " + str(num) + ' degree'
               img1 = cv2.putText(img1, ROM, (10, 150 + 25*(idx+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
               img2 = cv2.putText(img2, ROM, (10, 150 + 25*(idx+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imwrite(os.path.join("./ROM",  str(cúi.split('/')[-2]), str(cúi.split('/')[-1])), img1)
            cv2.imwrite(os.path.join("./ROM",  str(ưỡn.split('/')[-2]), str(ưỡn.split('/')[-1])), img2)
