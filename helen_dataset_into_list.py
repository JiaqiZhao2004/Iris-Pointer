import math
import os
from PIL import Image


def data_into_list(image_dir, annotation_dir, a=math.e * 100):
    annotations = sorted(os.listdir(annotation_dir))
    images = sorted(os.listdir(image_dir))

    # labels: ['img_path', (left eye)[[x1, y1], [x2, y2], ..., [x20, y20]], (right eye)[[x1, y1], [x2, y2], ...,[x20, y20]]
    labels = []
    for index in range(len(annotations)):
        left_eye_coordinates = []
        right_eye_coordinates = []
        f = open(os.path.join(annotation_dir, annotations[index]), "r").readlines()
        f[0] = f[0][:-1] + '.jpg'
        if f[0] in images:
            (width, height) = Image.open(os.path.join(image_dir, f[0])).size
            # np.array()
            for i in [115, 115+10, 115+5, 115+15]:
                f[i] = f[i][:-1]  # remove \n
                f[i] = f[i].split(" , ")
                for j in range(2):
                    f[i][j] = float(f[i][j])
                f[i][1] -= height / a  # !!! y direction offset
                f[i][0] -= width / a  # !!! x direction offset
                right_eye_coordinates.append(f[i])
            for i in [135, 135+10, 135+5, 135+15]:
                f[i] = f[i][:-1]  # remove \n
                f[i] = f[i].split(" , ")
                for j in range(2):
                    f[i][j] = float(f[i][j])
                f[i][1] -= height / a  # !!! y direction offset
                f[i][0] -= width / a  # !!! x direction offset
                left_eye_coordinates.append(f[i])
            labels.append([os.path.join(image_dir, f[0]), left_eye_coordinates, right_eye_coordinates])
    return labels

# distinguish left and right eye
# Do not need to find midpoint since all annotations are standard

# sample_annotation = labels[444]
# # x_min = 1e9
# # x_max = 0
# for i in range(len(sample_annotation[1])):
#     plt.scatter(sample_annotation[1][i][0], sample_annotation[1][i][1])
#     plt.scatter(sample_annotation[2][i][0], sample_annotation[2][i][1])
#     # x_min = min(x_min, i[0])
#     # x_max = max(x_max, i[0])
# # x_midpoint = (x_min + x_max) // 2
#
# # print([((sample_annotation[1][i][0] - x_midpoint) > 0) for i in range(len(sample_annotation[1]))])
#
# plt.imshow(Image.open(sample_annotation[0]))
# plt.show()
