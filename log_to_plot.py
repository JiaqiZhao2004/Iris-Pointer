import matplotlib.pyplot as plt
import os

log = True
prefix = "4_points_resnet18_linear_1_small_img_"


train = []
val = []

if log:
    o = """"""
    o = o.split('\n')

    i = []
    for l in range(len(o)):
        try:
            if o[l][-1] == ']':
                i.append(o[l])
        except:
            pass

    for l in range(len(i)):
        if l % 2 == 0:
            train.append(i[l])
        else:
            val.append(i[l])

    train = [float(train[i].split("Loss==")[1].split("]")[0]) for i in range(len(train))]
    val = [float(val[i].split("Loss==")[1].split("]")[0]) for i in range(len(val))]
    plt.semilogy(train)
    plt.semilogy(val)


else:
    o = os.listdir("weights")
    x = []
    y = []
    for item in o:
        if item.split("epoch=")[0] == prefix:
            x.append(float(item.split("epoch=")[1].split("_loss")[0]))
            y.append(float(item.split("epoch=")[1].split("loss=")[1].split(".pth")[0]))

    plt.scatter(x, y)
    plt.yscale('log')

plt.show()
