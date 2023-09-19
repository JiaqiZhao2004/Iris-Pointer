o = """"""
o = o.split('\n')

i = []
for l in range(len(o)):
    try:
        if o[l][-1] == ']':
            i.append(o[l])
    except:
        pass

train = []
val = []
for l in range(len(i)):
    if l % 2 == 0:
        train.append(i[l])
    else:
        val.append(i[l])

train = [float(train[i].split("Loss==")[1].split("]")[0]) for i in range(len(train))]
val = [float(val[i].split("Loss==")[1].split("]")[0]) for i in range(len(val))]
import matplotlib.pyplot as plt

plt.semilogy(train)
plt.semilogy(val)
plt.show()
