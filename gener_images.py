from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import numpy as np

A1 = 'A'
C1 = 'C'
T1 = 'T'
G1 = 'G'

exec("%s = %d" % (A1,0))
exec("%s = %d" % (C1,80))
exec("%s = %d" % (T1,160))
exec("%s = %d" % (G1,255))

Sec_01 = open('sec_1.txt','r')
b = Sec_01.read().split(sep=' ')

print(b)

b1 = np.array([list(b)],'U1')
print('b1')
print(b1)
b2 = b1.dtype
print(b2)
b1_as_int = b1.view(np.uint32)
print(b1_as_int)
b3 = b1_as_int.dtype
print(b3)

a = np.array([b1_as_int]).reshape(1, 6)
print(a)
q = a.dtype
print(q)
plt.imshow(a, cmap = "brg")
plt.show()

img = Image.new(plt.imshow(a, cmap = "brg"), (800, 400))
draw = ImageDraw.Draw(img)
img.save('image.png')
img.show()
