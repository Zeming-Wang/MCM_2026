from PIL import Image
import math

# 定义图像的宽度和高度
DIM = 800

# 定义三个颜色计算函数
def RD(i, j):
    s = 3.0 / (j + 99)
    y = (j + math.sin((i * i + _sq(j - 700) * 5) / 100.0 / DIM) * 35) * s
    return (int((i + DIM) * s + y) % 2 + int((DIM * 2 - i) * s + y) % 2) * 127

def GR(i, j):
    s = 3.0 / (j + 99)
    y = (j + math.sin((i * i + _sq(j - 700) * 5) / 100.0 / DIM) * 35) * s
    return (int(5 * ((i + DIM) * s + y)) % 2 + int(5 * ((DIM * 2 - i) * s + y)) % 2) * 127

def BL(i, j):
    s = 3.0 / (j + 99)
    y = (j + math.sin((i * i + _sq(j - 700) * 5) / 100.0 / DIM) * 35) * s
    return (int(29 * ((i + DIM) * s + y)) % 2 + int(29 * ((DIM * 2 - i) * s + y)) % 2) * 127

# 辅助函数，计算平方
def _sq(x):
    return x * x

# 创建一个新图像，模式为RGB
img = Image.new("RGB", (DIM, DIM))

# 为图像的每个像素点赋值
for i in range(DIM):
    for j in range(DIM):
        r = RD(i, j)
        g = GR(i, j)
        b = BL(i, j)
        img.putpixel((i, j), (r, g, b))

# 显示图像
img.show()

# 可选：保存图像
img.save("output1.png")