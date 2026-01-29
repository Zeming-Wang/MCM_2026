from PIL import Image, ImageSequence
import math

# 定义图像的宽度和高度
DIM = 800
frames = []

# 定义三个颜色计算函数，加入时间变量t，并修改颜色生成逻辑
def RD(i, j, t):
    s = 3.0 / (j + 99)
    y = (j + math.sin((i * i + _sq(j - 700) * 5) / 100.0 / DIM + t) * 35) * s
    return int((math.sin(y + 2.0 * t) + 1) * 127.5)  # 使用sin函数产生平滑的颜色变化

def GR(i, j, t):
    s = 3.0 / (j + 99)
    y = (j + math.sin((i * i + _sq(j - 700) * 5) / 100.0 / DIM + t) * 35) * s
    return int((math.cos(y + 2.0 * t) + 1) * 127.5)  # 使用cos函数产生平滑的颜色变化

def BL(i, j, t):
    s = 3.0 / (j + 99)
    y = (j + math.sin((i * i + _sq(j - 700) * 5) / 100.0 / DIM + t) * 35) * s
    return int((math.sin(y + 4.0 * t) + 1) * 127.5)  # 使用sin函数并增加相位差

# 辅助函数，计算平方
def _sq(x):
    return x * x

# 生成多帧图像，每帧时间变量t递增
for t in range(60):  # 生成60帧动画以增加流畅度
    img = Image.new("RGB", (DIM, DIM))
    for i in range(DIM):
        for j in range(DIM):
            r = RD(i, j, t / 10.0)
            g = GR(i, j, t / 10.0)
            b = BL(i, j, t / 10.0)
            img.putpixel((i, j), (r, g, b))
    frames.append(img)

# 保存为GIF动画，每帧持续时间减少以增加动画速度
frames[0].save('enhanced_ripple_effect.gif', save_all=True, append_images=frames[1:], optimize=False, duration=30, loop=0)