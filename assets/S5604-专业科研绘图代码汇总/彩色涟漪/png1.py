"""
本文件实现了彩色波动涟漪效果
"""
from PIL import Image
import math

# 定义图像的宽度和高度
DIM = 800
frames = []

def RGB(i, j, t):
    """
    根据给定的参数生成RGB颜色值。
    
    Args:
        i (int): X坐标。
        j (int): Y坐标。
        t (float): 时间参数。
    
    Returns:
        tuple: 包含RGB颜色值的元组，格式为(r, g, b)，每个颜色分量的范围是0-255。
    """
    s = 3.0 / (j + 99)
    y = (j + math.sin((i * i + (j - 700) ** 2 * 5) / 100.0 / DIM + t) * 35) * s
    r = (int((i + DIM) * s + y) % 2 + int((DIM * 2 - i) * s + y) % 2) * 127
    g = (int(5 * ((i + DIM) * s + y)) % 2 + int(5 * ((DIM * 2 - i) * s + y)) % 2) * 127
    b = (int(29 * ((i + DIM) * s + y)) % 2 + int(29 * ((DIM * 2 - i) * s + y)) % 2) * 127
    return (r, g, b)

# 生成多帧图像，每帧时间变量t递增
for t in range(30):
    img = Image.new("RGB", (DIM, DIM))
    for i in range(DIM):
        for j in range(DIM):
            img.putpixel((i, j), RGB(i, j, t / 10.0))
    frames.append(img)

# 保存为GIF动画
frames[0].save('colorful_ripple.gif', save_all=True, append_images=frames[1:], optimize=False, duration=50, loop=0)