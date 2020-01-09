import pyautogui
import time

x = 1119
y = 307
y = y + 50 * 3
x = x + 50 * 2
# jiaoyi

i = 1
while i < 4:
    currentMouseX, currentMouseY = pyautogui.position()
    print(currentMouseX, currentMouseY)
    time.sleep(1)
    pyautogui.rightClick(x=x, y=y,  interval=1)
    x = x + 50
    i = i + 1

time.sleep(1)
pyautogui.moveTo(x=300, y=694)
time.sleep(0.5)
pyautogui.click(interval=1)
# pyautogui.leftClick(x=300, y=694, interval=4)
# time.sleep(1)
# pyautogui.leftClick(x=300, y=694, interval=1)
# 交易框位置: 300 694
# 整理背包位置: 1166 264

# 第00格:1119 307
# 左右各50增加
