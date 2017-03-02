# if you are putting your test script folders under {git project folder}/tests/, it will work fine.
# otherwise, you either add it to system path before you run or hard coded it in here.
sys.path.append(sys.argv[2])
import os
import ast
import gdoc
import shutil
import browser

chrome = browser.Chrome()
gd = gdoc.gDoc()
chrome.clickBar()
chrome.enterLink(sys.argv[3])
gd.wait_for_loaded()

setAutoWaitTimeout(10)
sample1_fp = os.path.join(sys.argv[4], sys.argv[5])
sample2_fp = os.path.join(sys.argv[4], sys.argv[5].replace('sample_1', 'sample_2'))
os.remove(sample1_fp)

paste("Start")
sleep(0.5)
type(Key.ENTER)
sleep(2)
x = 0
y = 0
width = 0
height = 0
viewport = ast.literal_eval(sys.argv[6])
capture_width = viewport['x'] + viewport['width']
capture_height = viewport['y'] + viewport['height']
capimg1 = capture(0, 0, capture_width, capture_height)
count = 0
while count < 60:
    type("#aaaaaaaaaaaaaaaaaaaa")
    sleep(1)
    count += 1
capimg2 = capture(0, 0, capture_width, capture_height)

shutil.move(capimg1, sample1_fp.replace('jpg', 'png'))
shutil.move(capimg2, sample2_fp.replace('jpg', 'png'))
