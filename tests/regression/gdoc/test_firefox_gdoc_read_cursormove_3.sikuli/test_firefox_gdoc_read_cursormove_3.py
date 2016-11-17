# if you are putting your test script folders under {git project folder}/tests/, it will work fine.
# otherwise, you either add it to system path before you run or hard coded it in here.
sys.path.append(sys.argv[2])
import browser
import common
import gdoc
import shutil
import os

com = common.General()
ff = browser.Firefox()
gd = gdoc.gDoc()

# ff.clickBar()
# ff.enterLink(sys.argv[3])

setAutoWaitTimeout(10)
# gd.wait_for_loaded()
sample1_fp = os.path.join(sys.argv[4], sys.argv[5])
sample2_fp = os.path.join(sys.argv[4], sys.argv[5].replace('sample_1', 'sample_2'))
# sleep(2)
capimg = capture(0, 0, 1024, 768)
os.remove(sample1_fp)
shutil.move(capimg, sample1_fp.replace('jpg', 'png'))
wheel(Pattern("pics/doc_content_left_top_page_region.png").similar(0.85), WHEEL_DOWN, 100)
capimg = capture(0, 0, 1024, 768)
shutil.move(capimg, sample2_fp.replace('jpg', 'png'))
