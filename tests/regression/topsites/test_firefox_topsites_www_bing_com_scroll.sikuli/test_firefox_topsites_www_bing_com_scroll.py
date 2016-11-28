
sys.path.append(sys.argv[2])
import os
import browser
import common
import shutil

com = common.General()
ff = browser.Firefox()

#ff.clickBar()
#ff.enterLink(sys.argv[3])
setAutoWaitTimeout(10)

icon_loc = wait(Pattern('www_bing_com.png').similar(0.80), 60).getTarget()
x_offset = 0
y_offset = 150
inside_window = Location(icon_loc.getX() + x_offset, icon_loc.getY() + y_offset)

mouseMove(inside_window)
sample1_fp = os.path.join(sys.argv[4], sys.argv[5])
sample2_fp = os.path.join(sys.argv[4], sys.argv[5].replace('sample_1', 'sample_2'))
capimg = capture(0, 0, 1024, 768)
os.remove(sample1_fp)
shutil.move(capimg, sample1_fp.replace('jpg', 'png'))

ff.scroll_down(100)
capimg = capture(0, 0, 1024, 768)
shutil.move(capimg, sample2_fp.replace('jpg', 'png'))