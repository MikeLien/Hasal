wait("ff_urlbar.png")
click(Pattern("ff_urlbar.png").targetOffset(-100, 0))
paste("https://us.yahoo.com/")
type(Key.ENTER)
setAutoWaitTimeout(30)
wait(Pattern('us_yahoo_com.png').similar(0.80), 60)
wheel(Pattern("us_yahoo_com.png").similar(0.85).targetOffset(0, 150), WHEEL_DOWN, 100)
sleep(2)
wheel(Pattern("us_yahoo_com.png").similar(0.85).targetOffset(0, 150), WHEEL_UP, 100)
