find("chrome_urlbar.png")
click("chrome_urlbar.png")
type("https://www.facebook.com/groups/mozhasalvideo")
type(Key.ENTER)
setAutoWaitTimeout(10)
wait("1468830683686.png")
doubleClick(Pattern("1468577119085-1.png").targetOffset(0,200))
wait("1468914267501.png",240)