const puppeteer = require('puppeteer');
var fs = require('fs');

async function getIMGURL(imgnumber, name) {
    createfile( name)
    const browser = await puppeteer.launch({ headless: false });
    const page = await browser.newPage();
    try {
        // 打开百度网站
        await page.goto('https://image.baidu.com/');
        // 输入搜索的名字
        await page.type("#kw", name, { delay: 400 })
        // 点击搜索
        await page.click(".s_search")

        await page.waitForSelector("#imgid .imgpage")
        

        let i = 1
        var li
        var lastn = 0
        while (true) {
            await page.evaluate(function (i) {
                /* 这里做的是渐进滚动，如果一次性滚动则不会触发获取新数据的监听 */
                for (var y = 0; y <= 1000 * i; y += 100) {
                    window.scrollTo(0, y)
                }
            }, i++)
            // 判断数量是否足够
            li = await page.$$("#imgid .imgpage ul li")
            
            if (lastn == li.length) {
                console.log("没有更多图片!");
                break
            }
            // console.log(m);
            if (imgnumber <= li.length) {
                break
            }
            lastn = li.length
            await page.waitFor(1000)
        }

        for (let i = 0; i < li.length; i++) {
            const el = li[i];
            let s = await el.$eval("img", e => e.src)
            console.log(i, s);
            await down(browser, s, name, i)
            if (imgnumber<= i) {
                break
            }
        }

    } catch (error) {
        console.log(error);
    }
    await browser.close();
}

async function down(browser, url, name, i) {
    try {
        const page = await browser.newPage();
        await page.goto(url)
        await page.waitFor(500)
        let l = await page.$("img")
        log(i, url, name)
        await l.screenshot({
            path: "data/" + name + "/" + name + i + ".png",
            delay: 100,
        })
        await page.close()
    } catch (error) {
        console.log(error);
    }
}

function log(i, url, name) {
    if (url.substr(0,4) == "data") {
        url = ""
    }
    
    fs.appendFileSync("1.log", new Date().toLocaleString() + "\t" + i + "\t" + name + "\t"+ url+ "\n")
}

function createfile(name) {
    name = "data/" + name
    if (fs.existsSync(name)) {
        return 
    }
    fs.mkdirSync( name)
    return 
}

getIMGURL(10, "狗")