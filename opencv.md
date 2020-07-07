# opencv关键函数

## 颜色空间转换 cv2.cvtColor()

主要的两种转换方式:BGR ↔ Gray 和 BGR ↔ HSV
cv2.cvtColor(input_image,flag) ,flag为转换类型;

BGR ↔ Gray 的转换, flag: cv2.COLOR_BGR2GRAY。

BGR ↔ HSV 的转换, flag : cv2.COLOR_BGR2HSV。

在 HSV 颜色空间中要比在 BGR 空间中更容易表示一个特定颜色

## 统计直方图
通过直方图你可以对整幅图像的颜色分布有一个整体的了解,直方图的 x 轴是灰度值(0 到 255),y 轴是图片中具有同一个灰度值的
点的数目。
`cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])`

> 1. images: 原图像(图像格式为 uint8 或 float32)。当传入函数时应该用中括号 [] 括起来,例如:[img]。
> 2. channels: 同样需要用中括号括起来,它会告诉函数我们要统计那幅图 像的直方图。
> 如果输入图像是灰度图,它的值就是 [0];如果是彩色图像的话,传入的参数可以是 [0],[1],[2] 
> 它们分别对应着通道 B,G,R。
> 3. mask: 掩模图像。要统计整幅图像的直方图就把它设为 None。但是如
> 果你想统计图像某一部分的直方图的话,你就需要制作一个掩模图像,并
> 使用它。(后边有例子)
> 4. histSize:BIN 的数目。也应该用中括号括起来,例如:[256]。
> 5. ranges: 像素值范围,通常为 [0,256]

