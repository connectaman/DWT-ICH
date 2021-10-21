import dwt_ich
import matplotlib.pyplot as plt

img_p =  r'D:\Projects\AI Projects\ICH-DWT\004.png'   

brain = dwt_ich.ICH()
res = brain.predict(img_p)
plt.imshow(res,cmap='gray')
plt.show()