from PIL import Image
import os

directory1 = r'C:\Users\KIIT\Documents\College stuffs\Projects\Realtime mask detector\Realtime-mask-detector\dataset\with_mask'
c=1
for filename in os.listdir(directory1):
    if filename.endswith(".png"):
        im = Image.open(filename)
        name='img'+str(c)+'.jpg'
        rgb_im = im.convert('RGB')
        rgb_im.save(name)
        c+=1
        print(os.path.join(directory1, filename))
        continue
    else:
        continue

directory2 = r'C:\Users\KIIT\Documents\College stuffs\Projects\Realtime mask detector\Realtime-mask-detector\dataset\without_mask'
c=1
for filename in os.listdir(directory2):
    if filename.endswith(".png"):
        im = Image.open(filename)
        name='img'+str(c)+'.jpg'
        rgb_im = im.convert('RGB')
        rgb_im.save(name)
        c+=1
        print(os.path.join(directory2, filename))
        continue
    else:
        continue    