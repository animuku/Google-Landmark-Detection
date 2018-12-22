import os
file = open("test_labels.txt","w")

cnt=0
for path, subdirs, files in os.walk('./train_images/'):
  for name in files:
    if cnt==0:
      cnt=cnt+1
      continue
    else:
      img_path = os.path.join(path,name)
      t1,t2=os.path.split(img_path)
      w,folder,correct_cat=t1.split('/')
      x,end=t2.split('.')
      if end=='jpg':
        if correct_cat == "0":
          label = 0 
        elif correct_cat == "1":
          label = 1
        else:
          label = 2
        temp = img_path + " " + str(label) + "\n"
        file.write(temp)
        cnt=cnt+1
print(cnt)