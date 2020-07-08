{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: '/root/mlops/cnn.py'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-62ab00efd188>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0md\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mopen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"/root/mlops/cnn.py\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"r\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mcontents\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0md\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mreadlines\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0md\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mclose\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: '/root/mlops/cnn.py'"
     ]
    }
   ],
   "source": [
    "d = open(\"/root/mlops/cnn.py\", \"r\")\n",
    "contents = d.readlines()\n",
    "d.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "conv_count =0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "ename": "IndentationError",
     "evalue": "expected an indented block (<ipython-input-3-70ec2e1c11e8>, line 2)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"<ipython-input-3-70ec2e1c11e8>\"\u001b[1;36m, line \u001b[1;32m2\u001b[0m\n\u001b[1;33m    if x == \"model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu')) \\n\":\u001b[0m\n\u001b[1;37m     ^\u001b[0m\n\u001b[1;31mIndentationError\u001b[0m\u001b[1;31m:\u001b[0m expected an indented block\n"
     ]
    }
   ],
   "source": [
    "for x in contents:\n",
    "if x == \"model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu')) \\n\":\n",
    "    conv_count = conv_count+1 \n",
    "\n",
    "print(conv_count)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "hidden_count =0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'contents' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-5-58f4c6f09536>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[1;32mfor\u001b[0m \u001b[0mx\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mcontents\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mx\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m\"model.add(Dense(units=128, activation='relu'))\\n\"\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m         \u001b[0mhidden_count\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mhidden_count\u001b[0m\u001b[1;33m+\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mhidden_count\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'contents' is not defined"
     ]
    }
   ],
   "source": [
    "for x in contents:\n",
    "    if x == \"model.add(Dense(units=128, activation='relu'))\\n\":\n",
    "        hidden_count = hidden_count+1 \n",
    "\n",
    "print(hidden_count)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Add a Convolutional layer\n"
     ]
    },
    {
     "ename": "NameError",
     "evalue": "name 'contents' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-6-f3a0c21c3b1e>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;32mif\u001b[0m \u001b[0mconv_count\u001b[0m \u001b[1;33m<\u001b[0m \u001b[1;36m2\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m     \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Add a Convolutional layer\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m     \u001b[0mcontents\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0minsert\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m12\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu')) \\n\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m     \u001b[0mcontents\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0minsert\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m13\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu')) \\n\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m     \u001b[0mcontents\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0minsert\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m14\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"model.add(MaxPooling2D(pool_size=(2, 2))) \\n\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mNameError\u001b[0m: name 'contents' is not defined"
     ]
    }
   ],
   "source": [
    "if conv_count < 2:\n",
    "    print(\"Add a Convolutional layer\")\t\n",
    "    contents.insert(12, \"model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu')) \\n\")\n",
    "    contents.insert(13, \"model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu')) \\n\")\n",
    "    contents.insert(14, \"model.add(MaxPooling2D(pool_size=(2, 2))) \\n\")\n",
    "    num = int(contents[34][9])\n",
    "    num = num + 2\n",
    "    str_num = str(num)\n",
    "    contents[34] = \"epochs = \"+str_num+\"\\n\"\n",
    "    print(contents[34])\n",
    "    f = open(\"/root/mlops/cnn.py\", \"w\")\n",
    "    contents = \"\".join(contents)\n",
    "    f.write(contents)\n",
    "    f.close()\n",
    "elif conv_count == 2:\n",
    "    print(\"Convolution layers are correct\")\n",
    "    if hidden_count < 2:\n",
    "        print(\"Add a Hidden layer\")\n",
    "        contents.insert(19, \"model.add(Dense(units=128, activation='relu'))\\n\")\n",
    "        contents.insert(20, \"model.add(Dense(units=64, activation='relu'))\\n\")\n",
    "        num = int(contents[36][9])\n",
    "        num = num + 2\n",
    "        str_num = str(num)\n",
    "        contents[36] = \"epochs = \"+str_num+\"\\n\"\n",
    "        print(contents[36])\n",
    "        f = open(\"/root/mlops/cnn.py\", \"w\")\n",
    "        contents = \"\".join(contents)\n",
    "        f.write(contents)\n",
    "        f.close()\n",
    "    elif hidden_count == 2:\n",
    "        print(\"Hidden layers are correct\")\n",
    "        print(\"Increase Epochs\")\n",
    "        num = int(contents[36][9])\n",
    "        num = num + 2\n",
    "        str_num = str(num)\n",
    "        contents[36] = \"epochs = \"+str_num+\"\\n\"\n",
    "        print(contents[36])\n",
    "        f = open(\"/root/mlops/cnn.py\", \"w\")\n",
    "        contents = \"\".join(contents)\n",
    "        f.write(contents)\n",
    "        f.close()\n",
    "else:\n",
    "    print(\"Close the program\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
