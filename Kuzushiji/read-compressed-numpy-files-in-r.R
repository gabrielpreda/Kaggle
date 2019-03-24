library(reticulate)
np <- import("numpy")

#read train and test
npz_train_imgs <- np$load('../input/train-imgs.npz')
npz_test_imgs <- np$load('../input/test-imgs.npz')
npz_train_labels <- np$load('../input/train-labels.npz')

#extract the data from compressed numpy arrays
train_images = npz_train_imgs$f[["arr_0"]]
test_images = npz_test_imgs$f[["arr_0"]]
train_labels = npz_train_labels$f[["arr_0"]]

#check the shape of the matrices and vector
dim(train_images)
dim(test_images)
dim(train_labels)

#plot the labels
barplot(table(train_labels), col=rainbow(10, 0.5), main="n Digits in Train")