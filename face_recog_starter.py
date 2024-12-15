import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def show_orignal_images(pixels):
    fig, axes = plt.subplots(6, 10, figsize=(11, 7),
                             subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(64, 64), cmap='gray')
    plt.show()

def show_eigenfaces(pca):
    fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                             subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(64, 64), cmap='gray')
        ax.set_title("PC " + str(i+1))
    plt.show()


df = pd.read_csv("face_data.csv")

# print(df.head())
labels = df["target"]
pixels = df.drop(["target"], axis=1)

# show_orignal_images(pixels)


x_train, x_test, y_train, y_test = train_test_split(pixels, labels) 


pca = PCA(n_components=130).fit(x_train)


plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.show()

# show_eigenfaces(pca)


x_train_pca = pca.transform(x_train)

clf = SVC(kernel='rbf', C=1000, gamma=0.01)
clf = clf.fit(x_train_pca, y_train)

x_test_pca = pca.transform(x_test)

y_pred = clf.predict(x_test_pca)

print(classification_report(y_test, y_pred))


