import numpy as np
import matplotlib.pyplot as plt
import os

train_data = np.load(f"main/dataset/My_Data/features.npy")
train_labels = np.load(f"main/dataset/My_Data/labels.npy")

blues_train = train_data[np.where(train_labels == "blues")]
classical_train = train_data[np.where(train_labels == "classical")]
country_train = train_data[np.where(train_labels == "country")]
disco_train = train_data[np.where(train_labels == "disco")]
hiphop_train = train_data[np.where(train_labels == "hiphop")]
jazz_train = train_data[np.where(train_labels == "jazz")]
metal_train = train_data[np.where(train_labels == "metal")]
pop_train = train_data[np.where(train_labels == "pop")]
reggae_train = train_data[np.where(train_labels == "reggae")]
rock_train = train_data[np.where(train_labels == "rock")]

# model building
mean_blues = np.mean(blues_train, axis=0)
cov_blues = np.cov(blues_train, rowvar=False)
mean_classical = np.mean(classical_train, axis=0)
cov_classical = np.cov(classical_train, rowvar=False) 
mean_country = np.mean(country_train, axis=0)
cov_country = np.cov(country_train, rowvar=False) 
mean_disco = np.mean(disco_train, axis=0)
cov_disco = np.cov(disco_train, rowvar=False) 
mean_hiphop = np.mean(hiphop_train, axis=0)
cov_hiphop = np.cov(hiphop_train, rowvar=False) 
mean_jazz = np.mean(jazz_train, axis=0)
cov_jazz = np.cov(jazz_train, rowvar=False)
mean_metal = np.mean(metal_train, axis=0)
cov_metal = np.cov(metal_train, rowvar=False) 
mean_pop = np.mean(pop_train, axis=0)
cov_pop = np.cov(pop_train, rowvar=False) 
mean_reggae = np.mean(reggae_train, axis=0)
cov_reggae = np.cov(reggae_train, rowvar=False) 
mean_rock = np.mean(rock_train, axis=0)
cov_rock = np.cov(rock_train, rowvar=False) 

fig, ax = plt.subplots(2,5)

im = ax[0][0].imshow(cov_blues, vmin=-.5, vmax=.5)
ax[0][0].set_title("Blues")
ax[0][1].imshow(cov_classical, vmin=-.5, vmax=.5)
ax[0][1].set_title("Classical")
ax[0][2].imshow(cov_country, vmin=-.5, vmax=.5)
ax[0][2].set_title("Country")
ax[0][3].imshow(cov_disco, vmin=-.5, vmax=.5)
ax[0][3].set_title("Disco")
ax[0][4].imshow(cov_hiphop, vmin=-.5, vmax=.5)
ax[0][4].set_title("Hiphop")
ax[1][0].imshow(cov_jazz, vmin=-.5, vmax=.5)
ax[1][0].set_title("Jazz")
ax[1][1].imshow(cov_metal, vmin=-.5, vmax=.5)
ax[1][1].set_title("Metal")
ax[1][2].imshow(cov_pop, vmin=-.5, vmax=.5)
ax[1][2].set_title("Pop")
ax[1][3].imshow(cov_reggae, vmin=-.5, vmax=.5)
ax[1][3].set_title("Reggae")
ax[1][4].imshow(cov_rock, vmin=-.5, vmax=.5)
ax[1][4].set_title("Rock")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
plt.colorbar(im, cax=cbar_ax)

plt.figure()
mean_blues = np.mean(train_data, axis=0)
cov_blues = np.cov(train_data, rowvar=False)

plt.imshow(cov_blues, vmin=-.5, vmax=.5)

plt.colorbar()


plt.figure()

lab_colors = ["blue"]*100 + ["dimgray"]*100 + ["forestgreen"]*100 + ["cyan"]*100 + ["darkred"]*100 + ["mediumpurple"]*99 + ["black"]*100 + ["magenta"]*100 + ["orange"]*100 + ["red"]*100 

plt.scatter(x=train_data[:,0], y=train_data[:,1], c=lab_colors)
plt.show()