import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics


def plot2d(features, means):
    '''
    plotting 2d decision boundaries (lines)
    '''
    data = np.load(f"../dataset/My_Data/pca_{features}.npy")

    # decision boundaries 
    w = []
    x = []
    for i in range(0, len(means)-1):
        w.append(means[i] - means[i+1])
        x.append(1/2* (means[i]+means[i+1])) # reduced formula since prior is always the same

    w.append(means[0] - means[len(means)-1])
    x.append(1/2* (means[0]+means[len(means)-1]))

    x_m = []
    z1 = []
    z2 = []
    z3 = []
    z4 = []
    z5 = []
    z6 = []
    z7 = []
    z8 = []
    z9 = []
    z10 = []
    for c in range(-10,15,1):
        for r in range(-2,2,1):
            x_m.append(np.array([c]))
            z1.append(w[0].T.dot(np.array([c,r])-x[0]))
            z2.append(w[1].T.dot(np.array([c,r])-x[1]))
            z3.append(w[2].T.dot(np.array([c,r])-x[2]))
            z4.append(w[3].T.dot(np.array([c,r])-x[3]))
            z5.append(w[4].T.dot(np.array([c,r])-x[4]))
            z6.append(w[5].T.dot(np.array([c,r])-x[5]))
            z7.append(w[6].T.dot(np.array([c,r])-x[6]))
            z8.append(w[7].T.dot(np.array([c,r])-x[7]))
            z9.append(w[8].T.dot(np.array([c,r])-x[8]))
            z10.append(w[9].T.dot(np.array([c,r])-x[9]))

    x_m = np.array(x_m)
    z1 = np.array(z1)
    z2 = np.array(z2)
    z3 = np.array(z3)
    z4 = np.array(z4)
    z5 = np.array(z5)
    z6 = np.array(z6)
    z7 = np.array(z7)
    z8 = np.array(z8)
    z9 = np.array(z9)
    z10 = np.array(z10)

    # scatter colors
    colors = ["blue"]*100 + ["dimgray"]*100 + ["forestgreen"]*100 + ["cyan"]*100 + ["darkred"]*100 + ["mediumpurple"]*99 + ["black"]*100 + ["magenta"]*100 + ["orange"]*100 + ["red"]*100 

    fig = plt.figure("Decision Boundaries - 2D")
    ax = fig.add_subplot()
    ax.scatter(data[:,0], data[:,1], c=colors)
    
    im1 = ax.plot(x_m[:,0], z1, "blue")[0]
    im2 = ax.plot(x_m[:,0], z2, "dimgray")[0]
    im3 = ax.plot(x_m[:,0], z3, "forestgreen")[0]
    im4 = ax.plot(x_m[:,0], z4, "cyan")[0]
    im5 = ax.plot(x_m[:,0], z5, "darkred")[0]
    im6 = ax.plot(x_m[:,0], z6, "mediumpurple")[0]
    im7 = ax.plot(x_m[:,0], z7, "black")[0]
    im8 = ax.plot(x_m[:,0], z8, "magenta")[0]
    im9 = ax.plot(x_m[:,0], z9, "orange")[0]
    im10 = ax.plot(x_m[:,0], z10, "red")[0]
    imgs = [im10, im1, im2, im3, im4, im5, im6, im7, im8, im9]

    def toggle_images(event):
        'toggle the visible state of the images'

        key = event.key
        if key in ['1','2','3','4','5','6','7','8','9','0']:
            imgs[int(key)].set_visible(not(imgs[int(key)].get_visible()))
        else:
            return
        
        plt.draw()

    plt.connect('key_press_event', toggle_images)

    ax.set_xlim((-20,20))
    ax.set_ylim((-20,20))

    plt.show()


def discriminative(train_data, train_labels, test_data, features):
    '''
    computes decision boundaries for each class

    :returns: predictions for test_data
    '''

    # class divided train data
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

    # compute discriminant functions
    g1x = []
    g2x = []
    g3x = []
    g4x = []
    g5x = []
    g6x = []
    g7x = []
    g8x = []
    g9x = []
    g10x = []
    num_features = train_data.shape[1]
    for i in range(test_data.shape[0]):
        g1x.append(-1/2*((test_data[i,:]-mean_blues).T).dot(np.linalg.inv(cov_blues)).dot((test_data[i,:]-mean_blues)) - (num_features/2)*np.log(2*np.pi)- 1/2 * np.log(np.linalg.det(cov_blues)) + np.log(1/10))
        g2x.append(-1/2*((test_data[i,:]-mean_classical).T).dot(np.linalg.inv(cov_classical)).dot((test_data[i,:]-mean_classical)) - (num_features/2)*np.log(2*np.pi)- 1/2 * np.log(np.linalg.det(cov_classical)) + np.log(1/10))
        g3x.append(-1/2*((test_data[i,:]-mean_country).T).dot(np.linalg.inv(cov_country)).dot((test_data[i,:]-mean_country)) - (num_features/2)*np.log(2*np.pi)- 1/2 * np.log(np.linalg.det(cov_country)) + np.log(1/10))
        g4x.append(-1/2*((test_data[i,:]-mean_disco).T).dot(np.linalg.inv(cov_disco)).dot((test_data[i,:]-mean_disco)) - (num_features/2)*np.log(2*np.pi)- 1/2 * np.log(np.linalg.det(cov_disco)) + np.log(1/10))
        g5x.append(-1/2*((test_data[i,:]-mean_hiphop).T).dot(np.linalg.inv(cov_hiphop)).dot((test_data[i,:]-mean_hiphop)) - (num_features/2)*np.log(2*np.pi)- 1/2 * np.log(np.linalg.det(cov_hiphop)) + np.log(1/10))
        g6x.append(-1/2*((test_data[i,:]-mean_jazz).T).dot(np.linalg.inv(cov_jazz)).dot((test_data[i,:]-mean_jazz)) - (num_features/2)*np.log(2*np.pi)- 1/2 * np.log(np.linalg.det(cov_jazz)) + np.log(1/10))
        g7x.append(-1/2*((test_data[i,:]-mean_metal).T).dot(np.linalg.inv(cov_metal)).dot((test_data[i,:]-mean_metal)) - (num_features/2)*np.log(2*np.pi)- 1/2 * np.log(np.linalg.det(cov_metal)) + np.log(1/10))
        g8x.append(-1/2*((test_data[i,:]-mean_pop).T).dot(np.linalg.inv(cov_pop)).dot((test_data[i,:]-mean_pop)) - (num_features/2)*np.log(2*np.pi)- 1/2 * np.log(np.linalg.det(cov_pop)) + np.log(1/10))
        g9x.append(-1/2*((test_data[i,:]-mean_reggae).T).dot(np.linalg.inv(cov_reggae)).dot((test_data[i,:]-mean_reggae)) - (num_features/2)*np.log(2*np.pi)- 1/2 * np.log(np.linalg.det(cov_reggae)) + np.log(1/10))
        g10x.append(-1/2*((test_data[i,:]-mean_rock).T).dot(np.linalg.inv(cov_rock)).dot((test_data[i,:]-mean_rock)) - (num_features/2)*np.log(2*np.pi)- 1/2 * np.log(np.linalg.det(cov_rock)) + np.log(1/10))

    gx = np.concatenate([np.array(g1x)[:,np.newaxis],np.array(g2x)[:,np.newaxis], np.array(g3x)[:,np.newaxis], np.array(g4x)[:,np.newaxis], np.array(g5x)[:,np.newaxis], np.array(g6x)[:,np.newaxis], np.array(g7x)[:,np.newaxis], np.array(g8x)[:,np.newaxis], np.array(g9x)[:,np.newaxis], np.array(g10x)[:,np.newaxis]],axis=-1)

    predictions = np.argmax(gx,axis=-1)

    means = np.array([mean_blues[:2], mean_classical[:2], mean_country[:2], mean_disco[:2], mean_hiphop[:2], mean_jazz[:2], mean_metal[:2], mean_pop[:2], mean_reggae[:2], mean_rock[:2]])
    plot2d(features, means)

    return predictions