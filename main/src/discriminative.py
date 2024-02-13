import numpy as np


def discriminative(train_data, train_labels, test_data):
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

    return predictions