import matplotlib.pyplot as plt

def histogram_array_generator(image):
    histogram_array = {}
    for index in range(256):
        histogram_array[index] = 0
    for row in image:
        for element in row:
            histogram_array[element] += 1
    return histogram_array

def plot_histogram(array, channel):
    plt.figure(figsize=(18,8))
    plt.bar(array.keys(), array.values())
    plt.ylabel('Number of occurrences')
    plt.xlabel('Pixel values')
    plt.title(channel)
    #plt.show()



def probability_distribution_generator(array, image_dimension):
    prob_dist_array = {}
    for index in range(256):
        prob_dist_array[index] = array[index]/image_dimension
    return prob_dist_array

