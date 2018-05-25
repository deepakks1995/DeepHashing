import numpy as np 
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import os

plotting_data = []
plotting_type = []
plotting_color = []
legend_label = []

def load_data(data_path, similar_samples):
    data = []
    class_folders = os.listdir(os.getcwd() + "/" + data_path)
    for cls in class_folders:
        counter = 0
        listings = os.listdir(os.getcwd() + "/" + data_path + "/" + cls)
        for file_index in range(len(listings)):
            if counter == 0:
                secondary_list = []
            img = image.load_img(os.getcwd() + "/" + data_path + "/" + cls + "/"  + listings[file_index])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            secondary_list.append(img)
            counter = (counter + 1)%similar_samples
            if counter == 0:
                data.append(secondary_list)
    return data

def split_train_test_data(percentage, data_path, test_path):
    load_data = []
    load_data_labels = []
    class_folders = os.listdir(os.getcwd() + "/" + data_path)
    class_weights = {}
    total_samples = 0
    file_paths = []
    for cls in class_folders:
        if len(cls) <= 4 :
            label = int(cls[len(cls)-1]) - 1
        else:
            label = int(cls[3])*10 + int(cls[3]) - 1
        # label = int(cls[3]) - 1
        listings = os.listdir(os.getcwd() + "/" + data_path + "/" + cls)
        class_weights[label] = len(listings)
        total_samples = total_samples + len(listings)
        one_hot_encoding = np.zeros(len(class_folders))
        one_hot_encoding[label] = 1
        # one_hot_encoding_list = [one_hot_encoding for _ in range(len(listings))]
        # i = 0
        for file in listings:
            img = image.load_img(os.getcwd() + "/" + data_path + "/" + cls + "/"  + file)
            x = image.img_to_array(img)
            load_data.append(x)
            load_data_labels.append(one_hot_encoding)
            file_paths.append(data_path + "/" + cls + "/"  + file)
            # i += 1
    for cls in range(len(class_folders)):
        print (' No. of samples:',class_weights[cls], )
        class_weights[cls] = float(class_weights[cls]) / total_samples
        print (' Class Weights:', class_weights[cls] )
    shuff_indices = [_ for _ in range(len(load_data))]
    random.shuffle(shuff_indices)
    train_data = []
    train_data_labels = []
    test_data = []
    test_data_labels = []
    count = 0
    test_files = []
    for i in shuff_indices:
        if count < int(percentage*len(load_data)):
            test_data.append(load_data[i])
            test_data_labels.append(load_data_labels[i])
            test_files.append(file_paths[i])
        else:
            train_data.append(load_data[i])
            train_data_labels.append(load_data_labels[i])
        count = count + 1
    # save_test_data(test_data, data_path, test_path)
    return train_data, test_data, train_data_labels, test_data_labels, class_weights, test_files


def save_test_data(test_data, data_path, test_path):
    for i in range(len(test_data)):
        img = Image.fromarray(test_data[i].astype('uint8'))
        img.save(os.getcwd() + "/" + test_path + "/" + str(i), "JPEG")


'''
*	This function is to add data to a global variable
*	which is used to plot in future
*	author: Deepak
'''
def add_plotting_data(x,str,color, legend):
    global plotting_data, plotting_type, plotting_color
    plotting_data.append(x)
    plotting_type.append(str)
    plotting_color.append(color)
    legend_label.append(legend)
    return True

'''
*	This function is to plot curves with different types
*	author: Deepak
'''
def plot_curve(str,axis, axis_label):
    global plotting_data, plotting_type, plotting_color, legend_label
    plt.ioff()
    fig = plt.figure(str)
    ax = fig.add_subplot(111)
    plt.axis(axis) if isinstance(axis, list) else True
    lines = []
    for i in range(len(plotting_data)):
        x = []
        y = []
        x = [itr[0] for itr in plotting_data[i]]
        y = [itr[1] for itr in plotting_data[i]]
        if plotting_type[i] != 'None':
            line, = ax.plot(x,y,plotting_type[i], color=plotting_color[i], label=legend_label[i])
        else:
            line, = ax.plot(x,y,color=plotting_color[i], label= legend_label[i])
        lines.append(line)
    ax.set_xlabel(axis_label[0])
    ax.set_ylabel(axis_label[1])
    fig.legend(lines, legend_label)
    fig.savefig("Output/" + str + ".png")
    plt.close(fig)
    plotting_data = []
    plotting_color = []
    plotting_type = []
    legend_label = []
    return True



if __name__=='__main__':
    print ('You are in the Wrong file')