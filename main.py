import numpy as np
import os
import sys
from mosi_utils_anim.utilities import load_json_file, write_to_json_file
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models.fullBody_pose_encoder import FullBodyPoseEncoder
import tensorflow as tf 
from tensorflow.keras import Sequential, layers, Model, optimizers, losses
EPS = 1e-6


def PCA_on_preprocessed_data():
    data_folder = r'data\2 - Preprocessing'
    capturing_systems = ['art', 'capturystudio', 'optitrack', 'vicon']
    motion_types = ['l', 'r']
    npc = 10
    results = {}
    n_frames = 65

    for sys in capturing_systems:
        for type in motion_types:
            training_data = []
            json_data = load_json_file(os.path.join(data_folder, '_'.join(['walk', sys, type, 'featureVector.json'])))
            for filename, data in json_data.items():
                training_data.append(np.ravel([item for sublist in data for subsublist in sublist for item in subsublist]))
            training_data = np.asarray(training_data)
            pca = PCA(n_components=npc)
            pca.fit(training_data)
            # explained_variance = np.sum(pca.explained_variance_ratio_)
            projection = pca.transform(training_data)
            backprojection = pca.inverse_transform(projection)
            origin_var = np.sum(training_data.var(axis=0))
            recon_var = np.sum(backprojection.var(axis=0))
            explained_variance = recon_var / origin_var
            mse = np.sum((training_data - backprojection)**2) / n_frames
            print("mean square error: ", mse)
            results['_'.join([sys, type])] = {'explained_variance': explained_variance,
                                              'mse': mse}
    print(results)
    write_to_json_file('pca_preprocessed_data.json', results)


def bar_plot_pca():
    result_data = load_json_file('pca_preprocessed_data.json')
    print(result_data.keys())
    labels = ['capturystudio', 'optitrack', 'vicon', 'art']
    left_var = []
    right_var = []
    left_mse = []
    right_mse = []
    for sys in labels:
        left_var.append(result_data['_'.join([sys, 'l'])]['explained_variance'])
        left_mse.append(result_data['_'.join([sys, 'l'])]['mse'])
        right_var.append(result_data['_'.join([sys, 'r'])]['explained_variance'])
        right_mse.append(result_data['_'.join([sys, 'r'])]['mse'])

    width = 0.35
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    # rects_var_l = ax.bar(x - width/2, left_var, width, label='left step')
    # rects_var_r = ax.bar(x + width/2, right_var, width, label='right step')
    rects_var_l = ax.bar(x - width/2, left_mse, width, label='left step')
    rects_var_r = ax.bar(x + width/2, right_mse, width, label='right step')
    # ax.set_ylabel("explained variance")
    ax.set_ylabel('mse')
    ax.set_title("PCA on quaternion frames")
    ax.set_xticks(x)
    # plt.ylim(0.85, 1.0)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')   

    autolabel(rects_var_l)
    autolabel(rects_var_r)
    fig.tight_layout()

    plt.show()    


def train_autoencoder_on_preprocessed_data():
    data_folder = r'data\2 - Preprocessing'
    capturing_systems = ['art', 'capturystudio', 'optitrack', 'vicon']
    motion_types = ['l', 'r']
    npc = 10
    n_frames = 65

    ##### MODEL TRAINING
    for sys in capturing_systems:
        for type in motion_types:
            training_data = []
            json_data = load_json_file(os.path.join(data_folder, '_'.join(['walk', sys, type, 'featureVector.json'])))
            for filename, data in json_data.items():
                training_data.append(np.ravel([item for sublist in data for subsublist in sublist for item in subsublist]))
            training_data = np.asarray(training_data)
            mean = training_data.mean(axis=0)[np.newaxis, :]
            std = training_data.std(axis=0)[np.newaxis, :]
            std[std < EPS] = EPS
            normalized_data = (training_data - mean) / std

            dropout_rate = 0.05
            epochs = 1000
            batchsize = 1
            learning_rate = 1e-4
            pose_encoder = FullBodyPoseEncoder(output_dim=normalized_data.shape[1], dropout_rate=dropout_rate, npc=npc)
            pose_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                                loss='mse')            

            pose_encoder.fit(normalized_data, normalized_data, epochs=epochs, batch_size=batchsize)
            save_path = os.path.join(r'data/models', sys, type)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pose_encoder.save_weights(os.path.join(save_path, '_'.join([str(dropout_rate), str(epochs), str(learning_rate), '.ckpt'])))



def evaluate_autoencoder_on_preprocessed_data():
    data_folder = r'data\2 - Preprocessing'
    capturing_systems = ['art', 'capturystudio', 'optitrack', 'vicon']
    motion_types = ['l', 'r']
    npc = 10
    n_frames = 65
    res = {}    
    for sys in capturing_systems:
        for type in motion_types:
            training_data = []
            json_data = load_json_file(os.path.join(data_folder, '_'.join(['walk', sys, type, 'featureVector.json'])))
            for filename, data in json_data.items():
                training_data.append(np.ravel([item for sublist in data for subsublist in sublist for item in subsublist]))
            training_data = np.asarray(training_data)
            mean = training_data.mean(axis=0)[np.newaxis, :]
            std = training_data.std(axis=0)[np.newaxis, :]
            std[std < EPS] = EPS
            normalized_data = (training_data - mean) / std
            #### load model
            dropout_rate = 0.05
            epochs = 1000
            batchsize = 1
            learning_rate = 1e-4
            pose_encoder = FullBodyPoseEncoder(output_dim=normalized_data.shape[1], dropout_rate=dropout_rate, npc=npc)            
            model_path = os.path.join(r'data/models', sys, type, '_'.join([str(dropout_rate), str(epochs), str(learning_rate), '.ckpt']))
            pose_encoder.compile(optimizer=optimizers.Adam(learning_rate),
                                loss='mse')
            pose_encoder.build(input_shape=normalized_data.shape)             
            pose_encoder.load_weights(model_path)

            reconstructed_motions = np.asarray(pose_encoder(normalized_data))
            reconstructed_motions = reconstructed_motions * std + mean
            origin_var = np.sum(training_data.var(axis=0))
            recon_var = np.sum(reconstructed_motions.var(axis=0))
            ratio = recon_var / origin_var

            mse = np.sum((training_data - reconstructed_motions) ** 2) / n_frames
            print("mean square error: ", mse)
            res['_'.join([sys, type])] = {'explained_variance': ratio,
                                          'mse': mse}
    print(res)
    write_to_json_file('autoencoder_preprocessed_data.json', res)



def bar_plot_results_autoencoder():
    result_data = load_json_file('autoencoder_preprocessed_data.json')
    print(result_data.keys())
    labels = ['capturystudio', 'optitrack', 'vicon', 'art']
    left_var = []
    right_var = []
    left_mse = []
    right_mse = []
    for sys in labels:
        left_var.append(result_data['_'.join([sys, 'l'])]['explained_variance'])
        left_mse.append(result_data['_'.join([sys, 'l'])]['mse'])
        right_var.append(result_data['_'.join([sys, 'r'])]['explained_variance'])
        right_mse.append(result_data['_'.join([sys, 'r'])]['mse'])

    width = 0.35
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    # rects_var_l = ax.bar(x - width/2, left_var, width, label='left step')
    # rects_var_r = ax.bar(x + width/2, right_var, width, label='right step')
    rects_var_l = ax.bar(x - width/2, left_mse, width, label='left step')
    rects_var_r = ax.bar(x + width/2, right_mse, width, label='right step')
    # ax.set_ylabel("explained variance")
    ax.set_ylabel("mse")
    ax.set_title("VAE on quaternion frames")
    ax.set_xticks(x)
    # plt.ylim(0.85, 1.2)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')   

    autolabel(rects_var_l)
    autolabel(rects_var_r)
    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    PCA_on_preprocessed_data()
    bar_plot_pca()
    # train_autoencoder_on_preprocessed_data()
    # evaluate_autoencoder_on_preprocessed_data()