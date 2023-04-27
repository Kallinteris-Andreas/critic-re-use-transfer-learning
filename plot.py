import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_directory", nargs='+', default='good_res/TD3_InvertedDoublePendulum-v4/')
    args = parser.parse_args()
    eval_paths = args.result_directory

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for eval_path in eval_paths:
        config = yaml.safe_load(open(eval_path + '/' + 'config.yaml', 'r'))

        # get data
        data = []
        for file in os.listdir(eval_path):
            if file.endswith('.csv'):
                print(file)
                data.append(np.genfromtxt(eval_path + "/" + file, delimiter=','))
        data = np.stack(data, axis=1)
        avg = np.average(data, axis=1)
        min_v = np.min(data, axis=1)
        max_v = np.max(data, axis=1)

        # plot
        x_axis = np.arange(start=30000, step=config['domain']['evaluation_frequency'], stop=2_000_000)

        ax.plot(x_axis, avg, label=str(config['domain']['factorization']))
        ax.fill_between(x_axis, min_v, max_v, alpha=0.2)

        ax.set_title("Average Regret over " + str(data.shape[1]) + " statistical runs, on " + config['domain']['name'])

    ax.set_ylabel("Return")
    ax.set_xlabel("Timestep")
    ax.legend()
    #plt.show()

    file_name = config['domain']['name']
    plt.savefig(file_name + ".eps", bbox_inches="tight")
    plt.savefig(file_name + ".png", bbox_inches="tight")
