import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_directory", nargs='+', default='good_res/TD3_InvertedDoublePendulum-v4/')
    parser.add_argument("--mode", default="average")
    args = parser.parse_args()
    eval_paths = args.result_directory

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for eval_path in eval_paths:
        config = yaml.safe_load(open(eval_path + '/' + 'config.yaml', 'r'))

        # get data
        files = []
        data = []
        for file in os.listdir(eval_path):
            if file.endswith('.csv'):
                # print(file)
                files.append(file)
                data.append(np.genfromtxt(eval_path + "/" + file, delimiter=','))
        # process data
        data = np.stack(data, axis=1)
        avg = np.average(data, axis=1)
        min_v = np.min(data, axis=1)
        max_v = np.max(data, axis=1)
        print(f"the best of {eval_path} is: {files[np.argmax(np.max(data, axis=0))]}, with max = {np.max(data)}")

        # plot
        x_axis = np.arange(start=30000, step=config['domain']['evaluation_frequency'], stop=2_000_000)

        label = f"{config['domain']['factorization']}-{config['domain']['algo']}"
        linestyle = "solid"

        if config.get("other", None) is not None and config["other"]["load_Q"] and config["other"]["load_PI"]:
            label += " TLR"
            linestyle = 'dashed'
        elif config.get("other", None) is not None and config["other"]["load_Q"]:
            label += " TL"
            linestyle = 'dashdot'

        if args.mode == "average":
            ax.plot(x_axis, avg, label=label, linestyle=linestyle)
            ax.fill_between(x_axis, min_v, max_v, alpha=0.2)
        elif args.mode == "max":
            ax.plot(x_axis, max_v, label=label, linestyle=linestyle)

        ax.set_title(f"{args.mode} over {str(data.shape[1])} statistical runs, on {config['domain']['name']}")

    ax.set_ylabel("Return")
    ax.set_xlabel("Timestep")
    ax.legend()
    # plt.show()

    file_name = f"./figures/figure_{config['domain']['name']}_{args.mode}"
    fig.set_figwidth(16)
    fig.set_figheight(9)
    plt.savefig(file_name + ".eps", bbox_inches="tight")
    plt.savefig(file_name + ".png", bbox_inches="tight")
    plt.savefig(file_name + ".pdf", bbox_inches="tight")
