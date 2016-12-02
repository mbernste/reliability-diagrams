from optparse import OptionParser
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns


def reliability_diagrams(predictions, truths, confidences, bin_size=0.1, n_boot=1000):

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    accs = []

    # Compute empirical probability for each bin
    plot_x = []
    for conf_thresh in upper_bounds:
        acc, perc_pred, avg_conf = compute_accuracy(conf_thresh-bin_size, conf_thresh, confidences, predictions, truths)
        plot_x.append(avg_conf)
        accs.append(acc)

    # Produce error bars for each bin
    upper_bound_to_bootstrap_est = {x:[] for x in upper_bounds}
    for i in range(n_boot):

        # Generate bootstrap
        boot_strap_outcomes = []
        boot_strap_confs = random.sample(confidences, len(confidences))
        for samp_conf in boot_strap_confs:
            correct = 0
            if random.random() < samp_conf:
                correct = 1
            boot_strap_outcomes.append(correct)

        # Compute error frequency in each bin
        for upper_bound in upper_bounds:
            conf_thresh_upper = upper_bound
            conf_thresh_lower = upper_bound - bin_size

            filtered_tuples = [x for x in zip(boot_strap_outcomes, boot_strap_confs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
            correct = len([x for x in filtered_tuples if x[0] == 1])
            acc = float(correct) / len(filtered_tuples) if len(filtered_tuples) > 0 else 0

            upper_bound_to_bootstrap_est[upper_bound].append(acc)
       
    upper_bound_to_bootstrap_upper_bar = {}
    upper_bound_to_bootstrap_lower_bar = {}
    for upper_bound, freqs in upper_bound_to_bootstrap_est.iteritems():
        top_95_quintile_i = int(0.975 * len(freqs))
        lower_5_quintile_i = int(0.025 * len(freqs))

        upper_bar = sorted(freqs)[top_95_quintile_i]
        lower_bar = sorted(freqs)[lower_5_quintile_i]

        upper_bound_to_bootstrap_upper_bar[upper_bound] = upper_bar
        upper_bound_to_bootstrap_lower_bar[upper_bound] = lower_bar

    upper_bars = []
    lower_bars = []
    for i, upper_bound in enumerate(upper_bounds):
        if upper_bound_to_bootstrap_upper_bar[upper_bound] == 0:
            upper_bars.append(0)
            lower_bars.append(0)
        else:
            # The error bar arguments need to be the distance from the data point, not the y-value
            upper_bars.append(abs(plot_x[i] - upper_bound_to_bootstrap_upper_bar[upper_bound]))
            lower_bars.append(abs(plot_x[i] - upper_bound_to_bootstrap_lower_bar[upper_bound]))

    #print zip(upper_bars, lower_bars)

    sns.set(font_scale=2)
    fig, ax = plt.subplots()
    ax.errorbar(plot_x, plot_x, yerr=[lower_bars, upper_bars], label="Perfect classifier calibration")

    new_plot_x = []
    new_accs = []
    for i, bars in enumerate(zip(lower_bars, upper_bars)):
        if bars[0] == 0 and bars[1] == 0:
            continue
        new_plot_x.append(plot_x[i])
        new_accs.append(accs[i])
    
    ax.plot(new_plot_x, new_accs, '-o', label="Accuracy", color="red")
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    sns.plt.ylabel('Empirical probability')
    sns.plt.xlabel('Estimated probability')

    fig.set_size_inches(5, 5)
    #fig.savefig("reliability.tif", format='tif', bbox_inches='tight', dpi=1200)
    #fig.savefig("reliability.eps", format='eps', bbox_inches='tight', dpi=1200)

    plt.show()
        


def compute_accuracy(conf_thresh_lower, conf_thresh_upper, conf, pred, true):

    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])
        avg_conf = sum([x[2] for x in filtered_tuples]) / len(filtered_tuples)
        accuracy = float(correct)/len(filtered_tuples)
        perc_of_data = float(len(filtered_tuples))/len(conf)
        return accuracy, perc_of_data, avg_conf


def main():
    parser = OptionParser()
    (options, args) = parser.parse_args()

    results_f = args[0]
    with open(results_f, "r") as f:
        results = json.load(f)

    predictions = results["predicted"]
    truth = results["truth"]
    confidences = results["confidences"]
    reliability_diagrams(predictions, truth, confidences)

if __name__ == "__main__":
    main()
