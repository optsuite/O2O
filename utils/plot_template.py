import matplotlib.pyplot as plt
import torch


def plot_template(y_record, x_label, y_label, save_path, lower_x, upper_x, lower_y, upper_y, plot_type, x_record = None, line_style = '-', total_marker = 16, set_lim = False, save = True, plot_threshold = False, threshold = 3e-4):
    markers = ["o", "D", "s", "X", "P", "p", "<", ">", "v", "^", "*"]
    colors = ['b', 'g', 'c', 'm', 'r', 'y', 'k', 'w']

    plt.figure(figsize=(8, 6), dpi=300)
    plt.rc('font', family='DejaVu Sans')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    if set_lim:
        if lower_x:
            plt.xlim(left = lower_x)
        if upper_x:
            plt.xlim(right = upper_x)
        if lower_y:
            plt.ylim(bottom = lower_y)
        if upper_y:
            plt.ylim(top = upper_y)

    # it_max = len(y_record[list(y_record.keys())[0]])

    if plot_type == 'loglog':
        plt_func = plt.loglog
    elif plot_type == 'semilogy':
        plt_func = plt.semilogy
    elif plot_type == 'plot':
        plt_func = plt.plot
    elif plot_type == 'scatter':
        plt_func = plt.scatter
    else:
        raise UserWarning("Wrong plot type name!")

    if x_record == None:
        for (name, y_list), marker, color in zip(y_record.items(), markers, colors):
            plt_func(
                y_list,
                marker+line_style,
                label=name,
                markevery=max(1, len(y_list) // total_marker),
                # markevery=60,
                markersize=12.0,
                markerfacecolor='white',
                color=color
            )
    else:
        for (name, x_list), (_, y_list), marker, color in zip(x_record.items(), y_record.items(), markers, colors):
            plt_func(
                x_list,
                y_list,
                marker+line_style,
                label=name,
                markevery=max(1, len(y_list) // total_marker),
                # markevery=60,
                markersize=12.0,
                markerfacecolor='white',
                color=color
            )

    # plt.legend(fontsize=16)
    plt.legend(framealpha=1, frameon=True, fontsize=16)
    plt.minorticks_off()

    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.grid(axis="both", color=".8", linestyle="-", linewidth=1)
    plt.tight_layout()

    if plot_threshold:
        plt.axhline(threshold, color='r', linestyle='--')
    if save:
        # plt.savefig(save_path + ".png", dpi=600)
        # plt.show()
        plt.savefig(save_path + ".pdf", dpi=600)
    else:
        fig = plt.gcf()
        return fig
    
    plt.close()