import matplotlib.pyplot as plt
import pandas as pd
import sys


def swap(x):
    for i in range(len(x) // 2):
        x[i], x[len(x) - i - 1] = x[len(x) - i - 1], x[i]


# I changed the x and y-axis
def print_graph(x, y, l, swap=False):

    leg = ["1", "", "", "", "", "", "", "1k"]
    markers = ["o", "*", "^", "s"]

    # Plotting the lines
    for i in range(len(x)):
        plt.plot(x[i], y[i], label="Line 1", marker=markers[i % len(markers)])

        # legend in each points
        k = 0
        for px, py in zip(x[i], y[i]):
            # print(leg[k])
            label = leg[k]
            plt.annotate(
                label,  # this is the text
                (px, py),  # these are the coordinates to position the label
                textcoords="offset points",  # how to position the text
                xytext=(0, 10),  # distance from text to points (x,y)
                ha="center",
            )  # horizontal alignment can be left, right or center
            k += 1

    # Adding labels and title
    if swap:
        plt.ylabel("Execution time speedup")
        plt.xlabel("Tuning time speedup")
    else:
        plt.xlabel("Execution time speedup")
        plt.ylabel("Tuning time speedup")
    # plt.title('Plot with Three Lines')

    # Adding a legend
    plt.legend(l)

    # Displaying the plot
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        path_fig = sys.argv[1]
    else:
        print("error")

    df = pd.read_csv(path_fig, header=None)

    x, y, l = [], [], []
    for i in range(df.shape[0]):
        l.append(list(df.iloc[i, 0:1])[0])
        x.append(list(df.iloc[i, 2:10]))
        y.append(list(df.iloc[i, 10:]))

    is_swap = False

    if is_swap:
        for yl in y:
            swap(yl)
            print(yl)
        print(y)

    if is_swap:
        print_graph(y, x, l, is_swap)
    else:
        print_graph(x, y, l, is_swap)
