
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def play(test_loader):

    examples = iter(test_loader)
    example_data, example_targets = next(examples)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


    def animate(i):
        ax.clear()
        xlist = []
        ylist = []
        zlist = []
        hand = example_data[0][i]
        for index in range(len(hand) // 3):
            x = hand[index * 3].item()
            y = hand[index * 3 + 1].item()
            z = hand[index * 3 + 2].item()
            xlist.append(x)
            ylist.append(y)
            zlist.append(z)
            ax.scatter(x, y, z)

        # thumb
        ax.plot3D([xlist[0], xlist[1]], [ylist[0], ylist[1]], [zlist[0], zlist[1]], "gray")
        ax.plot3D([xlist[1], xlist[2]], [ylist[1], ylist[2]], [zlist[1], zlist[2]], "gray")
        ax.plot3D([xlist[2], xlist[3]], [ylist[2], ylist[3]], [zlist[2], zlist[3]], "gray")
        ax.plot3D([xlist[3], xlist[4]], [ylist[3], ylist[4]], [zlist[3], zlist[4]], "gray")
        # # index
        ax.plot3D([xlist[0], xlist[5]], [ylist[0], ylist[5]], [zlist[0], zlist[5]], "gray")
        ax.plot3D([xlist[5], xlist[6]], [ylist[5], ylist[6]], [zlist[5], zlist[6]], "gray")
        ax.plot3D([xlist[6], xlist[7]], [ylist[6], ylist[7]], [zlist[6], zlist[7]], "gray")
        ax.plot3D([xlist[7], xlist[8]], [ylist[7], ylist[8]], [zlist[7], zlist[8]], "gray")
        # # middle
        ax.plot3D(
            [xlist[9], xlist[10]], [ylist[9], ylist[10]], [zlist[9], zlist[10]], "gray"
        )
        ax.plot3D(
            [xlist[10], xlist[11]], [ylist[10], ylist[11]], [zlist[10], zlist[11]], "gray"
        )
        ax.plot3D(
            [xlist[11], xlist[12]], [ylist[11], ylist[12]], [zlist[11], zlist[12]], "gray"
        )
        # # ring
        ax.plot3D(
            [xlist[13], xlist[14]], [ylist[13], ylist[14]], [zlist[13], zlist[14]], "gray"
        )
        ax.plot3D(
            [xlist[14], xlist[15]], [ylist[14], ylist[15]], [zlist[14], zlist[15]], "gray"
        )
        ax.plot3D(
            [xlist[15], xlist[16]], [ylist[15], ylist[16]], [zlist[15], zlist[16]], "gray"
        )
        # # pinky
        ax.plot3D(
            [xlist[0], xlist[17]], [ylist[0], ylist[17]], [zlist[0], zlist[17]], "gray"
        )
        ax.plot3D(
            [xlist[17], xlist[18]], [ylist[17], ylist[18]], [zlist[17], zlist[18]], "gray"
        )
        ax.plot3D(
            [xlist[18], xlist[19]], [ylist[18], ylist[19]], [zlist[18], zlist[19]], "gray"
        )
        ax.plot3D(
            [xlist[19], xlist[20]], [ylist[19], ylist[20]], [zlist[19], zlist[20]], "gray"
        )
        # # knuckle
        ax.plot3D([xlist[5], xlist[9]], [ylist[5], ylist[9]], [zlist[5], zlist[9]], "gray")
        ax.plot3D(
            [xlist[9], xlist[13]], [ylist[9], ylist[13]], [zlist[9], zlist[13]], "gray"
        )
        ax.plot3D(
            [xlist[13], xlist[17]], [ylist[13], ylist[17]], [zlist[13], zlist[17]], "gray"
        )


    ani = animation.FuncAnimation(
        fig, animate, frames=len(example_data[0]), interval=100, repeat=True
    )
    plt.show()