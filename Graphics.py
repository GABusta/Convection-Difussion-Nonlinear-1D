import matplotlib.pyplot as plt

def plotting(OP1, OP2, OP3, tp):

    # --- Figure 1 ---
    plt.figure(1)
    for i in range(0, tp - 1, 1):
        plt.plot(OP1[:, i])
    plt.xlabel('Distance [m]')
    plt.ylabel('theta')
    plt.title('Exer.2 - Time steps vs. Theta (PICARD)')
    plt.grid(True)
    plt.show()

    # --- Figure 2 ---
    plt.figure(2)
    for i in range(0, tp - 1, 1):
        plt.plot(OP1[:, i])
    plt.xlabel('Distance [m]')
    plt.ylabel('theta')
    plt.title('Exer.2 - Time steps vs. Theta (Newton Raphson)')
    plt.grid(True)
    plt.show()

    # --- Figure 3 ---
    plt.figure(3)
    for i in range(0, tp - 1, 1):
        plt.plot(OP1[:, i])
    plt.xlabel('Distance [m]')
    plt.ylabel('theta')
    plt.title('Exer.3 - Time steps vs. Theta')
    plt.grid(True)
    plt.show()

    a=1
    return a


