import matplotlib.pyplot as plt

def plot_cutoff_vs_player(cutoff_history, player_series, player_name):
    plt.figure()

    plt.plot(cutoff_history, label='Cutoff')
    plt.axhline(player_series[player_name], linestyle='--', label=player_name)

    plt.xlabel('Simulation')
    plt.ylabel('Points')
    plt.legend()

    plt.title('Cutoff vs Player Expected Points')
    plt.show()