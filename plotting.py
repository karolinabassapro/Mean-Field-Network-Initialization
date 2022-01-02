from main import *

plt.plot(x, accs_r, label="relu")
plt.plot(x, accs_t, label="tanh")
plt.plot(x, accs_ht, label="hard_tanh")
plt.title("Accuracies for Different Inits")
plt.legend()
plt.show()