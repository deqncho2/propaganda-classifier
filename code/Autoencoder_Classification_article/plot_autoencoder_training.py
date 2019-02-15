import matplotlib.pyplot as plt 
import pickle


pickleFile = open('autoencoder_loss.pkl', 'rb')
loss = pickle.load(pickleFile)

plt.figure(0)
plt.plot(loss)
plt.title("Autoencoder Training loss over 1000 epochs")
plt.xlabel("Epoch")
plt.ylabel('Root mean squared error (RMSE)')
plt.savefig('autoencoder_loss.pdf')
