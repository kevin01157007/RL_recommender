import matplotlib.pyplot as plt
def plot(test_prec_hist, test_rec_hist, test_ndcg_hist, K, timesteps):
    plt.figure(); plt.plot(timesteps, test_prec_hist); plt.xlabel('Epoch'); plt.ylabel(f'Precision@{K}'); plt.title('Test Precision')
    plt.figure(); plt.plot(timesteps, test_rec_hist); plt.xlabel('Epoch'); plt.ylabel(f'Recall@{K}'); plt.title('Test Recall')
    plt.figure(); plt.plot(timesteps, test_ndcg_hist); plt.xlabel('Epoch'); plt.ylabel(f'NDCG@{K}'); plt.title('Test NDCG')
    plt.show()