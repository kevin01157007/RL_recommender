import matplotlib.pyplot as plt
def plot(test_prec_hist, test_rec_hist, test_ndcg_hist, K, timesteps):
    plt.figure(); 
    plt.plot(timesteps, test_prec_hist, marker='o'); 
    plt.xlabel('timesteps'); 
    plt.ylabel(f'Precision@{K}'); 
    plt.title('Test Precision'); 
    plt.xticks(timesteps)
    # plt.yticks(test_prec_hist)

    plt.figure(); 
    plt.plot(timesteps, test_rec_hist, marker='o'); 
    plt.xlabel('timesteps'); 
    plt.ylabel(f'Recall@{K}'); 
    plt.title('Test Recall'); 
    plt.xticks(timesteps)
    # plt.yticks(test_rec_hist)

    plt.figure(); 
    plt.plot(timesteps, test_ndcg_hist, marker='o'); 
    plt.xlabel('timesteps'); 
    plt.ylabel(f'NDCG@{K}'); 
    plt.title('Test NDCG'); 
    plt.xticks(timesteps)
    # plt.yticks(test_ndcg_hist)

    plt.show()