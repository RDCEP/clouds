
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_cos(encs):
    """ Originally from 902_checkLatentRep
    """
    x = encs[0].ravel().reshape(1, -1)
    cos_sim = []
    for i in range(encs.shape[0]):
        cos_sim.append(cosine_similarity(x, encs[i].ravel().reshape(1, -1)))
    return np.squeeze(np.array(cos_sim))


def load_pkl(filename):
  """ https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
  """
  with open(filename, 'rb') as handle:
    data = pickle.load(handle)
  return data


def run_main(filename,npatches=100,copy_size=12):
    """
        npatches: number of original patches
        copy_size: number of replicated patches
    """
    decs_dict = load_pkl(filename)

    decs_results = {}
    for ikey in decs_dict.keys():
      print(" Compute Cosine Sim. for Layer {} \n".format(ikey))
      decs = decs_dict[ikey]
      tmp_cossim_list = []
      for i in range(npatches):
          tmp = compute_cos(decs[i*copy_size:(i+1)*copy_size])
          tmp_cossim_list.append(tmp)
      decs_results[ikey] = tmp_cossim_list
    
    #with open('decoder_cossim_results_100.pkl', 'wb') as handle:
    #with open(f'encoder_cossim_results_{npatches}.pkl', 'wb') as handle:
    #with open(f'nri_encoder_cossim_results_{npatches}.pkl', 'wb') as handle:
    with open(f'nri_decoder_cossim_results_{npatches}.pkl', 'wb') as handle:
      pickle.dump(decs_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("NORMAL END", flush=True)
      

if __name__ == "__main__":
  #run_main(filename="./decoder_results_100.pkl")
  run_main(filename="./nri_decoder_results_10.pkl", npatches=10)
