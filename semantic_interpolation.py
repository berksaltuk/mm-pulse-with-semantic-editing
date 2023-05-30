from manipulator import linear_interpolate
from tqdm import tqdm

def semantic_interpolation(latent_codes, boundary):
    # Select a sample ID to perform interpolation
    sample_id = 0  # Update this as per your requirement

    total_num = len(latent_codes)
    total_interpolations = []

    for sample_id in tqdm(range(total_num), leave=False):
      interpolations = linear_interpolate(latent_codes[sample_id:sample_id + 1],
                                        boundary,
                                        -3.0,
                                        3.0,
                                        10)
      total_interpolations.append(interpolations)

    return total_interpolations