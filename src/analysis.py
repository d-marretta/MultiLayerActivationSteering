import torch
from collections import defaultdict

def compute_layer_scores(retain_acts, forget_acts, timesteps, top_k):
    """Compute discriminability, consistency, and combined scores for each layer at specified timesteps."""
    results = {}

    for timestep in timesteps :
        timestep_dict = {}
        for layer in retain_acts:
            P = retain_acts[layer][:, timestep-timesteps[0], :].float()  # Positive (N, D)
            N = forget_acts[layer][:, timestep-timesteps[0], :].float()  # Negative (N, D)
    
            if P.shape != N.shape:
                print(f'P shape and N shape differs in {layer}')
    
            n_samples = P.shape[0]
    
            all_acts = torch.cat([P, N], dim=0) # (2N, D)
            mu_l = all_acts.mean(dim=0, keepdim=True)  # (1, D)
            sigma_l = all_acts.std(dim=0, keepdim=True) + 1e-8 # (1, D)
    
            P_tilde = (P - mu_l) / sigma_l
            N_tilde = (N - mu_l) / sigma_l

            v_l = (N - P).mean(dim=0)
    
            
            # Calculate means of normalized data
            mu_pos = P_tilde.mean(dim=0) # (D)
            mu_neg = N_tilde.mean(dim=0) # (D)
    
            # Instead of creating (D, D) matrix, project means onto v_l
            proj_pos = torch.dot(mu_pos, v_l)
            proj_neg = torch.dot(mu_neg, v_l)
            
            # v^T Sb v = N * (proj_pos^2 + proj_neg^2)
            sb_val = n_samples * (proj_pos**2 + proj_neg**2)
    
            # Center the data class-wise
            P_centered = P_tilde - mu_pos.unsqueeze(0) # (N, D)
            N_centered = N_tilde - mu_neg.unsqueeze(0) # (N, D)
    
            # Instead of creating (D, D) covariance, project data onto v_l
            # This calculates the variance of the data along the direction of v_l
            p_proj = torch.mv(P_centered, v_l) # (N)
            n_proj = torch.mv(N_centered, v_l) # (N)
    
            sw_pos_val = torch.sum(p_proj**2)
            sw_neg_val = torch.sum(n_proj**2)
            sw_val = sw_pos_val + sw_neg_val
    
            
            D_l = (sb_val / (sb_val + sw_val + 1e-8)).item()
    
            pair_diffs = N_tilde - P_tilde # (N, D)
            dot_products = torch.mv(pair_diffs, v_l) # (N)
            pair_norms = torch.norm(pair_diffs, dim=1) 
            v_norm = torch.norm(v_l)
            
            cosine_sims = dot_products / (pair_norms * v_norm + 1e-8)
            C_l = cosine_sims.mean().item()
    
            S_l = D_l + C_l
    
            timestep_dict[layer] = {
                "score": S_l,
                "discriminability": D_l,
                "consistency": C_l
            }
            
            del P_tilde, N_tilde, all_acts, P_centered, N_centered
            torch.cuda.empty_cache()
            
        sorted_layers = sorted(timestep_dict.items(), key=lambda x: x[1]['score'], reverse=True)
        results[timestep] = [x for x in sorted_layers[:top_k]]
        
    return results


def compute_steering_vectors(forget_layers_act, retain_layers_act):
    """Compute steering vectors for each layer by taking the mean difference between forget and retain activations."""
    result = {}
    for (layer, X), (_, Y) in zip(forget_layers_act.items(), retain_layers_act.items()):
        result[layer] = (X - Y).mean(dim=0)

    return result


def contrastive_pca(X, Y, alpha: float = 1.0, threshold: float = 0.95, min_eigen: float = 0.5):
    """Perform contrastive PCA on the given activations X and Y, returning the principal components that capture the most discriminative variance."""
    X = X.float()
    Y = Y.float()

    X_center = X.mean(dim=0, keepdim=True)
    Xc = X - X_center
    Yc = Y - Y.mean(dim=0, keepdim=True)
    
    N, D = Xc.shape
    M, _ = Yc.shape

    S_X = (Xc.T @ Xc) / (N - 1)
    S_Y = (Yc.T @ Yc) / (M - 1)

    S_c = S_X - alpha * S_Y

    eigvals, eigvecs = torch.linalg.eigh(S_c)

    eigvals, idx = torch.sort(eigvals, descending=True)
    eigvecs = eigvecs[:, idx]

    mask = eigvals > min_eigen
    eigvals_pos = eigvals[mask]
    eigvecs_pos = eigvecs[:, mask]

    cum_var = torch.cumsum(eigvals_pos, dim=0) / eigvals_pos.sum()
    num_components = (cum_var < threshold).sum() + 1

    eigvals_selected = eigvals_pos[:num_components]
    eigvecs_selected = eigvecs_pos[:, :num_components]

    return eigvecs_selected.cpu(), eigvals_selected.cpu(), X_center.cpu()


def compute_principal_components(forget_acts, retain_acts, layers: list[str], min_subspace_dim: int = 5, alpha: float = 1.0, threshold: float = 0.95, min_eigen: float = 0.5):
    """Compute principal components for each layer and timestep by performing contrastive PCA on the forget and retain activations."""
    result = defaultdict(lambda: defaultdict(int))
    for (layer, X), (_, Y) in zip(forget_acts.items(), retain_acts.items()): 
        X = forget_acts[layer]
        Y = retain_acts[layer]

        assert X.shape == Y.shape

        for ts in range(X.size(1)):
            eigvecs, eigvals, mean = contrastive_pca(X[:, ts, :], Y[:, ts, :], alpha, threshold, min_eigen)

            if len(eigvals) >= min_subspace_dim: # we require at least 5 PC  
                
                result[layer][ts] = (eigvecs, mean)

    return result


