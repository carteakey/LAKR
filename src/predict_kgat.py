
import torch
import numpy as np
from models.KGAT import KGAT
from parser.parser_kgat import *
from dataloader.loader_kgat import DataLoaderKGAT
from utils.model_helper import *
import logging

def load_pretrained_model(args, data):
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    return model

def get_top_k_recommendations(model, user_id, item_ids, k, device):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.LongTensor([user_id]).to(device)
        item_tensor = torch.LongTensor(item_ids).to(device)
        scores = model(user_tensor, item_tensor, mode='predict')
        scores = scores.squeeze().cpu().numpy()
        top_k_items = np.argsort(scores)[-k:][::-1]
        top_k_scores = scores[top_k_items]
    return top_k_items, top_k_scores

def recommend_for_user(args, user_id, interacted_items, k=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    data = DataLoaderKGAT(args, logging)
    
    # Load pre-trained model
    model = load_pretrained_model(args, data)
    model.to(device)
    
    # Get all item IDs
    all_items = set(range(data.n_items))
    
    # Remove items the user has already interacted with
    candidate_items = list(all_items - set(interacted_items))
    
    # Get top-k recommendations
    top_k_items, top_k_scores = get_top_k_recommendations(model, user_id, candidate_items, k, device)
    
    # Map item IDs back to original item IDs (if necessary)
    recommended_items = [candidate_items[i] for i in top_k_items]
    
    return list(zip(recommended_items, top_k_scores))

# Example usage
if __name__ == "__main__":
    args = parse_kgat_args()  # Make sure to import or define this function
    user_id = 0
    
    # Project Hail Mary - B08GB58KD5 - 71785
    # Conversations with God - 0399142789 - 116894
    # We Are Legion (We Are Bob) (Bobiverse) - 1680680587 - 10703
    # For We Are Many (Bobiverse) - 1680680595 - 86597
    # All These Worlds (Bobiverse) - 1680680609 - 44094
    # 
    interacted_items = [71785, 116894, 10703]
    # interacted_items = [37252, 62155, 156750, 144623, 82928, 34065, 22226, 90325, 108632, 115801, 189, 90590, 142239]
    k = 10
    
    recommendations = recommend_for_user(args, user_id, interacted_items, k)
    
    print(f"Top {k} recommendations for user {user_id}:")
    for item_id, score in recommendations:
        print(f"Item ID: {item_id}, Score: {score:.4f}")