import pickle
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import os

class DishRecommender:
    def __init__(self):
        self.user_model = None
        self.item_model = None
        self.X = None
        self.dish_names = None
        self.dish_names_inv = None
        self.dish_mapper = None
        self.dish_inv_mapper = None
        self.user_mapper = None
        self.user_inv_mapper = None

        self.user_df = None
        self.dish_user_review_df = None
    
    def find_similar_dishes(self, dish_id=None, dish_name=None, n_similar_dishes=5):
        try:
            if dish_id is None and dish_name is None:
                raise ValueError("Must provide either dish_id or dish_name")
            
            if dish_id is None:
                dish_id = self.dish_names_inv.get(dish_name)
                if dish_id is None:
                    raise KeyError(f"Dish name '{dish_name}' not found")
                
            dish_id = str(dish_id)
            dish_ind = self.dish_mapper[dish_id]
            dish_vec = self.X[dish_ind].reshape(1, -1)

            user_model = NearestNeighbors(n_neighbors=n_similar_dishes, algorithm="brute", metric='cosine')
            user_model.fit(self.X)
            
            neighbor_indices = self.item_model.kneighbors(dish_vec, 
                                                return_distance=False).flatten()
            
            neighbor_indices = neighbor_indices[1:]
            similar_dishes = []
            for idx in neighbor_indices:
                dish_id = self.dish_inv_mapper[idx]
                dish_name = self.dish_names[dish_id]
                similar_dishes.append((dish_name, dish_id))
            
            return similar_dishes
            
        except Exception as e:
            print(f"Error finding similar dishes: {str(e)}")
            return []
    
    def find_similar_dishes_by_user(self, user_id=None, user_name=None, n_similar_users=5, n_dishes_for_user=5):
        if user_id is None and user_name is None:
            raise ValueError("Must provide either user_id or user_name")
        
        if user_id is None:
            user_matches = self.user_df[self.user_df['name'] == user_name]
            if len(user_matches) == 0:
                raise KeyError(f"User name '{user_name}' not found")
            user_id = str(user_matches.index[0])  
            
        user_id = str(user_id)
        
        if user_id not in self.user_mapper:
            raise KeyError(f"User ID '{user_id}' not found in ratings data")
        user_idx = self.user_mapper[user_id]
        
        X_user = self.X.T
        user_model = NearestNeighbors(n_neighbors=n_similar_users + 1, algorithm="brute", metric='cosine')
        user_model.fit(X_user)
        
        user_vector = X_user[user_idx].reshape(1, -1)
        distances, indices = user_model.kneighbors(user_vector, return_distance=True)
        similar_user_indices = indices.flatten()[1:]  
        
        recommended_dishes = []
        user_rated_dishes = set(self.dish_user_review_df[self.dish_user_review_df['user'] == user_id]['dish'])
        
        for similar_user_idx in similar_user_indices:
            similar_user_id = self.user_inv_mapper[similar_user_idx]
            similar_user_ratings = self.dish_user_review_df[self.dish_user_review_df['user'] == similar_user_id]
            
            potential_recommendations = similar_user_ratings[
                (similar_user_ratings['rating'] >= 4) & 
                (~similar_user_ratings['dish'].isin(user_rated_dishes))
            ]
            
            for _, row in potential_recommendations.iterrows():
                dish_id = row['dish']
                dish_name = self.dish_names[dish_id]
                if (dish_name, dish_id) not in recommended_dishes:
                    recommended_dishes.append((dish_name, dish_id))
                    
                if len(recommended_dishes) >= n_dishes_for_user:
                    break
                    
            if len(recommended_dishes) >= n_dishes_for_user:
                break
        
        print("Based on similar users' highly rated dishes:\n")
        
        return recommended_dishes[:n_dishes_for_user]

    def find_personalized_recommendations(self, user_id=None, user_name=None, n_similar_users=5, n_dishes_for_user=5, n_similar_per_dish=3, max_recommendations=15):
        try:
            initial_recommendations = self.find_similar_dishes_by_user(
                user_id=user_id,
                user_name=user_name,
                n_similar_users=n_similar_users,
                n_dishes_for_user=n_dishes_for_user
            )
            
            all_recommendations = []
            seen_dish_ids = set()
            
            for dish_name, dish_id in initial_recommendations:
                if dish_id not in seen_dish_ids:
                    all_recommendations.append((dish_name, dish_id))
                    seen_dish_ids.add(dish_id)
            
            for dish_name, dish_id in initial_recommendations:
                similar_dishes = self.find_similar_dishes(
                    dish_id=dish_id,
                    n_similar_dishes=n_similar_per_dish + 1  
                )
                
                for similar_dish_name, similar_dish_id in similar_dishes:
                    if similar_dish_id not in seen_dish_ids:
                        all_recommendations.append((similar_dish_name, similar_dish_id))
                        seen_dish_ids.add(similar_dish_id)
                        
                        if len(all_recommendations) >= max_recommendations:
                            return all_recommendations[:max_recommendations]
            
            return all_recommendations
            
        except Exception as e:
            print(f"Error generating personalized recommendations: {str(e)}")
            return []

    @classmethod
    def save(cls, rec, filepath):
        data = {
            'item_model': rec.item_model,
            'user_model': rec.user_model,
            'X': rec.X,
            'dish_names': rec.dish_names,
            'dish_names_inv': rec.dish_names_inv,
            'dish_mapper': rec.dish_mapper,
            'dish_inv_mapper': rec.dish_inv_mapper,
            'user_mapper': rec.user_mapper,
            'user_inv_mapper': rec.user_inv_mapper,
            'user_df': rec.user_df,
            'dish_user_review_df': rec.dish_user_review_df,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        recommender = cls()
        recommender.item_model = data['item_model']
        recommender.user_model = data['user_model']
        recommender.X = data['X']
        recommender.dish_names = data['dish_names']
        recommender.dish_names_inv = data['dish_names_inv']
        recommender.dish_mapper = data['dish_mapper']
        recommender.dish_inv_mapper = data['dish_inv_mapper']
        recommender.user_mapper = data['user_mapper']
        recommender.user_inv_mapper = data['user_inv_mapper']
        recommender.user_df = data['user_df']
        recommender.dish_user_review_df = data['dish_user_review_df']

        return recommender