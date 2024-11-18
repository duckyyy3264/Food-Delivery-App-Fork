import pandas as pd
from pandas import DataFrame
from pathlib import Path
import numpy as np
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import ast
import random
from pathlib import Path
def extract_id(obj):
    try:
        if isinstance(obj, str):
            if not obj.strip().startswith('{'):
                return str(obj)
            _dict = ast.literal_eval(obj)
            return _dict.get('id')
        elif isinstance(obj, dict):
            return obj.get('id')
        return str(obj) if obj is not None else None
    except:
        return None

def convert_object_to_string(df):
    object_columns = df.select_dtypes(include=['object']).columns
    df[object_columns] = df[object_columns].astype('string')
    return df

def process_df(df):
    object_columns = df.select_dtypes(include=['object']).columns
    print(object_columns)
    df[object_columns] = df[object_columns].map(extract_id)
    return df



import os
from django.conf import settings
def c_path(path, output_folder="data"):
    return Path(output_folder, path)

order_path = os.path.join(settings.BASE_DIR, 'w_rc_sys', 'data','order.csv')
restaurant_path = os.path.join(settings.BASE_DIR, 'w_rc_sys', 'data','restaurant.csv')
dish_category = os.path.join(settings.BASE_DIR, 'w_rc_sys', 'data','dish_category.csv')
dish_review_path = os.path.join(settings.BASE_DIR, 'w_rc_sys', 'data','dish_review.csv')

user_path = os.path.join(settings.BASE_DIR, 'w_rc_sys', 'data','user.csv')
profile_path = os.path.join(settings.BASE_DIR, 'w_rc_sys', 'data','profile.csv')
dish_path = os.path.join(settings.BASE_DIR, 'w_rc_sys', 'data','dish.csv')




order_df = pd.read_csv(order_path, index_col=0)
order_df = process_df(order_df)

profile_df = pd.read_csv(profile_path, index_col=0)
profile_df = process_df(profile_df)

restaurant_df = pd.read_csv(restaurant_path, index_col=0)
restaurant_df = process_df(restaurant_df)

dish_category_df = pd.read_csv(dish_category, index_col=0)
dish_category_df = process_df(dish_category_df)

dish_review_df = pd.read_csv(dish_review_path, index_col=0)
dish_review_df = process_df(dish_review_df)

user_df = pd.read_csv(user_path, index_col=0)
user_df = process_df(user_df)

dish_df = pd.read_csv(dish_path, index_col=0)
dish_df = process_df(dish_df)


dish_review_df = process_df(dish_review_df)
dish_review_df

dish_review_df.head()


dish_df = pd.read_csv(dish_path, index_col=0)
dish_df.drop(dish_df.iloc[:, 7:], inplace=True, axis=1)
dish_df.drop(columns=['image', 'description', 'rating', 'category'], inplace=True)
dish_df.head()

user_df.head()

user_dish_review_df = pd.merge(user_df, dish_review_df, left_on='id', right_on='user')

user_dish_review_df.head()

dish_user_review_df = pd.merge(user_dish_review_df, dish_df, left_on='dish', right_on='id')
dish_user_review_df.rename(columns={'name_x': 'user_name', 'name_y': 'dish_name'}, inplace=True)
dish_user_review_df.drop(columns=[
    'phone_number', 
    'avatar', 
    'title',
    'content', 
    'created_at', 
    'updated_at', 
    'total_likes',
    'total_replies',
    'is_liked',
    'images',
    'replies',
    'original_price',
    'discount_price'
], inplace=True)




dish_user_review_df.info()




dish_user_review_df.shape



user_df.head()



from scipy.sparse import csr_matrix

def create_matrix(df):
    N = len(df['user'].unique())
    M = len(df['dish'].unique())
      
    user_mapper = dict(zip(np.unique(df["user"]), list(range(N))))
    dish_mapper = dict(zip(np.unique(df["dish"]), list(range(M))))
      
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["user"])))
    dish_inv_mapper = dict(zip(list(range(M)), np.unique(df["dish"])))
      
    user_index = [user_mapper[i] for i in df['user']]
    dish_index = [dish_mapper[i] for i in df['dish']]
  
    X = csr_matrix((df["rating"].values, (dish_index, user_index)), 
                    shape=(M, N))      
    return X, user_mapper, dish_mapper, user_inv_mapper, dish_inv_mapper



X, user_mapper, dish_mapper, user_inv_mapper, dish_inv_mapper = create_matrix(dish_user_review_df)


from sklearn.neighbors import NearestNeighbors
item_model = NearestNeighbors(n_neighbors=5, algorithm="brute", metric='cosine')
item_model.fit(X)



dish_names = dict(zip(dish_user_review_df['dish'], dish_user_review_df['dish_name']))


dish_names_inv = {v: k for k, v in dish_names.items()}


dish_user_review_df.loc[dish_user_review_df['dish_name']  ==  'Cookie thank particularly']



def find_similar_dishes(dish_id=None, dish_name=None, n_similar_dishes=5):
    try:
        if dish_id is None and dish_name is None:
            raise ValueError("Must provide either dish_id or dish_name")
        
        if dish_id is None:
            dish_id = dish_names_inv.get(dish_name)
            if dish_id is None:
                raise KeyError(f"Dish name '{dish_name}' not found")
            
        dish_id = str(dish_id)
        dish_ind = dish_mapper[dish_id]
        dish_vec = X[dish_ind].reshape(1, -1)

        item_model = NearestNeighbors(n_neighbors=n_similar_dishes + 1, algorithm="brute", metric='cosine')
        item_model.fit(X)
        neighbor_indices = item_model.kneighbors(dish_vec, 
                                            return_distance=False).flatten()
        
        neighbor_indices = neighbor_indices[1:]
        similar_dishes = []

        print(f"Since you eat ----- {dish_name if dish_name else dish_names[dish_id]} -- {dish_id}:")

        for idx in neighbor_indices:
            dish_id = dish_inv_mapper[idx]
            dish_name = dish_names[dish_id]
            similar_dishes.append((dish_name, dish_id))
        
        return similar_dishes
        
    except Exception as e:
        print(f"Error finding similar dishes: {str(e)}")
        return []




import random

name_ = random.choice(list(dish_names_inv.keys()))
find_similar_dishes(dish_name=name_)




from sklearn.neighbors import NearestNeighbors
user_model = NearestNeighbors(n_neighbors=5, algorithm="brute", metric='cosine')
user_model.fit(X.T)




def find_similar_dishes_by_user(user_id=None, user_name=None, n_similar_users=5, n_dishes_for_user=5):
    if user_id is None and user_name is None:
        raise ValueError("Must provide either user_id or user_name")
        
    if user_id is None:
        user_matches = user_df[user_df['name'] == user_name]
        if len(user_matches) == 0:
            raise KeyError(f"User name '{user_name}' not found")
        user_id = str(user_matches.index[0])
    
    user_id = str(user_id)
    
    if user_id not in user_mapper:
        raise KeyError(f"User ID '{user_id}' not found in ratings data")
    user_idx = user_mapper[user_id]
    
    X_user = X.T
    
    user_vector = X_user[user_idx].reshape(1, -1)
    
    distances, indices = user_model.kneighbors(user_vector, return_distance=True)
    similar_user_indices = indices.flatten()[1:]  
    
    recommended_dishes = []
    user_rated_dishes = set(dish_user_review_df[dish_user_review_df['user'] == user_id]['dish'])
    
    for similar_user_idx in similar_user_indices:
        similar_user_id = user_inv_mapper[similar_user_idx]
        similar_user_ratings = dish_user_review_df[dish_user_review_df['user'] == similar_user_id]
        
        potential_recommendations = similar_user_ratings[
            (similar_user_ratings['rating'] >= 4) & 
            (~similar_user_ratings['dish'].isin(user_rated_dishes))
        ]
        
        for _, row in potential_recommendations.iterrows():
            dish_id = row['dish']
            dish_name = dish_names[dish_id]
            if (dish_name, dish_id) not in recommended_dishes:
                recommended_dishes.append((dish_name, dish_id))
                
            if len(recommended_dishes) >= n_dishes_for_user:
                break
                
        if len(recommended_dishes) >= n_dishes_for_user:
            break
    
    print("Based on similar users' highly rated dishes:\n")
    
    return recommended_dishes[:n_dishes_for_user]




random_user_id = random.choice(list(user_inv_mapper.values()))
print(random_user_id)
recommendations = find_similar_dishes_by_user(
    user_id=random_user_id,
    n_similar_users=1,
    n_dishes_for_user=2
)
recommendations
    

