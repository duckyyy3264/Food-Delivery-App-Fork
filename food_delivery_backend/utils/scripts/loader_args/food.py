import random
import os
from faker import Faker
from django.conf import settings
from django.db import transaction

from account.models import User
from food.models import (
    Dish,
    DishLike, 
    DishCategory, 
    DishOption,
    DishOptionItem
)
from restaurant.models import Restaurant
from review.models import (
    DishReview,
    DishReviewLike,
)
from utils.function import (
    load_intermediate_model,
    load_one_to_many_model,
    load_normal_model,
)
from utils.scripts.data import (
    dish_categories,
    category_options
)
from utils.decorators import script_runner

fake = Faker()

MODEL_MAP = {
    'dish': Dish,
    'dish_like': DishLike,
    'dish_category': DishCategory,
    'dish_option': DishOption,
    'dish_option_item': DishOptionItem,
    'dish_review': DishReview,
    'dish_review_like': DishReviewLike,
}

@script_runner(MODEL_MAP)
@transaction.atomic
def load_food(
    max_categories=30, 
    max_dishes_per_category=150,
    max_likes_per_dish=50,
    max_reviews_per_dish=25,
    max_review_likes_per_dish=25,
    models_to_update=None,
    map_queryset=None,
    action=None,
):
    if DishCategory in models_to_update:
        category_list = []
        for _x in dish_categories:
            category_data = {
                "name": _x.get('name'),
                "description": fake.word() + fake.text(max_nb_chars=200),
                "image": "food/category/" + _x.get('image'),
            }
            category, created = DishCategory.objects.update_or_create(
                name=_x.get('name'),
                defaults=category_data
            )
            category_list.append(category)
            print(f"\tSuccessfully {'created' if created else 'updated'} Dish Category: {category}")
    
    # if Dish in models_to_update:
    #     dish_list = []
    #     category_list = map_queryset.get(DishCategory)
    #     for category in category_list:
    #         category_name = category.name.lower().replace(' ', '_')
    #         category_folder_path = os.path.join(settings.MEDIA_ROOT, f"food/{category_name}")
            
    #         if not os.path.exists(category_folder_path):
    #             print(f"Category folder not found: {category_folder_path}")
    #             continue
            
    #         image_files = os.listdir(category_folder_path)
    #         if not image_files: continue
            
    #         dish_list += load_one_to_many_model(
    #             model_class=Dish,
    #             primary_field='category',
    #             primary_objects=category_list,
    #             max_related_count=min(max_dishes_per_category, len(image_files)),
    #             attributes={
    #                 "name": lambda name=category.name: f"{name} {fake.word()} {fake.word()}",
    #                 "description": lambda: fake.text(max_nb_chars=200),
    #                 "original_price": lambda: fake.pydecimal(left_digits=2, right_digits=2, positive=True, min_value=10, max_value=100),
    #                 "discount_price": lambda: fake.pydecimal(left_digits=2, right_digits=2, positive=True, min_value=5, max_value=50),
    #                 "image": lambda image_files=image_files:  f"food/{category_name}/{image_files.pop(random.randint(0, len(image_files) - 1))}" if len(image_files) > 0 else "",
    #             },
    #             action=action
    #         )

    if Dish in models_to_update:
        dish_list = []
        category_list = map_queryset.get(DishCategory)
        
        for category in category_list:
            category_name = category.name.lower().replace(' ', '_')
            category_folder_path = os.path.join(settings.MEDIA_ROOT, f"food/{category_name}")
            
            if not os.path.exists(category_folder_path):
                print(f"Category folder not found: {category_folder_path}")
                continue
            
            image_files = os.listdir(category_folder_path)
            if not image_files:
                continue
            
            max_dishes = min(max_dishes_per_category, len(image_files))
            
            if action == 'delete':
                Dish.objects.filter(category=category).delete()
            
            existing_dishes = list(Dish.objects.filter(category=category)) if action == 'update' else []
            
            category_temp_min, category_temp_max = {
                "Burger": (20, 30),      
                "Taco": (25, 35),        
                "Burrito": (15, 25),     
                "Drink": (30, 40),       
                "Pizza": (20, 30),       
                "Donut": (15, 25),       
                "Salad": (25, 35),       
                "Pho": (10, 20),         
                "Sandwich": (15, 25),    
                "Pasta": (15, 25),       
                "Ice Cream": (30, 40),   
                "Rice": (20, 30),        
                "Takoyaki": (15, 25),    
                "Fruit": (25, 35),       
                "Hot Dog": (20, 30),     
                "Goi Cuon": (25, 35),    
                "Cookie": (10, 20),      
                "Pudding": (10, 20),     
                "Banh Mi": (20, 30),     
                "Dumpling": (10, 20),    
            }.get(category.name, (15, 25))  

            for i in range(max_dishes):
                if len(image_files) == 0:
                    break
                
                image_file = image_files.pop(random.randint(0, len(image_files) - 1))
                
                # Generate a random environmental temperature within the suitable range
                optimal_temp = round(fake.pyfloat(
                    left_digits=2, right_digits=2, positive=True, 
                    min_value=category_temp_min, max_value=category_temp_max
                ), 2)
                
                temp_tolerance = round(fake.pyfloat(
                    left_digits=1, right_digits=2, positive=True, 
                    min_value=2, max_value=10  # Larger tolerances for broader enjoyment
                ), 2)
                
                if action == 'update' and i < len(existing_dishes):
                    dish = existing_dishes[i]
                    dish.name = f"{category.name} {fake.word()} {fake.word()}"
                    dish.description = fake.text(max_nb_chars=200)
                    dish.original_price = fake.pydecimal(left_digits=2, right_digits=2, positive=True, min_value=10, max_value=100)
                    dish.discount_price = fake.pydecimal(left_digits=2, right_digits=2, positive=True, min_value=5, max_value=50)
                    dish.image = f"food/{category_name}/{image_file}" if image_file else ""
                    dish.optimal_temp = optimal_temp
                    dish.temp_tolerance = temp_tolerance
                else:
                    dish = Dish(
                        category=category,
                        name=f"{category.name} {fake.word()} {fake.word()}",
                        description=fake.text(max_nb_chars=200),
                        original_price=fake.pydecimal(left_digits=2, right_digits=2, positive=True, min_value=10, max_value=100),
                        discount_price=fake.pydecimal(left_digits=2, right_digits=2, positive=True, min_value=5, max_value=50),
                        image=f"food/{category_name}/{image_file}" if image_file else "",
                        optimal_temp=optimal_temp,
                        temp_tolerance=temp_tolerance,
                    )
                
                dish.save()
                dish_list.append(dish)

    # if Dish in models_to_update:
    #     dish_list = []
    #     category_list = map_queryset.get(DishCategory)

    #     for category in category_list:
    #         category_name = category.name.lower().replace(' ', '_')
    #         category_folder_path = os.path.join(settings.MEDIA_ROOT, f"food/{category_name}")

    #         if not os.path.exists(category_folder_path):
    #             print(f"Category folder not found: {category_folder_path}")
    #             continue

    #         image_files = os.listdir(category_folder_path)
    #         if not image_files:
    #             continue

    #         max_dishes = min(max_dishes_per_category, len(image_files))

    #         def get_dish_attributes():
    #             image_file = image_files.pop(random.randint(0, len(image_files) - 1)) if len(image_files) > 0 else ""
    #             return {
    #                 "name": f"{category.name} {fake.word()} {fake.word()}",
    #                 "description": fake.text(max_nb_chars=200),
    #                 "original_price": fake.pydecimal(left_digits=2, right_digits=2, positive=True, min_value=10, max_value=100),
    #                 "discount_price": fake.pydecimal(left_digits=2, right_digits=2, positive=True, min_value=5, max_value=50),
    #                 "image": f"food/{category_name}/{image_file}" if image_file else ""
    #             }
            
    #         dish_list += load_normal_model(
    #             model_class=Dish,
    #             max_items=max_dishes,
    #             oto_attribute={"category": category},
    #             attributes=get_dish_attributes, 
    #             action=action
    #         )
            # for dish in map_queryset.get(Dish):
            #     if category.name in category_options:
            #         options = category_options[category.name]
            #         for option_type, option_list in options.items():
            #             dish_option, _ = DishOption.objects.update_or_create(dish=dish, name=option_type)
                        
            #             for option_name in option_list:
            #                 DishOptionItem.objects.update_or_create(
            #                     option=dish_option,
            #                     name=option_name,
            #                     price=float(fake.pydecimal(left_digits=1, right_digits=2, positive=True, min_value=0.5, max_value=10.0))
            #                 )
            #     print(f"\tSuccessfully created or updated Dish: {dish}, {dish.image}\n")

            for restaurant in map_queryset.get(Restaurant):
                for category in restaurant.categories.all():
                    tmp = list(category.dishes.all())
                    for _ in range(random.randint(1, max_dishes_per_category)):
                        if not tmp: break
                        max_loop = 100
                        dish = random.choice(tmp)
                        while dish.restaurant == None and max_loop:
                            dish.restaurant = restaurant                
                            dish = random.choice(tmp)
                            max_loop -= 1
                        dish.save()
                        print(f"\tSuccessfully added Dish: {dish} to Restaurant: {restaurant}")

    # if DishReview in models_to_update:
    #     user_list = list(User.objects.all())
    #     load_intermediate_model(
    #         model_class=DishReview,
    #         primary_field='dish',
    #         related_field='user',
    #         primary_objects=dish_list,
    #         related_objects=user_list,
    #         max_items=max_reviews_per_dish,
    #         attributes={
    #             'rating': lambda: fake.random_int(min=1, max=5),
    #             'title': lambda: fake.sentence(nb_words=6),
    #             'content': lambda: fake.text(max_nb_chars=200)
    #         },
    #         action=action
    #     )

    # if DishReviewLike in models_to_update:
    #     user_list = list(User.objects.all())
    #     review_list = map_queryset.get(DishReview)
    #     load_intermediate_model(
    #         model_class=DishReviewLike,
    #         primary_field='review',
    #         related_field='user',
    #         primary_objects=review_list,
    #         related_objects=user_list,
    #         max_items=max_review_likes_per_dish,
    #         action=action,
    #     )

    if DishLike in models_to_update:
        user_list = list(User.objects.all())
        dish_list = map_queryset.get(Dish)
        load_intermediate_model(
            model_class=DishLike,
            primary_field='dish',
            related_field='user',
            primary_objects=dish_list,
            related_objects=user_list,
            max_items=max_likes_per_dish,
            action=action,
        )


def run(*args):
    load_food(*args)


# """
# follow my instruction when i run the script if no arguments are specified it will run all
# the first argument will be update or update_with_del if update_with_del it will delete data in that model
# the second argument specifies which model you can load that model and lowercase to match the argument
# if no attributes are specified it will update all else that specific argument
# each model which need the result of previous will list(Model.objects.all())
# (you see the arguments in a model in a dict you split to validate if the attribute argument
# update each case will just delete that model
# if update attribute it will not delete)

# modify this no attribute argument just model argument
# if no arguments specified will delete all models and load again
# else if first argument is update and no second argument is update it will not delete instead it update all models
# else if first argument is update and second argument is (can be list model) it will not delete that model instead it update
# else if first argument is delete and second argument (can be list model) it will delete those models 
# """