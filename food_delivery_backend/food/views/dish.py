from rest_framework import (
    response, 
    viewsets, 
    status,
    mixins,
)
from django.conf import settings
from rest_framework.decorators import action

from food.models import (
    Dish,
    DishImage
)
from review.models import DishReview


from account.serializers import BasicUserSerializer
from food.serializers import (
    DishSerializer, 
    SuggestedDishSerializer,
    DetailDishSerializer, 
    CreateUpdateDishSerializer,

    DishImageSerializer,
    DishLikeSerializer,
    DishInCartOrOrderSerializer
)
from review.serializers import (
    DishReviewSerializer,
    DishReviewLikeSerializer,
)
from review.mixins import ReviewFilterMixin
from utils.views import ManyRelatedViewSet
from utils.pagination import CustomPagination
from utils.mixins import (
    DefaultGenericMixin,
    DynamicFilterMixin,
)
from utils.recommender import DishRecommender

class DishPagination(CustomPagination):
    def __init__(self):
        super().__init__()
        self.page_size_query_param = 'dish_page_size'

class DishViewSet(DefaultGenericMixin, DynamicFilterMixin, ManyRelatedViewSet, ReviewFilterMixin):
    queryset = Dish.objects.filter(is_disabled=False, restaurant__isnull=False)
    serializer_class = DishSerializer
    pagination_class = CustomPagination
    mapping_serializer_class = {
        'suggested_dish': SuggestedDishSerializer,
    }
    many_related_serializer_class = {
        'retrieve': DetailDishSerializer,
        'create': CreateUpdateDishSerializer,
        'update': CreateUpdateDishSerializer,
        'liked_by_users': BasicUserSerializer,
        'reviewed_by_users': BasicUserSerializer,
        'likes': DishLikeSerializer,
        'dish_reviews': DishReviewSerializer,
        'in_carts_or_orders': DishInCartOrOrderSerializer,
    }   
    # many_related = {
    #     'liked_by_users': {
    #         'action': (['GET'], 'liked-by-users'),
    #         'queryset': lambda instance: instance.liked_by_users.all(),
    #         'serializer_class': UserAbbrSerializer,
    #     },
    #     'reviewed_by_users': {
    #         'action': (['GET'], 'reviewed-by-users'),
    #         'queryset': lambda instance: instance.reviewed_by_users.all(),
    #         'serializer_class': UserAbbrSerializer,
    #     }
    # }

    def get_object(self):
        if self.action == 'dish_reviews':
            return ReviewFilterMixin.get_object(self)
        return super().get_object()
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        if self.action == "liked_by_users" or self.action == "reviewed_by_users":
            context.update({"many": True})
        return context
    
    def get_recommender(self):
        if not hasattr(self, '_recommender'):
            self._recommender = DishRecommender.load(settings.RECOMMENDER_MODEL_PATH)
        return self._recommender

    @action(detail=False, methods=['GET'], url_path='suggested-dish')
    def suggested_dish(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        flag = request.query_params.get('flag', 'temperature')
        temperature = request.query_params.get('temperature', None)
        w_name = request.query_params.get('w_name', "").lower()
        p_name = request.query_params.get('p_name', "").lower()
        dish_id = request.query_params.get('id', None)

        try:
            if dish_id is not None:
                try:
                    base_dish = queryset.get(id=dish_id)
                    recommender = self.get_recommender()
                    similar_dishes = recommender.find_similar_dishes(dish_id=base_dish.id, k=5)
                    
                    similar_dish_ids = [dish_id for _, dish_id in similar_dishes]
                    
                    queryset = queryset.filter(id__in=similar_dish_ids)
                    
                except queryset.model.DoesNotExist:
                    return response.Response(
                        {"error": "Dish not found."}, 
                        status=status.HTTP_404_NOT_FOUND
                    )
            
            elif flag == 'temperature':
                if temperature is None:
                    return response.Response(
                        {"error": "Temperature value is required."}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                temperature = float(temperature)
                queryset = [
                    dish for dish in queryset 
                    if dish.calculate_suitability_score(temperature) > 0.5 and w_name in dish.name.lower()
                ]
                queryset.sort(
                    key=lambda x: x.calculate_suitability_score(temperature), 
                    reverse=True
                )
                
            elif flag == 'preferences':
                recommender = self.get_recommender()
                n_similar_users = int(request.query_params.get('n_similar_users', 5))
                n_dishes_for_user = int(request.query_params.get('n_dishes_for_user', 20))
                n_similar_per_dish = int(request.query_params.get('n_similar_per_dish', 5))
                max_recommendations = int(request.query_params.get('max_recommendations', 100))
                
                if not self.request.user.is_authenticated:
                    # base_dishes = [
                    #     dish.dish for dish in DishReview.objects.filter(
                    #         rating__in=[4, 5]
                    #     ).order_by('-rating')[0:10]
                    # ]
                    # suggested_dishes = []
                    # for dish in base_dishes:
                    #     similar_dishes = recommender.find_similar_dishes(dish_id=dish.id, k=10)
                    #     suggested_dishes.extend(similar_dishes)
                    import random
                    from account.models import User
                    suggested_dishes = recommender.find_personalized_recommendations(
                        user_id=User.objects.all()[25].id,
                        n_similar_users=n_similar_users,
                        n_dishes_for_user=n_dishes_for_user,
                        n_similar_per_dish=n_similar_per_dish,
                        max_recommendations=max_recommendations,
                    )
                else:
                    suggested_dishes = recommender.find_personalized_recommendations(
                        user_id=self.request.user.id,
                        n_similar_users=n_similar_users,
                        n_dishes_for_user=n_dishes_for_user,
                        n_similar_per_dish=n_similar_per_dish,
                        max_recommendations=max_recommendations,
                    )
                
                unique_suggestions = [dish_id for _, dish_id in suggested_dishes]                
                queryset = queryset.filter(id__in=unique_suggestions, name__icontains=p_name)

                # recommender = self.get_recommender()
                # if not self.request.user.is_authenticated:
                #     from review.models import DishReview
                #     base_dishes = [
                #         _dish.dish for _dish in DishReview.objects.filter(
                #             rating__in=[4, 5]
                #         ).order_by('-rating')[0: 10]
                #     ]
                # else:
                #     base_dishes = [
                #         _dish.dish for _dish in self.request.user.dish_reviews.filter(
                #             rating__in=[4, 5]
                #         ).order_by('-rating')
                #     ]
                
                # suggested_dishes = []
                # for dish in base_dishes:
                #     similar_dishes = recommender.find_similar_dishes(dish_id=dish.id, k=10)
                #     suggested_dishes.extend(similar_dishes)

                # seen = set()
                # unique_suggestions = []
                # for name, dish_id in suggested_dishes:
                #     if dish_id not in seen:
                #         seen.add(dish_id)
                #         unique_suggestions.append(dish_id)
                
                # queryset = queryset.filter(id__in=unique_suggestions, name__icontains=p_name)

            if flag in ['temperature', 'preferences']:
                page = self.paginate_queryset(queryset)
                if page is not None:
                    serializer = self.get_serializer(page, many=True)
                    return self.get_paginated_response(serializer.data)
                
                serializer = self.get_serializer(queryset, many=True)
                return response.Response(serializer.data, status=status.HTTP_200_OK)
                
            return response.Response(
                {"error": "Invalid flag value."}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        except ValueError:
            return response.Response(
                {"error": "Invalid temperature value. Must be a number."}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return response.Response(
                {"error": f"An unexpected error occurred: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class DishImageViewSet(
    DefaultGenericMixin, 
    mixins.ListModelMixin, 
    mixins.RetrieveModelMixin, 
    mixins.CreateModelMixin, 
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    queryset = DishImage.objects.all()
    serializer_class = DishImageSerializer
    pagination_class = CustomPagination
    