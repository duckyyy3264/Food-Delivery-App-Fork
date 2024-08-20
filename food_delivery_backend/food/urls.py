# food/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    DishOptionViewSet, DishOptionItemViewSet,
    DishCategoryViewSet, 
    DishLikeViewSet, 
    DishViewSet
)

router = DefaultRouter()
router.register(r'dish-option', DishOptionViewSet)
router.register(r'dish-option-item', DishOptionItemViewSet)
router.register(r'dish-category', DishCategoryViewSet)
router.register(r'dish-like', DishLikeViewSet)
router.register(r'dish', DishViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
