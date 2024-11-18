import os
import pickle
from pathlib import Path

from django.core.management.base import BaseCommand
from django.conf import settings

import w_rc_sys.recommendation as rec 
from utils.recommender import DishRecommender

class Command(BaseCommand):
    help = 'Trains and saves the dish recommendation model'

    def handle(self, *args, **options):
        self.stdout.write('Training recommendation model...')

        save_path = Path(settings.BASE_DIR) / 'w_rc_sys' / 'ml_models' / 'dish_recommender.pkl'
        recommender = DishRecommender.save(rec, save_path)
        self.stdout.write(self.style.SUCCESS('Successfully trained model'))